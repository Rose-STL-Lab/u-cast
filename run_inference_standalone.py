#!/usr/bin/env python3
"""
Standalone ERA5 weather forecasting inference script for U-Cast.

Runs autoregressive ensemble inference with a trained DhariwalUNet checkpoint,
producing an xarray Dataset of forecasts. Optionally scores predictions with
area-weighted CRPS and ensemble-mean RMSE (--score) and logs to W&B (--wandb-project).

Install:
    pip install torch xarray zarr einops tqdm pyyaml huggingface_hub wandb gcsfs

Files needed:
    1. Checkpoint (.ckpt) — local path or ``hf:<repo>/<file>`` to download from HuggingFace Hub (default)
    2. ERA5 data directory — local path or GCS (gs://weatherbench2/datasets/era5; default)
    3. Normalization statistics — included in this repo at ``data/stats/`` (used by default)

Quick start (download checkpoint from HuggingFace, use GCS data):

    python run_inference_standalone.py \
        --ckpt-path hf:salvaRC/u-cast/ucast.ckpt \
        --data-dir gs://weatherbench2/datasets/era5 \
        --ic-start-dates 2020-01-01

With local data, ensembling, more initial conditions, and scoring & logging to Weights & Biases:

    python run_inference_standalone.py \
        --ckpt-path hf:salvaRC/u-cast/ucast.ckpt \
        --data-dir /path/to/era5/data \
        --prediction-horizon 30 \
        --ensemble-size 10 \
        --ic-start-dates 2020-01-01 2020-04-01 2020-07-01 2020-10-01 \
        --score --wandb-project=ERA5-Eval

Deep ensemble (multiple checkpoints, members split evenly across models):

    python run_inference_standalone.py \
        --ckpt-path hf:salvaRC/u-cast/ucast.ckpt \
        --ckpt-paths hf:salvaRC/u-cast/ucast_de2.ckpt hf:salvaRC/u-cast/ucast_de3.ckpt hf:salvaRC/u-cast/ucast_de4.ckpt \
        --data-dir /path/to/era5/data \
        --ensemble-size 10 \
        --ic-start-dates 2020-01-01

-> Forecasts are not saved to disk by default; add ``--output-path forecasts.nc`` to save.
-> Set the GPU device with e.g. ``--device cuda:1``.

Compute (1.5-degree, H200 GPU): ~10 sec/IC for 10 member ensemble, 30 steps. ~12 GB GPU RAM.

Note: Models were trained on data up to 2019. Degradation is expected for dates far
from the training period, especially at short lead times. 2020 is recommended.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import yaml
from einops import rearrange
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 1.  Circular-padding helpers
# ════════════════════════════════════════════════════════════════════════
_ORIGINAL_F_CONV2D = torch.nn.functional.conv2d


def _f_conv2d_circ_h(input, weight, padding, torch_func, **kwargs):
    if padding == 0:
        return torch_func(input, weight, **kwargs)
    input = F.pad(input, (0, 0, padding, padding), mode="circular")
    return torch_func(input, weight, **kwargs, padding=(0, padding))


def set_circular_height_padding():
    """Monkey-patch F.conv2d globally for circular height padding."""
    torch.nn.functional.conv2d = partial(_f_conv2d_circ_h, torch_func=_ORIGINAL_F_CONV2D)


# ════════════════════════════════════════════════════════════════════════
# 2.  Building blocks
# ════════════════════════════════════════════════════════════════════════
_silu = torch.nn.functional.silu


def _weight_init(shape, fan_in):
    return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, up=False, down=False):
        super().__init__()
        assert not (up and down)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.padding = 0

        if kernel == 0:
            self.weight = None
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel, kernel))
            self.bias = nn.Parameter(torch.randn(out_channels))
            self.padding = kernel // 2

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2)

        if self.weight is not None:
            x = F.conv2d(x, self.weight, padding=self.padding)
        if self.bias is not None:
            x = x.add_(self.bias.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):
    """Group norm that automatically picks num_groups based on min_channels_per_group.

    Stores weight/bias directly (not wrapped in nn.GroupNorm) so that the parameter
    names match the checkpoint exactly (e.g. ``norm0.weight`` not ``norm0.gn.weight``).
    """

    def __init__(self, num_channels, eps=1e-5, min_channels_per_group=4):
        super().__init__()
        num_groups = 32
        while num_channels % num_groups != 0 or num_channels // num_groups < min_channels_per_group:
            num_groups //= 2
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps
        )


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / math.sqrt(q.shape[1])).to(torch.float32))
            .softmax(dim=2)
            .to(q.dtype)
        )
        return w


# ════════════════════════════════════════════════════════════════════════
# 3.  UNetBlock & DhariwalUNet
# ════════════════════════════════════════════════════════════════════════
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0.1,
        eps=1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = (
            0 if not attention else (num_heads if num_heads is not None else out_channels // channels_per_head)
        )

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3)
        self.dropout = nn.Dropout(p=dropout)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down)

        # Self-attention
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1)

    def forward(self, x):
        orig = x
        x = self.conv0(_silu(self.norm0(x)))
        x = _silu(self.norm1(x))
        x = self.conv1(self.dropout(x))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)

        if self.num_heads:
            b, c, h, w = x.shape
            nh = self.num_heads
            B2, C2 = b * nh, c // nh
            q, k, v = self.qkv(self.norm2(x)).reshape(B2, C2, 3, -1).unbind(2)
            attn_w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", attn_w, v)
            a = rearrange(a, "(b nh) c (h w) -> b (nh c) h w", b=b, nh=nh, h=h, w=w)
            x = self.proj(a).add_(x)

        return x


class DhariwalUNet(nn.Module):
    """ADM U-Net for weather forecasting."""

    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels=320,
        channel_mult=(1, 2, 3, 4),
        num_blocks=3,
        attn_levels=(2, 3),
        channels_per_head=64,
        dropout=0.1,
    ):
        super().__init__()
        block_kwargs = dict(channels_per_head=channels_per_head, dropout=dropout)

        img_resolution = 240
        # ── Encoder ──
        self.enc = nn.ModuleDict()
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            use_attn = level in attn_levels
            level_channels = int(model_channels * mult)
            if level == 0:
                cout = level_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=in_channels, out_channels=cout, kernel=3)
            else:
                # Down block preserves channel count from the previous level
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = level_channels
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=use_attn, **block_kwargs
                )

        skips = [block.out_channels for block in self.enc.values()]
        channel_mult_dec = list(reversed(channel_mult))
        # ── Decoder ──
        self.dec = nn.ModuleDict()
        for dec_block_i, mult in enumerate(channel_mult_dec):
            level = len(channel_mult_dec) - 1 - dec_block_i  # reverse level order for decoder
            res = img_resolution >> level
            use_attn = level in attn_levels
            level_channels = int(model_channels * mult)
            if level == len(channel_mult) - 1:
                # Bottleneck: two blocks that preserve channel count before upsampling begins
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                # Up block preserves channel count from the previous decoder level
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = level_channels
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=use_attn, **block_kwargs
                )

        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)

    def forward(self, inputs, dynamical_condition=None, static_condition=None):
        # Concatenate conditional channels
        parts = [inputs]
        if dynamical_condition is not None:
            parts.append(dynamical_condition)
        if static_condition is not None:
            parts.append(static_condition)
        x = torch.cat(parts, dim=1) if len(parts) > 1 else inputs

        # Encoder
        skips = []
        for block in self.enc.values():
            x = block(x)
            skips.append(x)

        # Decoder
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                skip = skips.pop()
                # Handle mismatched spatial dims via bilinear interpolation
                if skip.shape[-2:] != x.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
                x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.out_conv(_silu(self.out_norm(x)))
        return x


# ════════════════════════════════════════════════════════════════════════
# 4.  Normalizer & Packer
# ════════════════════════════════════════════════════════════════════════
class StandardNormalizer(nn.Module):
    """Dict-based z-score normalizer with residual-std support."""

    def __init__(
        self,
        means: Dict[str, torch.Tensor],
        stds: Dict[str, torch.Tensor],
        names: List[str],
        std_residual: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.means = means
        self.stds = stds
        self.names = names
        self.std_residual = std_residual
        self.scale_res_to_normed = None
        if std_residual is not None:
            self.scale_res_to_normed = {k: std_residual[k] / stds[k] for k in names}

    def _apply(self, fn, recurse=True):
        super()._apply(fn)
        self.means = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.means.items()}
        self.stds = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.stds.items()}
        if self.std_residual is not None:
            self.std_residual = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.std_residual.items()}
            self.scale_res_to_normed = {
                k: fn(v) if torch.is_tensor(v) else v for k, v in self.scale_res_to_normed.items()
            }
        return self

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (t - self.means[k]) / self.stds[k] for k, t in tensors.items()}

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: t * self.stds[k] + self.means[k] for k, t in tensors.items()}

    def residual_to_normalized(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert from residual-normalized space to normal-normalized space."""
        return {k: t * self.scale_res_to_normed[k] for k, t in tensors.items()}


class NaNCleaner(torch.nn.Module):
    """Wraps a normalizer, filling NaNs (e.g. SST over land) before/after normalization."""

    def __init__(self, normalizer: StandardNormalizer, mask: np.ndarray, **vars_to_fill_values):
        super().__init__()
        self.normalizer = normalizer
        self.mask = torch.from_numpy(mask).bool()  # True where NaN
        self.vars_to_fill_values = vars_to_fill_values

    def _apply(self, fn, recurse=True):
        super()._apply(fn)
        self.mask = fn(self.mask)
        return self

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for var, fill_value in self.vars_to_fill_values.items():
            if var in tensors:
                tensors[var] = torch.where(self.mask.to(tensors[var].device), fill_value, tensors[var])
        return self.normalizer.normalize(tensors)

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors = self.normalizer.denormalize(tensors)
        for var, fill_value in self.vars_to_fill_values.items():
            if var in tensors:
                tensors[var] = torch.where(self.mask.to(tensors[var].device), fill_value, tensors[var])
        return tensors

    def residual_to_normalized(self, tensors):
        return self.normalizer.residual_to_normalized(tensors)


def _extract_variables(ds: xr.Dataset, names: List[str]) -> Dict[str, torch.Tensor]:
    """Extract variables from a dataset, handling 2D-flattened pressure-level names."""
    extracted = {}
    for name in names:
        if name in ds:
            extracted[name] = torch.as_tensor(ds[name].values, dtype=torch.float)
            continue
        parts = name.split("_")
        var_name = "_".join(parts[:-1])
        pressure_level = parts[-1]
        if pressure_level.isdigit():
            level = int(pressure_level)
            extracted[name] = torch.as_tensor(ds[var_name].sel(level=level).values, dtype=torch.float)
        else:
            raise ValueError(f"{name} not found and not in <var>_<level> format. Keys: {list(ds.keys())}")
    return extracted


def ensure_latitude_is_ascending(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Ensure latitude is sorted ascending."""
    if not (np.diff(ds.latitude.values) > 0).all():
        return ds.reindex(latitude=list(reversed(ds.latitude.values)))
    return ds


def setup_normalizer(
    stats_dir: str | Path, var_names: List[str], ds: xr.Dataset, device: str = "cpu"
) -> StandardNormalizer:
    """Load normalizer and wrap with NaNCleaner if sea_surface_temperature is present."""
    normalizer = load_normalizer(stats_dir, var_names)

    # Handle SST NaN cleaning
    min_ds = xr.open_dataset(Path(stats_dir) / "era5_min.nc")
    fill_value_sst = float(min_ds["sea_surface_temperature"].values.item())
    sst_sample = ds.isel(time=0)["sea_surface_temperature"].compute()

    # Ensure latitude is sorted ascending for the mask as well
    sst_sample = ensure_latitude_is_ascending(sst_sample)
    sst_sample = sst_sample.transpose("longitude", "latitude")
    nan_mask = np.isnan(sst_sample.values)
    nan_frac = nan_mask.sum() / nan_mask.size
    assert nan_frac < 0.5, f"Expected {nan_frac}<0.5. SST in data are corrupted."
    normalizer = NaNCleaner(normalizer, mask=nan_mask, sea_surface_temperature=fill_value_sst)

    normalizer.to(device)
    return normalizer


def load_normalizer(stats_dir: str, var_names: List[str]) -> StandardNormalizer:
    """Load a StandardNormalizer from statistics .nc files."""
    stats_dir = Path(stats_dir)
    mean_ds = xr.open_dataset(stats_dir / "era5_mean.nc")
    std_ds = xr.open_dataset(stats_dir / "era5_std.nc")
    std_res_path = stats_dir / "era5_residual_std.nc"
    assert std_res_path.exists(), f"Required file not found: {std_res_path}"
    std_res_ds = xr.open_dataset(std_res_path)

    means = _extract_variables(mean_ds, var_names)
    stds = _extract_variables(std_ds, var_names)
    std_res = _extract_variables(std_res_ds, var_names)
    return StandardNormalizer(means=means, stds=stds, names=var_names, std_residual=std_res)


def pack(tensors: Dict[str, torch.Tensor], names: List[str], dim: int = -3) -> torch.Tensor:
    """Stack named tensors along *dim*."""
    return torch.stack([tensors[n] for n in names], dim=dim)


def unpack(tensor: torch.Tensor, names: List[str], dim: int = -3) -> Dict[str, torch.Tensor]:
    """Split a stacked tensor back into a named dict."""
    return {n: tensor.select(dim, i) for i, n in enumerate(names)}


# ════════════════════════════════════════════════════════════════════════
# 5.  Forcing computation  (year/day progress sin/cos)
# ════════════════════════════════════════════════════════════════════════
_SEC_PER_DAY = 86400
_AVG_DAY_PER_YEAR = 365.24219


def compute_forcings(time_vals: np.ndarray, lon_vals: np.ndarray, spatial_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Compute forcing fields for given datetimes.
    Returns tensor of shape (T, 4, n_lon, n_lat).
    """
    seconds = time_vals.astype("datetime64[s]").astype(np.int64).astype(np.float64)

    year_progress = np.mod(seconds / _SEC_PER_DAY / _AVG_DAY_PER_YEAR, 1.0).astype(np.float32)
    yp_sin = np.sin(2 * np.pi * year_progress)
    yp_cos = np.cos(2 * np.pi * year_progress)

    day_frac = np.mod(seconds, _SEC_PER_DAY) / _SEC_PER_DAY
    lon_offset = np.deg2rad(lon_vals) / (2 * np.pi)
    day_progress = np.mod(day_frac[:, None] + lon_offset[None, :], 1.0).astype(np.float32)
    dp_sin = np.sin(2 * np.pi * day_progress)
    dp_cos = np.cos(2 * np.pi * day_progress)

    n_lon, n_lat = spatial_shape
    forcing = np.stack(
        [
            np.broadcast_to(yp_sin[:, None, None], (len(time_vals), n_lon, n_lat)),
            np.broadcast_to(yp_cos[:, None, None], (len(time_vals), n_lon, n_lat)),
            np.broadcast_to(dp_sin[:, :, None], (len(time_vals), n_lon, n_lat)),
            np.broadcast_to(dp_cos[:, :, None], (len(time_vals), n_lon, n_lat)),
        ],
        axis=1,
    ).astype(
        np.float32
    )  # (T, 4, n_lon, n_lat)
    return torch.from_numpy(forcing)


# ════════════════════════════════════════════════════════════════════════
# 6.  Path resolution (HuggingFace download support)
# ════════════════════════════════════════════════════════════════════════
def _resolve_path(path: str) -> str:
    """Resolve a path, downloading from HuggingFace Hub if it starts with ``hf:``."""
    if not path.startswith("hf:"):
        return path
    from huggingface_hub import hf_hub_download

    hf_path = path.replace("hf://", "").replace("hf:", "")
    repo_id, filename = hf_path.rsplit("/", 1)
    if filename.endswith((".ckpt", ".pt")):
        cache_dir = ".cache/models/"
    elif filename.endswith(".yaml"):
        cache_dir = ".cache/configs/"
    else:
        cache_dir = ".cache/data/"
    os.makedirs(cache_dir, exist_ok=True)
    log.info(f"Downloading from HuggingFace Hub: {repo_id}/{filename} → {cache_dir}")
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


# ════════════════════════════════════════════════════════════════════════
# 7.  Data loading
# ════════════════════════════════════════════════════════════════════════
def open_era5_zarr(data_dir: str, dataset_name: str) -> xr.Dataset:
    """Open the ERA5 zarr dataset from a local or GCS path."""
    data_dir = [data_dir] if isinstance(data_dir, str) else data_dir
    for d in data_dir:
        path = os.path.join(d, dataset_name)
        if os.path.isdir(path):
            return xr.open_zarr(path)
    # Try GCS
    if isinstance(data_dir[0], str) and data_dir[0].startswith("gs://"):
        import gcsfs

        path = os.path.join(data_dir[0], dataset_name)
        fs = gcsfs.GCSFileSystem(token="anon")
        return xr.open_zarr(fs.get_mapper(path), zarr_format=2)
    raise FileNotFoundError(f"Cannot find {dataset_name} in {data_dir}")


def load_static_conditions(ds: xr.Dataset, static_fields: List[str]) -> torch.Tensor:
    """Load and z-score static fields, unconditionally returning tensor of shape (n_fields, lon, lat)."""
    arrays = []
    for field in static_fields:
        data = ds[field].compute().values
        if data.ndim > 2:
            data = data[0]

        coord_dims = list(ds[field].dims)
        spatial_dims = [d for d in coord_dims if d != "time"]

        # Enforce lon, lat ordering
        if spatial_dims[0] in ["latitude", "lat"]:
            data = data.T

        arrays.append(data.astype(np.float32))

    stacked = np.stack(arrays, axis=0)
    mean = stacked.mean(axis=(-2, -1), keepdims=True)
    std = stacked.std(axis=(-2, -1), keepdims=True)
    return torch.from_numpy((stacked - mean) / std).float()


def extract_batch_for_ic(
    ds: xr.Dataset,
    ic_datetime: np.datetime64,
    var_names: List[str],
    static_cond: torch.Tensor,
    normalizer,
    window: int,
    prediction_horizon: int,
    hourly_resolution: int,
    device: str,
) -> Dict[str, Any]:
    t_start = ic_datetime - np.timedelta64((window - 1) * hourly_resolution, "h")
    total_steps = window + prediction_horizon

    # Build the exact timestamps we want at model temporal resolution.
    # This correctly handles datasets stored at a finer resolution than the model
    # (e.g., a 6-hourly dataset with a 12-hourly model: we select every 2nd step).
    target_times = np.array([t_start + np.timedelta64(i * hourly_resolution, "h") for i in range(total_steps)]).astype(
        ds.time.values.dtype
    )

    batch = ds.sel(time=target_times).load()
    time_values = batch.time.values
    log.info(f"Requested times: {target_times[:2]}...{target_times[-1]}")
    if len(time_values) < total_steps:
        raise ValueError(
            f"Not enough time steps. Requested {total_steps} at {hourly_resolution}h intervals "
            f"starting from {t_start}, but only {len(time_values)} matched."
        )

    dynamics_raw = {}
    var3d_cache = {}
    for var in var_names:
        if var in batch:
            data = batch[var].values
            dynamics_raw[var] = torch.from_numpy(data.astype(np.float32))
        else:
            parts = var.split("_")
            var3d = "_".join(parts[:-1])
            level = int(parts[-1])
            if var3d not in var3d_cache:
                var3d_cache[var3d] = batch[var3d].values
            level_idx = list(batch.level.values).index(level)
            dynamics_raw[var] = torch.from_numpy(var3d_cache[var3d][:, level_idx].astype(np.float32))

    lon_vals = batch.longitude.values
    lat_vals = batch.latitude.values
    n_lon, n_lat = len(lon_vals), len(lat_vals)

    # Strictly permute ONLY if the shape is currently (time, lat, lon)
    for var in dynamics_raw:
        t = dynamics_raw[var]
        if t.ndim == 3 and t.shape[1] == n_lat and t.shape[2] == n_lon:
            dynamics_raw[var] = t.permute(0, 2, 1)  # Safely flips to (T, lon, lat)

    spatial_shape = (n_lon, n_lat)
    forcing = compute_forcings(time_values, lon_vals, spatial_shape)
    dynamics_raw = {k: v.to(device) for k, v in dynamics_raw.items()}
    dynamics_normed = normalizer.normalize({k: v.clone() for k, v in dynamics_raw.items()})

    return dict(
        dynamics_raw=dynamics_raw,
        dynamics_normed=dynamics_normed,
        static_condition=static_cond,
        forcing=forcing,
        time_values=time_values,
        latitude=lat_vals,
        longitude=lon_vals,
    )


# ════════════════════════════════════════════════════════════════════════
# 8.  Inference loop
# ════════════════════════════════════════════════════════════════════════
def enable_inference_dropout(model: nn.Module):
    """Keep dropout layers in training mode (stochastic) while rest is eval."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_model_from_checkpoint(ckpt_path: str, cfg: dict, device: str = "cuda") -> DhariwalUNet:
    """
    Load the DhariwalUNet from a PL checkpoint, applying EMA weights.
    """
    model_cfg = cfg["model"]
    dm_cfg = cfg["datamodule"]
    module_cfg = cfg["module"]

    var_names = list(dm_cfg["input_vars"])
    window = dm_cfg["window"]
    n_vars = len(var_names)
    n_static = len(dm_cfg.get("static_fields", []))
    n_forcing = len(dm_cfg.get("forcing_fields", []))
    n_cond = n_static + n_forcing
    in_channels = n_vars * window + n_cond
    out_channels = len(dm_cfg["output_vars"])

    model = DhariwalUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_cfg["model_channels"],
        channel_mult=tuple(model_cfg["channel_mult"]),
        num_blocks=model_cfg["num_blocks"],
        attn_levels=tuple(model_cfg.get("attn_levels", [])),
        channels_per_head=model_cfg.get("channels_per_head", 64),
        dropout=model_cfg.get("dropout", 0.1),
    )

    # Load checkpoint
    log.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Extract EMA weights if available
    use_ema = module_cfg.get("use_ema", True)
    if use_ema:
        ema_prefix = "model_ema."
        ema_buffers = {k[len(ema_prefix) :]: v for k, v in state_dict.items() if k.startswith(ema_prefix)}
        # Build mapping from dotless EMA buffer name → model parameter name
        # The EMA stores parameters with dots removed from names and prefixed with "model."
        # We need to match them back to the actual model parameter names.
        # The checkpoint model params are stored as "model.<param_name>" in state_dict
        model_param_names = [k for k in state_dict.keys() if k.startswith("model.") and not k.startswith("model_ema.")]
        ema_name_to_param = {}
        for param_name in model_param_names:
            # Remove "model." prefix to get the actual param name inside DhariwalUNet
            short_name = param_name[len("model.") :]
            # The EMA stores names with dots replaced by nothing
            ema_key = short_name.replace(".", "")
            ema_name_to_param[ema_key] = short_name

        loaded_sd = {}
        for ema_key, ema_val in ema_buffers.items():
            if ema_key in ("decay", "num_updates"):
                continue
            if ema_key in ema_name_to_param:
                loaded_sd[ema_name_to_param[ema_key]] = ema_val

        missing, unexpected = model.load_state_dict(loaded_sd, strict=True)
        if missing:
            log.warning(f"Missing keys when loading EMA weights: {missing[:5]}... ({len(missing)} total)")
        if unexpected:
            log.warning(f"Unexpected keys: {unexpected[:5]}...")

    else:
        _load_model_weights(model, state_dict)

    model = model.to(device).eval()
    model.epoch = ckpt.get("epoch", None)
    enable_inference_dropout(model)

    return model


def _load_model_weights(model: DhariwalUNet, state_dict: dict):
    """Load non-EMA model weights from a PL checkpoint state_dict."""
    model_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model.") and not k.startswith("model_ema."):
            model_sd[k[len("model.") :]] = v
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing[:5]}... ({len(missing)} total)")


@torch.inference_mode()
def run_autoregressive_inference(
    model: DhariwalUNet,
    batch: Dict[str, Any],
    var_names: List[str],
    normalizer,
    window: int = 2,
    prediction_horizon: int = 15,
    ensemble_size: int = 5,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run autoregressive inference.

    Returns dict with:
      preds_raw: dict of {var: tensor(ensemble, horizon, H, W)} – denormalized
      targets_raw: dict of {var: tensor(horizon, H, W)} – denormalized
    """
    dynamics_normed = batch["dynamics_normed"]
    dynamics_raw = batch["dynamics_raw"]
    forcing = batch["forcing"].to(device)  # (T_total, 4, H, W)
    static_cond = batch["static_condition"].to(device)  # (n_static, H, W)

    # Prepare normalized dynamics as tensors on device: {var: (T, H, W)}
    dyn_normed = {k: v.to(device) for k, v in dynamics_normed.items()}
    dyn_raw = {k: v.clone() for k, v in dynamics_raw.items()}  # keep on CPU

    # Storage for predictions and targets across all lead times
    all_preds_raw = {var: [] for var in var_names}  # list of (ensemble, H, W) per lead time
    all_targets_raw = {var: [] for var in var_names}  # list of (H, W) per lead time

    pbar = tqdm(range(prediction_horizon), desc="Lead time", leave=True)

    # Pre-expand static condition for ensemble: (N, n_static, H, W)
    static_batch = static_cond.unsqueeze(0).expand(ensemble_size, -1, -1, -1)

    # Window frames from input data, expanded for ensemble; each member starts with the SAME IC.
    window_frames_normed = [
        {var: dyn_normed[var][w_i].unsqueeze(0).expand(ensemble_size, -1, -1) for var in var_names}
        for w_i in range(window)
    ]

    for lead_t in pbar:
        # ── Pack input tensor: (N, window*C, H, W) ──
        x_input = torch.cat([pack(f, var_names, dim=1) for f in window_frames_normed], dim=1)

        # Forcing comes from the target timestep (index window + lead_t).
        forcing_idx = window + lead_t  # target time's forcing
        dyn_cond = forcing[forcing_idx]  # (4, H, W)

        dyn_cond_batch = dyn_cond.unsqueeze(0).expand(ensemble_size, -1, -1, -1)  # (N, 4, H, W)

        # ── Forward pass ──
        pred_packed = model(
            x_input,
            dynamical_condition=dyn_cond_batch,
            static_condition=static_batch,
        )  # (N, C_out, H, W) – in residual-normalized space
        assert not torch.any(torch.isnan(pred_packed)), "Model output contains NaNs"

        # ── Post-process: residual → normalized → denormalized ──
        pred_dict = unpack(pred_packed, var_names, dim=1)  # {var: (N, H, W)}

        # Convert from residual-normalized to normal-normalized space
        pred_normed = normalizer.residual_to_normalized(pred_dict)

        # Add residual to each member's own last input frame (in normalized space)
        last_input_normed = window_frames_normed[-1]  # dict of {var: (N, H, W)}
        pred_full_normed = {
            var: pred_normed[var] + last_input_normed[var] for var in var_names  # (N, H, W) + (N, H, W)
        }

        # Denormalize predictions
        pred_denormed = normalizer.denormalize({k: v.clone() for k, v in pred_full_normed.items()})

        # Store predictions and extract targets
        for var in var_names:
            all_preds_raw[var].append(pred_denormed[var].cpu())  # (N, H, W)
            if forcing_idx < dyn_raw[var].shape[0]:
                all_targets_raw[var].append(dyn_raw[var][forcing_idx].cpu())  # (H, W)
            else:
                raise ValueError(
                    f"Target time index {forcing_idx} out of bounds for variable {var} with shape {dyn_raw[var].shape}"
                )

        # Shift window forward: each member uses its own prediction, so members diverge via dropout.
        window_frames_normed = window_frames_normed[1:] + [pred_full_normed]

    # Stack lead times
    preds_stacked = {var: torch.stack(all_preds_raw[var], dim=1) for var in var_names}
    # preds_stacked[var].shape = (ensemble, horizon, H, W)

    targets_stacked = {var: torch.stack(all_targets_raw[var], dim=0) for var in var_names}
    # targets_stacked[var].shape = (horizon, H, W)

    return dict(preds_raw=preds_stacked, targets_raw=targets_stacked)


def _distribute_members(total_members: int, n_models: int) -> List[int]:
    """Split ``total_members`` evenly across ``n_models``; extras go to first models."""
    base, remainder = divmod(total_members, n_models)
    model_members = [base + (1 if i < remainder else 0) for i in range(n_models)]
    if any([m == 0 for m in model_members]):
        raise ValueError(f"Ensemble size {total_members} too small for {n_models} models. Increase --ensemble-size")
    return model_members


def _run_deep_ensemble_all_ics(
    ckpt_paths: List[str],
    configs: List[dict],
    ic_dates: List[np.datetime64],
    ds,
    var_names: List[str],
    static_cond,
    normalizer,
    window: int,
    prediction_horizon: int,
    ensemble_size: int,
    hourly_resolution: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], List[xr.Dataset]]:
    """
    Deep-ensemble inference over all ICs with model-outer / IC-inner loop order.
    Returns:
        all_results: Per-IC list of dicts with ``preds_raw`` (ensemble_size, H_steps, H, W)
            and ``targets_raw`` (H_steps, H, W) — full ensemble concatenated.
        all_datasets: Per-IC xarray Datasets ready for merging.
    """
    n_models = len(ckpt_paths)
    members_per_model = _distribute_members(ensemble_size, n_models)

    # Accumulator: per_ic_preds[ic_idx][var] collects tensors from each model
    per_ic_preds: List[Dict[str, List[torch.Tensor]]] = [{var: [] for var in var_names} for _ in ic_dates]
    per_ic_targets: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(ic_dates)
    ic_batches: List[Optional[Dict]] = [None] * len(ic_dates)

    for model_idx, (ckpt_path, cfg, n_members) in enumerate(zip(ckpt_paths, configs, members_per_model)):
        log.info(f"\n[DeepEnsemble] Loading model {model_idx + 1}/{n_models}: " f"{ckpt_path} — {n_members} member(s)")
        model = load_model_from_checkpoint(ckpt_path, cfg, device=device)

        for ic_idx, ic_dt in enumerate(ic_dates):
            log.info(f"  IC {ic_idx + 1}/{len(ic_dates)}: {ic_dt}")

            if ic_batches[ic_idx] is None:
                ic_batches[ic_idx] = extract_batch_for_ic(
                    ds=ds,
                    ic_datetime=ic_dt,
                    var_names=var_names,
                    static_cond=static_cond,
                    normalizer=normalizer,
                    window=window,
                    prediction_horizon=prediction_horizon,
                    hourly_resolution=hourly_resolution,
                    device=device,
                )

            with torch.inference_mode():
                result = run_autoregressive_inference(
                    model=model,
                    batch=ic_batches[ic_idx],
                    var_names=var_names,
                    normalizer=normalizer,
                    window=window,
                    prediction_horizon=prediction_horizon,
                    ensemble_size=n_members,
                    device=device,
                )

            for var in var_names:
                per_ic_preds[ic_idx][var].append(result["preds_raw"][var])
            if per_ic_targets[ic_idx] is None:
                per_ic_targets[ic_idx] = result["targets_raw"]

        # Free GPU memory before loading the next model
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    # Concatenate per-model results into final ensemble for each IC
    all_results = []
    all_datasets = []
    for ic_idx, ic_dt in enumerate(ic_dates):
        preds = {var: torch.cat(per_ic_preds[ic_idx][var], dim=0) for var in var_names}
        results = dict(preds_raw=preds, targets_raw=per_ic_targets[ic_idx])
        all_results.append(results)
        ic_ds = build_xarray_dataset(results, ic_batches[ic_idx], ic_dt, hourly_resolution, var_names)
        all_datasets.append(ic_ds)

    return all_results, all_datasets


# ════════════════════════════════════════════════════════════════════════
# 9.  xarray Dataset construction
# ════════════════════════════════════════════════════════════════════════
def build_xarray_dataset(
    results: Dict[str, Any],
    batch: Dict[str, Any],
    ic_datetime: np.datetime64,
    hourly_resolution: int,
    var_names: List[str],
) -> xr.Dataset:
    """
    Build an xarray Dataset from inference results for a single IC.

    Coords:
      - init_time:  scalar datetime64
      - lead_time:  prediction_horizon values in hours (timedelta)
      - latitude, longitude: spatial coords
      - ensemble_member: ensemble index (for predictions only)
      - data_type: "predicted" or "target"

    Data vars: one per weather variable, with dims depending on data_type.
    """
    preds = results["preds_raw"]  # {var: (N, H_steps, lon, lat)}
    targets = results["targets_raw"]  # {var: (H_steps, lon, lat)}
    lat = batch["latitude"]
    lon = batch["longitude"]

    n_ens = preds[var_names[0]].shape[0]
    n_lead = preds[var_names[0]].shape[1]

    lead_hours = np.arange(1, n_lead + 1) * hourly_resolution
    lead_time = [np.timedelta64(int(h), "h") for h in lead_hours]

    data_vars = {}
    for var in var_names:
        pred_np = preds[var].cpu().numpy()  # (N, H, lon, lat)
        targ_np = targets[var].cpu().numpy()  # (H, lon, lat)
        # Combine into (2, N_or_1, H, lon, lat) – but data_type dim makes shapes differ.
        # Instead, create two separate DataArrays and merge later.
        data_vars[f"{var}_predicted"] = xr.DataArray(
            pred_np,
            dims=["ensemble_member", "lead_time", "longitude", "latitude"],
            coords={
                "ensemble_member": np.arange(n_ens),
                "lead_time": lead_time,
                "longitude": lon,
                "latitude": lat,
            },
        )
        data_vars[f"{var}_target"] = xr.DataArray(
            targ_np,
            dims=["lead_time", "longitude", "latitude"],
            coords={
                "lead_time": lead_time,
                "longitude": lon,
                "latitude": lat,
            },
        )

    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(init_time=ic_datetime)
    return ds


# ════════════════════════════════════════════════════════════════════════
# 10.  Scoring: area-weighted CRPS and ensemble-mean RMSE
# ════════════════════════════════════════════════════════════════════════
def compute_area_weights(lat: np.ndarray) -> torch.Tensor:
    """Compute cosine-latitude area weights, normalized to mean=1."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _area_weighted_rmse_all_leads(ens_mean: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Area-weighted RMSE for all lead times at once.

    Args:
        ens_mean: (H, lon, lat)
        target:   (H, lon, lat)
        weights:  (lat,)
    Returns: (H,)
    """
    weighted_sq_err = (ens_mean - target) ** 2 * weights[None, None, :]
    return weighted_sq_err.nanmean(dim=(-2, -1)).sqrt()


def _area_weighted_crps_all_leads(preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Area-weighted fair CRPS for all lead times at once.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        preds:   (N, H, lon, lat) – ensemble predictions
        target:  (H, lon, lat)
        weights: (lat,)
    Returns: (H,)
    """
    N = preds.shape[0]
    term1 = (preds - target.unsqueeze(0)).abs().mean(dim=0)  # (H, lon, lat)
    if N > 1:
        diff = preds.unsqueeze(0) - preds.unsqueeze(1)  # (N, N, H, lon, lat)
        term2 = diff.abs().sum(dim=(0, 1)) / (N * (N - 1))  # (H, lon, lat)
    else:
        term2 = torch.zeros_like(term1)
    crps = term1 - 0.5 * term2  # (H, lon, lat)
    return (crps * weights[None, None, :]).nanmean(dim=(-2, -1))  # (H,)


def score_forecasts(
    all_results: List[Dict[str, Any]],
    latitude: np.ndarray,
    var_names: List[str],
    hourly_resolution: int,
    device: str = "cpu",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Score forecasts across multiple ICs.

    Returns dict with:
      crps:  {var: (prediction_horizon,)} averaged over ICs
      rmse:  {var: (prediction_horizon,)} averaged over ICs
    """
    weights = compute_area_weights(latitude).to(device)

    crps_sum: Dict[str, Optional[torch.Tensor]] = {var: None for var in var_names}
    rmse_sum: Dict[str, Optional[torch.Tensor]] = {var: None for var in var_names}

    for result in tqdm(all_results, desc="Scoring ICs", unit="IC"):
        preds = result["preds_raw"]
        targets = result["targets_raw"]
        for var in var_names:
            pred_t = preds[var].to(device)  # (N, H, lon, lat)
            targ_t = targets[var].to(device)  # (H, lon, lat)

            crps = _area_weighted_crps_all_leads(pred_t, targ_t, weights)  # (H,)
            rmse = _area_weighted_rmse_all_leads(pred_t.mean(dim=0), targ_t, weights)  # (H,)

            crps_sum[var] = crps if crps_sum[var] is None else crps_sum[var] + crps
            rmse_sum[var] = rmse if rmse_sum[var] is None else rmse_sum[var] + rmse

    n_ic = len(all_results)
    return dict(
        crps={var: (crps_sum[var] / n_ic).cpu().numpy() for var in var_names},
        rmse={var: (rmse_sum[var] / n_ic).cpu().numpy() for var in var_names},
    )


def log_scores_to_wandb(
    scores: dict,
    var_names: List[str],
    hourly_resolution: int,
    wandb_project: str,
    wandb_run_name: str = None,
    wandb_entity: str = None,
    extra_config: dict = None,
):
    """Log per-variable CRPS and RMSE to wandb, one step per lead time."""
    import wandb

    config = {"hourly_resolution": hourly_resolution}
    if extra_config:
        config.update(extra_config)
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or "era5_inference",
        entity=wandb_entity,
        config=config,
    )
    wandb.define_metric("lead_time_in_hours")
    wandb.define_metric("crps/*", step_metric="lead_time_in_hours")
    wandb.define_metric("rmse/*", step_metric="lead_time_in_hours")

    n_lead = len(scores["crps"][var_names[0]])
    lead_hours = np.arange(1, n_lead + 1) * hourly_resolution

    for h_idx, h in enumerate(lead_hours):
        wandb.log(
            {"lead_time_in_hours": int(h)}
            | {f"crps/{var}": scores["crps"][var][h_idx] for var in var_names}
            | {f"rmse/{var}": scores["rmse"][var][h_idx] for var in var_names}
        )

    run.finish()
    log.info("Scores logged to wandb.")


# ════════════════════════════════════════════════════════════════════════
# 11.  Main
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ERA5 standalone inference")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="hf://salv47/u-cast/ucast.ckpt",
        help="Path to the primary model checkpoint (.ckpt). "
        "Also used as the first entry when --ckpt-paths is provided.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config.yaml",
        help="Path to the config YAML (default: configs/config.yaml from this repo). "
        "Reused for all deep-ensemble models unless --config-paths is provided.",
    )
    parser.add_argument(
        "--ckpt-paths",
        type=str,
        nargs="+",
        default=None,
        help="Deep ensemble: additional checkpoint paths (one per model). "
        "Combined with --ckpt-path as the first entry.",
    )
    parser.add_argument(
        "--config-paths",
        type=str,
        nargs="+",
        default=None,
        help="Deep ensemble: config YAML paths matching --ckpt-paths. "
        "If omitted, --config-path is reused for every model.",
    )
    parser.add_argument(
        "--data-dir", type=str, nargs="+", default=["gs://weatherbench2/datasets/era5"], help="Data directory (local or GCS). Can pass multiple."
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="data/stats",
        help="Directory with normalization statistics (default: data/wb2/stats from this repo)",
    )
    parser.add_argument(
        "--prediction-horizon", type=int, default=30, help="Number of autoregressive steps (default: 30, so 15 days at 12h resolution)"
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of ensemble members (default: 1)",
    )
    parser.add_argument(
        "--ic-start-dates",
        type=str,
        nargs="+",
        default=["2020-01-01"],
        help="Initial condition datetimes (last input time), e.g. 2022-01-01T00 2022-07-01T12. "
        "Default: single date from test_slice start.",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Output path for the xarray Dataset (.nc)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--score", action="store_true", help="Compute area-weighted CRPS and RMSE")
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="wandb project name (enables wandb logging of scores)"
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    # ── Build checkpoint / config lists (single-model is just a list of 1) ──
    ckpt_path_list = [args.ckpt_path] + (args.ckpt_paths or [])
    if args.config_paths:
        config_path_list = [args.config_path] + args.config_paths
        if len(config_path_list) != len(ckpt_path_list):
            raise ValueError(
                f"--config-paths length ({len(config_path_list) - 1}) must match "
                f"--ckpt-paths length ({len(ckpt_path_list) - 1})"
            )
    else:
        config_path_list = [args.config_path] * len(ckpt_path_list)

    ckpt_path_list = [_resolve_path(p) for p in ckpt_path_list]
    config_path_list = [_resolve_path(p) for p in config_path_list]

    is_deep_ensemble = len(ckpt_path_list) > 1

    # ── Load configs ──
    cfg_list = []
    for cp in config_path_list:
        with open(cp) as f:
            cfg_list.append(yaml.safe_load(f))
    cfg = cfg_list[0]  # primary config used for data/normalizer setup

    dm_cfg = cfg["datamodule"]

    var_names = list(dm_cfg["input_vars"])  # = output_vars for this config
    window = dm_cfg["window"]
    hourly_resolution = dm_cfg["hourly_resolution"]
    prediction_horizon = args.prediction_horizon
    ensemble_size = args.ensemble_size

    log.info(f"Config: {window=}, {hourly_resolution=}h, {prediction_horizon=}, {ensemble_size=}")
    log.info(f"Device: {args.device}")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed(args.seed)

    # ── Set up circular padding BEFORE building the model ──
    set_circular_height_padding()

    # ── Open dataset ──
    dataset_name = dm_cfg["dataset"]
    ds = open_era5_zarr(args.data_dir, dataset_name)

    # Ensure latitude is sorted ascending
    ds = ensure_latitude_is_ascending(ds)
    log.info(f"Opened dataset: {dataset_name} with {len(ds.time)} time steps")

    # ── Load normalizer ──
    normalizer = setup_normalizer(args.stats_dir, var_names, ds, device=args.device)

    # ── Load static conditions ──
    static_cond = load_static_conditions(ds, list(dm_cfg["static_fields"]))

    # ── Determine IC dates ──
    ic_dates = [np.datetime64(d) for d in args.ic_start_dates]
    log.info(f"Initial condition dates ({len(ic_dates)}): {ic_dates}")

    # ── Run inference ──
    if is_deep_ensemble:
        log.info(
            f"Deep ensemble mode: {len(ckpt_path_list)} models, "
            f"{ensemble_size} total members ({_distribute_members(ensemble_size, len(ckpt_path_list))} per model)"
        )
        # Model-outer / IC-inner: each model loaded ONCE across all ICs
        all_results, all_datasets = _run_deep_ensemble_all_ics(
            ckpt_paths=ckpt_path_list,
            configs=cfg_list,
            ic_dates=ic_dates,
            ds=ds,
            var_names=var_names,
            static_cond=static_cond,
            normalizer=normalizer,
            window=window,
            prediction_horizon=prediction_horizon,
            ensemble_size=ensemble_size,
            hourly_resolution=hourly_resolution,
            device=args.device,
        )
    else:
        assert len(ckpt_path_list) == 1, "Single-model mode should have exactly one checkpoint path"
        model = load_model_from_checkpoint(ckpt_path_list[0], cfg, device=args.device)
        log.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        all_results = []
        all_datasets = []
        for ic_idx, ic_dt in enumerate(ic_dates):
            log.info(f"\n{'='*60}")
            log.info(f"IC {ic_idx + 1}/{len(ic_dates)}: {ic_dt}")
            log.info(f"{'='*60}")

            batch = extract_batch_for_ic(
                ds=ds,
                ic_datetime=ic_dt,
                var_names=var_names,
                static_cond=static_cond,
                normalizer=normalizer,
                window=window,
                prediction_horizon=prediction_horizon,
                hourly_resolution=hourly_resolution,
                device=args.device,
            )

            results = run_autoregressive_inference(
                model=model,
                batch=batch,
                var_names=var_names,
                normalizer=normalizer,
                window=window,
                prediction_horizon=prediction_horizon,
                ensemble_size=ensemble_size,
                device=args.device,
            )

            ic_ds = build_xarray_dataset(results, batch, ic_dt, hourly_resolution, var_names)
            all_results.append(results)
            all_datasets.append(ic_ds)

    # ── Score ──
    if args.score:
        log.info("Computing scores...")
        scores = score_forecasts(all_results, ds.latitude.values, var_names, hourly_resolution)
        lead_hours = np.arange(1, prediction_horizon + 1) * hourly_resolution

        # Print summary
        surface_vars = [
            v
            for v in var_names
            if "_" not in v
            or v.startswith("10m")
            or v.startswith("2m")
            or v == "sea_surface_temperature"
            or v == "mean_sea_level_pressure"
        ]
        if not surface_vars:
            surface_vars = var_names[:5]  # fallback
        print("\n" + "=" * 70)
        print("SCORES (area-weighted, averaged over ICs)")
        print("=" * 70)
        for var in surface_vars:
            print(f"\n  {var}:")
            for h_idx, h in enumerate(lead_hours):
                print(
                    f"    Lead {h:3d}h:  CRPS={scores['crps'][var][h_idx]:.4f}  "
                    f"RMSE={scores['rmse'][var][h_idx]:.4f}"
                )

        if args.wandb_project:
            log_scores_to_wandb(
                scores,
                var_names,
                hourly_resolution,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name,
                wandb_entity=args.wandb_entity,
                extra_config={
                    "ckpt_paths": ckpt_path_list,
                    "ensemble_size": ensemble_size,
                    "ic_dates": [str(d) for d in ic_dates],
                    "n_ics": len(ic_dates),
                    "prediction_horizon": prediction_horizon,
                    "n_vars": len(var_names),
                    "seed": args.seed,
                },
            )

    # ── Save ──
    if args.output_path:
        # ── Merge all ICs into a single Dataset ──
        if len(all_datasets) > 1:
            # Add init_time as a dimension
            for i, d in enumerate(all_datasets):
                all_datasets[i] = d.expand_dims("init_time")
            merged_ds = xr.concat(all_datasets, dim="init_time")
        else:
            merged_ds = all_datasets[0].expand_dims("init_time")

        log.info(f"Saving forecasts to {args.output_path}...")
        merged_ds.to_netcdf(args.output_path)
        log.info(f"Forecasts saved to {args.output_path}")
        return merged_ds


if __name__ == "__main__":
    main()
