from __future__ import annotations

import copy
import os
from collections import defaultdict
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from src.datamodules.forcings import add_derived_vars
from src.evaluation.aggregator import Aggregator
from src.utilities.normalization import get_normalizer
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    subsample_preselected_indices,
    to_torch_and_device,
)


log = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Packer:
    """Packs a dict of named per-variable tensors into one channel-stacked tensor, and unpacks it back."""

    def __init__(self, names: Sequence[str], axis: int = -3):
        self.names = list(names)
        self.axis = axis

    def pack(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([tensors[n] for n in self.names], dim=self.axis)

    def unpack(self, tensor: torch.Tensor) -> TensorDict:
        assert tensor.shape[self.axis] == len(self.names), f"{tensor.shape=}, {len(self.names)=}"
        batch_size = list(tensor.shape)
        batch_size.pop(self.axis)
        return TensorDict({n: tensor.select(self.axis, i) for i, n in enumerate(self.names)}, batch_size=batch_size)


def get_dims_of_dataset(datamodule_config: DictConfig) -> Dict[str, object]:
    """Infer the (input, output, spatial, conditional) channel dimensions from the ERA5 config."""
    spatial_by_token = {"64x32": (64, 32), "240x121": (240, 121), "360x181": (360, 181)}
    dataset_str = datamodule_config.dataset
    spatial = next((dims for tok, dims in spatial_by_token.items() if tok in dataset_str), None)
    if spatial is None:
        raise ValueError(f"Unknown dataset spatial dimensions in: {dataset_str}")
    static_fields = datamodule_config.get("static_fields", []) or []
    forcings = datamodule_config.get("forcing_fields", []) or []
    return {
        "input": len(datamodule_config.input_vars),
        "output": len(datamodule_config.output_vars),
        "spatial": spatial,
        "conditional": len(static_fields) + len(forcings),
    }


def _assert_increasing(x: np.ndarray):
    if not (np.diff(x) > 0).all():
        raise ValueError(f"array is not increasing: {x}")


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
    return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
    """Calculate the area overlap as a function of latitude."""
    bounds = _latitude_cell_bounds(points)
    _assert_increasing(bounds)
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    return np.sin(upper) - np.sin(lower)


def get_lat_weights(ds: xr.Dataset) -> xr.DataArray:
    """Computes latitude/area weights from latitude coordinate of dataset."""
    weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
    weights /= np.mean(weights)
    weights = ds.latitude.copy(data=weights)
    assert (weights > 0).all(), f"{weights=}"
    # print(f"{weights.mean()=}, {weights.min()=}, {weights.max()=}, {weights[:3]=}, {weights[-3:]=}")
    return weights


def open_zarr_dataset(zarr_path, **kwargs):
    try:
        ds = xr.open_zarr(zarr_path, **kwargs)
    except Exception:
        try:
            ds = xr.open_zarr(zarr_path, zarr_format=3, **kwargs)
        except Exception:
            kwargs["consolidated"] = False
            ds = xr.open_zarr(zarr_path, **kwargs)
    return ds


def find_path_from_dir_opts(data_dirs: List[str], dataset_name: str):
    for data_dir in data_dirs:
        potential_path = join(data_dir, dataset_name)
        if os.path.isfile(potential_path) or os.path.isdir(potential_path):
            return potential_path
    raise FileNotFoundError(f"Could not find {dataset_name} in any of the following data directories: {data_dirs}")


def extract_date(date_info):
    """Extract a date from a date string or datetime object, returning a numpy datetime64[D]."""
    if isinstance(date_info, datetime):
        date = date_info
    else:
        date = date_info.split(" ")[0]  # "2020-01-01 00:00:00" -> "2020-01-01"
    return np.datetime64(date, "D")


def get_date(date_str: str):
    if ":" in date_str and "T" in date_str:
        fmt = "%Y-%m-%dT%H:%M:%S"
    elif ":" in date_str:
        fmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%Y-%m-%d"
    return datetime.strptime(date_str, fmt)


def extract_time_subsample(dataset: str, hourly_resolution: int) -> None:
    # Infer hourly resolution of dataset
    if "-6h-" in dataset:
        if hourly_resolution == 6:
            time_subsample = 1
        elif hourly_resolution == 12:
            time_subsample = 2
        else:
            raise ValueError(f"Invalid hourly resolution: {hourly_resolution} for dataset: {dataset}")
    elif "-12h-" in dataset:
        if hourly_resolution == 12:
            time_subsample = 1
        else:
            raise ValueError(f"Invalid hourly resolution: {hourly_resolution} for dataset: {dataset}")
    elif "-1h-" in dataset:
        time_subsample = hourly_resolution
    else:
        raise ValueError(f"Could not infer hourly resolution from dataset: {dataset}")

    if time_subsample > 1:
        log.info(f"Setting slice subsample to {time_subsample} due to {hourly_resolution=} for {dataset=}.")
    return time_subsample


def get_slice(slice_, split: str, time_subsample: int) -> slice:
    if isinstance(slice_, Sequence) and len(slice_) == 2:
        slice_ = slice(*slice_)
    assert isinstance(slice_, slice), f"Invalid slice for {split}: {slice_}"
    # Convert start and end to dates, if only years are given
    if isinstance(slice_.start, int):
        slice_ = slice(f"{slice_.start}-01-01", slice_.stop, slice_.step)
    if isinstance(slice_.stop, int):
        slice_ = slice(slice_.start, f"{slice_.stop}-12-31", slice_.step)
    # If it does not have a step, set the step to time_subsample
    if slice_.step is None:
        slice_ = slice(slice_.start, slice_.stop, time_subsample)
    # To datetime
    slice_ = slice(get_date(str(slice_.start)), get_date(str(slice_.stop)), slice_.step)
    if split != "predict":
        assert slice_.step == time_subsample, f"Invalid step for {split=}: {slice_.step}"
    return slice_


class ERA5DataModule(pl.LightningDataModule):
    """Loads the WeatherBench-2 ERA5 zarr and yields train + validation dataloaders for U-Cast.

    Subclasses are not needed: this single class handles data loading, normalization, the train/val
    temporal splits, and the per-horizon validation aggregators.
    """

    def __init__(
        self,
        data_dir: str,
        data_dir_stats: Optional[str] = None,
        dataset: str = "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        train_slice: Optional[slice] = slice("2015-01-01", "2018-12-31"),
        val_slice: Optional[slice] = slice("2019-01-01", "2019-12-31"),
        hourly_resolution: int = 12,
        possible_initial_times_eval: Optional[List[str]] = None,
        window: int = 1,  # Number of time steps to use in the input
        horizon: int = 1,  # Number of time steps to predict into the future
        prediction_horizon: int = None,  # None means use horizon and no auto-regressive prediction
        static_fields: Sequence[str] = (
            "land_sea_mask",
            "geopotential_at_surface",
        ),
        forcing_fields: Sequence[str] = None,
        max_val_samples: int = None,
        input_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        output_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        model_config: DictConfig = None,
        batch_size: int = 2,
        eval_batch_size: int = 64,
        num_workers: int = -1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to the directory containing the zarr dataset (or a ``weatherbench2`` subdirectory).
            data_dir_stats: Path to the directory containing the normalization statistics (auto-found if None).
            dataset: Name of the WeatherBench-2 ``.zarr`` dataset.
            train_slice / val_slice: Temporal slices for training / validation.
            hourly_resolution:  1 for hourly, 6 for 6-hourly etc.
            possible_initial_times_eval: Allowed initial times of day.
            window: Number of input time steps; horizon: training steps to predict;
            prediction_horizon: validation steps to predict (autoregressive if > horizon).
            static_fields / forcing_fields: Conditional input fields.
            input_vars / output_vars: Atmospheric variables fed in / predicted.
            batch_size / eval_batch_size / num_workers / ...: Standard dataloader settings.
        """
        super().__init__()
        raise_error_if_invalid_type(data_dir, possible_types=[str, List, ListConfig], name="data_dir")
        assert hourly_resolution >= 1, f"Invalid hourly_resolution: {hourly_resolution}"
        assert dataset.endswith(".zarr"), f"dataset should be a .zarr file. Invalid dataset: {dataset}"
        if not isinstance(data_dir, (ListConfig, List)):
            data_dir = [data_dir]
        data_dir = list(data_dir)  # in case it's a ListConfig
        for i, data_dir_i in enumerate(data_dir):
            assert ".zarr" not in data_dir_i, "data_dir should not include the .zarr file. Specify in `dataset`"
            if "weatherbench2" not in data_dir_i:
                if os.path.isdir(join(data_dir_i, "weatherbench2")):
                    data_dir[i] = join(data_dir_i, "weatherbench2")

        self.zarr_path = find_path_from_dir_opts(data_dir, dataset)

        self.model_config = model_config
        self._data_train = self._data_val = None
        if num_workers == 0 and persistent_workers:
            log.warning("persistent_workers requires num_workers > 0; setting persistent_workers=False.")
            persistent_workers = False
        self.save_hyperparameters(ignore=["model_config"])
        self.hparams.data_dir = data_dir

        time_subsample = extract_time_subsample(dataset, hourly_resolution)
        if isinstance(train_slice, ListConfig):
            train_slice = list(train_slice)
        if isinstance(train_slice, List) and isinstance(train_slice[0], ListConfig):
            train_slice = [list(tslice) if isinstance(tslice, ListConfig) else tslice for tslice in train_slice]
        if not (isinstance(train_slice, List) and isinstance(train_slice[0], (List, Tuple, slice))):
            train_slice = [train_slice]
        for i, tslice in enumerate(train_slice):
            train_slice[i] = get_slice(tslice, "train", time_subsample)
        self.train_slice = train_slice
        self.val_slice = get_slice(val_slice, "val", time_subsample)

        # Sanity check: train must end before val starts
        train_end = extract_date(self.train_slice[-1].stop)
        val_start = extract_date(self.val_slice.start)
        assert train_end <= val_start, f"train_slice ends after val_slice starts: {train_slice}, {val_slice}"

        # Normalization
        if data_dir_stats is None:
            opts = data_dir + [os.path.dirname(data_dir_i) for data_dir_i in data_dir]  # also check parent directories
            data_dir_stats = find_path_from_dir_opts(opts, "statistics")
            log.info(f"data_dir_stats is not specified. Found data_dir_stats at: {data_dir_stats}")

        data_dir_stats = Path(data_dir_stats)
        path_mean = data_dir_stats / "era5_mean.nc"
        path_std = data_dir_stats / "era5_std.nc"
        path_std_res = data_dir_stats / "era5_residual_std.nc"
        path_min = data_dir_stats / "era5_min.nc"
        if not path_mean.exists() or not path_std.exists():
            raise FileNotFoundError(f"Could not find normalization files at ``{path_mean}`` and/or ``{path_std}``")

        self._latitude = self._longitude = None

        # ---- Normalizer + variable packers ----
        self.all_vars = list(set(input_vars) | set(output_vars))
        self.normalizer = get_normalizer(
            path_mean, path_std, names=self.all_vars, global_stds_res_path=path_std_res, input_names=input_vars
        )
        if "sea_surface_temperature" in self.all_vars:
            # Fill NaNs in sea_surface_temperature with the min value (over land).
            fill_value_sst = xr.open_dataset(path_min)["sea_surface_temperature"].values.item()
            any_sst = open_zarr_dataset(self.zarr_path).isel(time=0)["sea_surface_temperature"]
            if not (np.diff(any_sst.latitude.data) > 0).all():
                any_sst = any_sst.reindex(latitude=list(reversed(any_sst.latitude.data)))
            if any_sst.dims[0] not in ["longitude", "lon"]:
                any_sst = any_sst.transpose("longitude", "latitude")
            mask = np.isnan(any_sst.values)
            log.info(
                f"Filling NaNs in sea_surface_temperature with {fill_value_sst} degC. "
                f"Fraction of NaNs: {np.sum(mask) / mask.size:.4f}"
            )
            self.normalizer = NaNCleaner(self.normalizer, mask=mask, sea_surface_temperature=fill_value_sst)

        channel_axis = -3
        # Important: do NOT use set() here as it changes variable ordering.
        input_only_vars = [v for v in input_vars if v not in output_vars]
        in_vars_without_input_only = [v for v in input_vars if v not in input_only_vars]
        if len(input_only_vars) > 0:
            log.info(f"Input-only variables: {input_only_vars}")
        self.in_packer = Packer(in_vars_without_input_only, axis=channel_axis)
        self.in_only_packer = Packer(input_only_vars, axis=channel_axis) if len(input_only_vars) > 0 else None
        self.out_packer = Packer(output_vars, axis=channel_axis)

    def get_horizon(self, split: str) -> int:
        if split in ["val", "validate"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        assert split in ["train", "fit"], f"Invalid split: {split}"
        return self.hparams.horizon

    def _open_dataset_start(self, zarr_path, time_slice: slice) -> xr.Dataset:
        """Open the dataset with xarray."""
        try:
            ds = open_zarr_dataset(zarr_path, decode_times=True, chunks=None, mask_and_scale=False, consolidated=True)
        except Exception as e:
            raise RuntimeError(f"Could not open zarr dataset: {zarr_path}") from e

        time_subsample_here = extract_time_subsample(zarr_path, self.hparams.hourly_resolution)
        time_slice_here = slice(time_slice.start, time_slice.stop, time_subsample_here)
        ds = ds.sel(time=time_slice_here)

        # Assert that latitude is increasing order
        if not (np.diff(ds.latitude.data) > 0).all():
            ds = ds.reindex(latitude=list(reversed(ds.latitude.data)))
            assert (np.diff(ds.latitude.data) > 0).all(), "Latitude is not in increasing order after reindexing."
            log.info(f"Reindexed latitude to be increasing. Now: {ds.latitude.data[:5]} ... {ds.latitude.data[-5:]}")
        return ds

    def get_split_dataset(self, split: str, time_slice: slice, **kwargs) -> ERA5Dataset2D:
        assert split in ["fit", "train", "validate", "val"], f"Invalid split: {split}"
        ds = self._open_dataset_start(self.zarr_path, time_slice)

        # Training uses all available initial times; evaluation restricts to ``possible_initial_times_eval``.
        possible_initial_times = None if split in ["fit", "train"] else self.hparams.possible_initial_times_eval

        dset = ERA5Dataset2D(
            dataset=ds,
            split=split,
            horizon=self.get_horizon(split),
            input_vars=self.hparams.input_vars,
            output_vars=self.hparams.output_vars,
            normalizer=self.normalizer,
            in_only_packer=self.in_only_packer,
            zarr_path=self.zarr_path,
            window=self.hparams.window,
            static_fields=self.hparams.static_fields,
            forcing_fields=self.hparams.forcing_fields,
            hourly_resolution=self.hparams.hourly_resolution,
            possible_initial_times=possible_initial_times,
            **kwargs,
        )

        if self._latitude is None:
            self._latitude = ds.latitude
            self._longitude = ds.longitude
        return dset

    def setup(self, stage: Optional[str] = None):
        """Populate ``self._data_train`` and ``self._data_val`` from the configured slices."""
        if stage not in ("fit", "validate", None):
            return

        train_sets = []
        for train_slice in self.train_slice:
            train_set_ds = self.get_split_dataset("fit", train_slice)
            log.info(f"Training dataset slice {train_slice} with {len(train_set_ds)} samples loaded.")
            train_sets.append(train_set_ds)
        if len(train_sets) == 1:
            self._data_train = train_sets[0]
        else:
            self._data_train = torch.utils.data.ConcatDataset(train_sets)
            log.info(f"Total training samples from {len(self.train_slice)} slices: {len(self._data_train)}")

        self._data_val = self.get_split_dataset(
            split="validate", time_slice=self.val_slice, max_num_samples=self.hparams.max_val_samples
        )

        self.print_data_sizes(stage)

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        device: torch.device = None,
    ) -> Dict[str, Aggregator]:
        area_weights = to_torch_and_device(getattr(self, f"_data_{split}").area_weights_tensor, device)
        aggregators = {}
        for h in range(1, self.get_horizon(split) + 1):
            aggregators[f"t{h}"] = Aggregator(
                name=f"t{h * self.hparams.hourly_resolution}",
                area_weights=area_weights,
                is_ensemble=is_ensemble,
            )
        return aggregators

    # ---------------------------------- Dataloaders
    def _dataloader_kwargs(self) -> dict:
        kwargs = dict(
            num_workers=int(self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
        if self.hparams.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.hparams.prefetch_factor
        return kwargs

    def print_data_sizes(self, stage: str = None):
        if self._data_train is not None:
            log.info(f"Dataset sizes train: {len(self._data_train)}, val: {len(self._data_val)}")
        elif self._data_val is not None:
            log.info(f"Dataset validation size: {len(self._data_val)}")

    def train_dataloader(self):
        if self._data_train is None:
            return None
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def _adjust_eval_batch_size(self, dataset: Dataset) -> int:
        """Cap the eval batch size so every DDP rank gets at least one batch."""
        if self.trainer is None:
            return self.hparams.eval_batch_size
        batch_size = min(self.hparams.eval_batch_size, len(dataset) // self.trainer.world_size)
        if batch_size != self.hparams.eval_batch_size:
            log.info(f"Adjusting eval batch size to {batch_size} (dataset size {len(dataset)}).")
        return batch_size

    def val_dataloader(self):
        if self._data_val is None:
            return None
        return DataLoader(
            dataset=self._data_val,
            batch_size=self._adjust_eval_batch_size(self._data_val),
            shuffle=False,
            **self._dataloader_kwargs(),
        )


class NaNCleaner(torch.nn.Module):
    def __init__(self, normalizer, mask, **vars_to_fill_values):
        super().__init__()
        self.normalizer = normalizer
        self.mask = to_torch_and_device(mask, None)  # Boolean mask where True indicates valid data
        assert len(vars_to_fill_values) > 0, "Please provide at least one variable to fill NaNs for."
        self.vars_to_fill_values = vars_to_fill_values

    def _apply(self, fn, recurse=True):
        self.normalizer._apply(fn, recurse=recurse)
        self.mask = fn(self.mask)

    def fill_nans(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replace NaNs (over the masked region) with the per-variable fill value, in physical units."""
        for var, fill_value in self.vars_to_fill_values.items():
            if var in tensors.keys():
                tensors[var] = torch.where(self.mask.to(tensors[var].device), fill_value, tensors[var])
        return tensors

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.normalizer.normalize(self.fill_nans(tensors))

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors = self.normalizer.denormalize(tensors)
        for var, fill_value in self.vars_to_fill_values.items():
            tensor = tensors[var]
            tensors[var] = torch.where(self.mask, fill_value, tensor)
        return tensors

    def normalized_to_residual_normalized(self, *args, **kwargs):
        return self.normalizer.normalized_to_residual_normalized(*args, **kwargs)

    def normalized_residual_to_normalized(self, *args, **kwargs):
        return self.normalizer.normalized_residual_to_normalized(*args, **kwargs)


class ERA5Dataset2D(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        split: str,
        horizon: int,
        input_vars: Sequence[str],
        output_vars: Sequence[str],
        normalizer,
        static_fields: Sequence[str],
        zarr_path: str = None,
        forcing_fields: Sequence[str] = None,
        in_only_packer: Packer = None,
        window: int = 1,
        hourly_resolution: int = 12,
        possible_initial_times: Optional[Sequence[str]] = None,
        max_num_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.zarr_path = zarr_path
        self.dataset_id = split
        self.horizon = horizon
        self.window = window
        self.last_ic_idx = window - 1
        self.max_num_samples = max_num_samples
        self.possible_initial_times = (
            [int(h) for h in possible_initial_times] if possible_initial_times is not None else None
        )
        all_times = self.dataset.time.values[
            self.last_ic_idx : -horizon
        ]  # keep only times for which we can predict horizon hours ahead
        ds_idxs = np.arange(len(all_times), dtype=int)
        if self.possible_initial_times is not None:
            all_hours = all_times.astype("datetime64[h]").astype(int) % 24
            valid_hours = np.isin(all_hours, self.possible_initial_times)
            ds_idxs = ds_idxs[valid_hours]

        if max_num_samples is not None:
            ds_idxs = subsample_preselected_indices(ds_idxs, max_num_samples)

        self.ds_idxs = ds_idxs
        self.length = len(ds_idxs)
        self.lat_lon_format = ("longitude", "latitude")
        self.forcing_fields = forcing_fields or []

        if self.length < 0:
            raise ValueError(
                f"Invalid length: {self.length} for split: {split}; len(self.dataset.time)={len(self.dataset.time)}, horizon: {horizon}, max_num_samples: {max_num_samples}"
            )

        self.preproc_func = torch.from_numpy
        # Create static conditions
        if static_fields is not None and len(static_fields) > 0:
            static_conditions = [self.dataset[static_field].compute().values for static_field in static_fields]
            # Stack static conditions
            static_conditions = np.stack(static_conditions, axis=0)
            # Standardize static conditions across field dimension
            mean_sc = static_conditions.mean(axis=(-2, -1), keepdims=True)
            std_sc = static_conditions.std(axis=(-2, -1), keepdims=True)
            static_conditions = (static_conditions - mean_sc) / std_sc
            self.static_conditions = torch.from_numpy(static_conditions).float()
        else:
            self.static_conditions = None
        self._area_weights = get_lat_weights(self.dataset)
        nlon = self.dataset.longitude.size
        repeat_shape = (nlon, 1)
        self._area_weights_tensor = torch.as_tensor(self.area_weights.values, dtype=torch.float32).repeat(repeat_shape)

        # ERA5Dataset2D specific logic
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.normalizer = copy.copy(normalizer)
        self.normalizer.to("cpu")
        self.all_vars = set(input_vars) | set(output_vars)
        self.input_only_vars = set(input_vars) - set(output_vars)
        self.in_only_packer = in_only_packer
        self.skip_idxs = set()
        self.possible_2d_vars = [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "sea_surface_temperature",
        ]
        possible_3d_vars = [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ]
        self.vars2d = list(sorted([v for v in self.all_vars if v in self.possible_2d_vars]))
        vars_not_2d = [v for v in self.all_vars if v not in self.possible_2d_vars]
        self.var3d_to_levels = defaultdict(list)
        for v in vars_not_2d:
            var_name = "_".join(v.split("_")[:-1])
            if var_name in possible_3d_vars:
                p_level = int(v.split("_")[-1])
                self.var3d_to_levels[var_name].append(p_level)
            else:
                raise ValueError(f"Invalid variable: {v}")
        # Find all unique levels and sort them (higher levels first)
        self.all_levels = set()
        for v in self.var3d_to_levels:
            self.var3d_to_levels[v] = sorted(self.var3d_to_levels[v], reverse=True)
            self.all_levels.update(self.var3d_to_levels[v])
        self.all_levels = sorted(self.all_levels)

        self.all_vars_stem = sorted(set(self.var3d_to_levels.keys()) | set(self.vars2d))
        self.dataset = self.dataset[self.all_vars_stem]
        self.dataset = self.dataset.sel(level=self.all_levels)
        ds = self.dataset
        self.level_to_idx = {int(lvl): i for i, lvl in enumerate(ds.level.values)} if "level" in ds.sizes else None
        self._3d_ops = []
        for vr, ops in self.var3d_to_levels.items():
            ops_list = []
            for level in ops:
                idx = self.level_to_idx[int(level)]
                key = f"{vr}_{level}"
                ops_list.append((idx, key))
            self._3d_ops.append((vr, ops_list))

        if "val" in self.dataset_id:
            dates = [self.__get_date__(i) for i in [0, 1, 2, 3, -3, -2, -1] if i < self.__len__()]
            dates = [str(d) for d in dates]
            if self.max_num_samples is not None:
                log.info(f"Using {self.max_num_samples=} for split: `{self.dataset_id}`.\nDates examples: {dates}")

    @property
    def area_weights(self):
        return self._area_weights

    @property
    def area_weights_tensor(self):
        return self._area_weights_tensor

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        weights = self.area_weights_tensor
        var_to_weight = torch.ones(len(self.output_vars))

        # 1. Pressure weighting (GraphCast style): each level weighted proportionally to its pressure.
        # fmt: off
        all_levels = (
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
            450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
        )
        # fmt: on
        level_div = np.mean(all_levels)

        for i, ov in enumerate(self.output_vars):
            if ov not in self.possible_2d_vars:
                p_level = int(ov.split("_")[-1])
                var_to_weight[i] = p_level / level_div

        # 2. Surface variable weighting (GraphCast style)
        fixed_var_weights = {
            "2m_temperature": 1.0,
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "sea_surface_temperature": 0.1,
        }

        for i, ov in enumerate(self.output_vars):
            if ov in fixed_var_weights.keys():
                var_to_weight[i] = fixed_var_weights[ov]

        n_spatial_dims = len(weights.shape)
        var_to_weight = var_to_weight.view(len(self.output_vars), *([1] * n_spatial_dims))

        weights = var_to_weight * weights.unsqueeze(0)
        return weights

    def __len__(self):
        return self.length

    def __get_date__(self, idx):
        try:
            idx_actual = int(self.ds_idxs[idx])
        except IndexError:
            return None
        batch_start_time = self.dataset.coords["time"].values[idx_actual]
        return batch_start_time

    def __getitem__(self, idx):
        if idx in self.skip_idxs:
            return self.__getitem__(idx + 1)
        idx_actual = int(self.ds_idxs[idx])
        time_slice = slice(idx_actual, idx_actual + self.window + self.horizon)

        # static conditions are time-independent variables such as land_sea_mask, altitude, etc.
        arrays = dict(static_condition=self.static_conditions) if self.static_conditions is not None else dict()
        batch = self.dataset.isel(time=time_slice).load()
        batch_start_time = batch.coords["time"].values[self.last_ic_idx]
        dynamics = self._create_var_to_tensor_dict(batch)
        # Forcings
        add_derived_vars(batch)
        dyn_arrays = []
        any_tensor = dynamics[next(iter(dynamics))]
        t, h, w = any_tensor.shape[-3], any_tensor.shape[-2], any_tensor.shape[-1]
        for vr in self.forcing_fields:
            dyn_arr = torch.from_numpy(batch[vr].values)
            if "day_progress" in vr:
                dyn_arr = dyn_arr.unsqueeze(-1)
            elif "year_progress" in vr:
                dyn_arr = dyn_arr.view(t, 1, 1)

            dyn_arr = dyn_arr.expand(t, h, w)
            dyn_arrays.append(dyn_arr)
        arrays["dynamical_condition"] = torch.stack(dyn_arrays, dim=1)

        if len(self.input_only_vars) > 0:
            dynamical_condition = {vr: dynamics.pop(vr) for vr in self.input_only_vars}
            arrays["dynamical_condition"] = self.in_only_packer.pack(self.normalizer.normalize(dynamical_condition))

        arrays["dynamics"] = dynamics
        if self.dataset_id not in ["train", "fit"]:
            np_datetime = batch_start_time.astype("datetime64[s]").astype(np.int64)
            arrays["metadata"] = dict(datetime=np_datetime)

        return arrays

    def _create_var_to_tensor_dict(self, batch):
        dynamics = dict()
        for vr in self.vars2d:
            dynamics[vr] = self.preproc_func(batch[vr].values)

        for vr, ops in self._3d_ops:
            full_data = self.preproc_func(batch[vr].values)
            for idx, key in ops:
                dynamics[key] = full_data[:, idx]

        return dynamics
