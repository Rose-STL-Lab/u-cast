from typing import Dict, List

import torch
import xarray as xr

from src.utilities.utils import get_logger


log = get_logger(__name__)


class StandardNormalizer(torch.nn.Module):
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: Dict[str, torch.Tensor],
        stds: Dict[str, torch.Tensor],
        names=None,
        std_residual=None,
        input_names: List[str] = None,
    ):
        super().__init__()
        self.means = means
        self.stds = stds
        self.std_residual = std_residual
        self.names = names if names is not None else list(means.keys())

        assert isinstance(means, dict), "Means and stds must be dictionaries!"
        assert all(name in means for name in self.names), "All names must be keys in the means dictionary!"
        assert all(name in stds for name in self.names), "All names must be keys in the stds dictionary!"

        if self.std_residual is not None:
            scale_normed_to_residual_normed = dict()
            scale_normed_residual_to_normed = dict()
            for k in self.names:
                if input_names is not None and k not in input_names:
                    scale_normed_to_residual_normed[k] = torch.tensor(1.0)
                    scale_normed_residual_to_normed[k] = torch.tensor(1.0)
                    log.warning(f"Variable {k} not in input_names; setting residual scaling factors to 1.0")
                else:
                    scale_normed_to_residual_normed[k] = self.stds[k] / self.std_residual[k]
                    scale_normed_residual_to_normed[k] = self.std_residual[k] / self.stds[k]
            self.scale_normed_to_residual_normed = scale_normed_to_residual_normed
            self.scale_normed_residual_to_normed = scale_normed_residual_to_normed

    def _apply(self, fn, recurse=True):
        super()._apply(fn)
        self.means = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.means.items()}
        self.stds = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.stds.items()}
        if self.std_residual is not None:
            self.std_residual = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.std_residual.items()}
        if hasattr(self, "scale_normed_to_residual_normed"):
            self.scale_normed_to_residual_normed = {
                k: fn(v) if torch.is_tensor(v) else v for k, v in self.scale_normed_to_residual_normed.items()
            }
            self.scale_normed_residual_to_normed = {
                k: fn(v) if torch.is_tensor(v) else v for k, v in self.scale_normed_residual_to_normed.items()
            }

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (t - self.means[k]) / self.stds[k] for k, t in tensors.items()}

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: t * self.stds[k] + self.means[k] for k, t in tensors.items()}

    def normalized_to_residual_normalized(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: t * self.scale_normed_to_residual_normed[k] for k, t in tensors.items()}

    def normalized_residual_to_normalized(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: t * self.scale_normed_residual_to_normed[k] for k, t in tensors.items()}

    def __copy__(self):
        return StandardNormalizer(self.means, self.stds, self.names, self.std_residual)

    def clone(self):
        return self.__copy__()


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.as_tensor(x.values, dtype=torch.float)


def _extract_variables(ds: xr.Dataset, names: List[str]) -> Dict[str, torch.Tensor]:
    """Helper to extract specific variables or pressure levels from a dataset."""
    extracted = {}
    for name in names:
        # Case 1: Simple extraction (Direct match or not flattened mode)
        if name in ds:
            extracted[name] = to_tensor(ds[name])
            continue

        # Case 2: Flattened mode logic (parsing <var_name>_<pressure_level>)
        parts = name.split("_")
        var_name = "_".join(parts[:-1])
        pressure_level = parts[-1]

        if not pressure_level.isdigit():
            raise ValueError(f"{name} is not in format <var>_<level>. Available keys: {list(ds.keys())}")

        level = int(pressure_level)
        try:
            # Select specific level from the 3D variable
            data = ds[var_name].sel(level=level)
            extracted[name] = to_tensor(data)
        except KeyError as e:
            print(f"Available coords: {ds.coords.values}")
            raise KeyError(f"Variable {name} (var: {var_name}, level: {level}) not found in dataset.") from e

    return extracted


def get_normalizer(
    global_means_path: str,
    global_stds_path: str,
    names: List[str],
    global_stds_res_path: str = None,
    **kwargs,
) -> StandardNormalizer:
    # 1. Load Data
    mean_ds = xr.open_dataset(global_means_path)
    std_ds = xr.open_dataset(global_stds_path)
    std_res_ds = xr.open_dataset(global_stds_res_path) if global_stds_res_path is not None else None

    # 3. Extract tensors
    means = _extract_variables(mean_ds, names)
    stds = _extract_variables(std_ds, names)
    if std_res_ds is not None:
        std_res_ds = _extract_variables(std_res_ds, names)

    return StandardNormalizer(means=means, stds=stds, names=names, std_residual=std_res_ds, **kwargs)
