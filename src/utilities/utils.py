"""General utility helpers shared by the training pipeline."""

from __future__ import annotations

import collections.abc
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
from torch import Tensor


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a multi-GPU-friendly logger that only emits from rank zero."""
    from pytorch_lightning.utilities import rank_zero_only

    logger = logging.getLogger(name)
    logger.setLevel(level)
    for lvl in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, lvl, rank_zero_only(getattr(logger, lvl)))
    return logger


log = get_logger(__name__)


def torch_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, TensorDictBase):
        return {k: torch_to_numpy(v) for k, v in x.items()}
    if isinstance(x, dict):
        return {k: torch_to_numpy(v) for k, v in x.items()}
    return x


def to_torch_and_device(x, device):
    if isinstance(x, (xr.Dataset, xr.DataArray)):
        x = x.values
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x is None:
        return None
    return x.to(device) if device is not None else x


def rrearrange(data, pattern: str, find_batch_size_max: bool = True, **axes_lengths):
    """``einops.rearrange`` extended to TensorDict / nested dict / distribution inputs."""
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        return rearrange(data, pattern, **axes_lengths)
    if isinstance(data, TensorDictBase):
        new_data = {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
        return to_tensordict(new_data, find_batch_size_max=find_batch_size_max)
    if isinstance(data, dict):
        return {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
    raise ValueError(f"Cannot rearrange {type(data)}")


def subtract_if_present(a, b):
    if isinstance(a, (TensorDictBase, dict)):
        return {key: subtract_if_present(a.get(key), b.get(key)) for key in a.keys()}
    if a is not None and b is not None:
        return a - b
    return a


def to_DictConfig(obj: Optional[Union[List, Dict]]) -> DictConfig:
    if isinstance(obj, DictConfig):
        return obj
    if isinstance(obj, list):
        try:
            return OmegaConf.from_dotlist(obj)
        except ValueError:
            return OmegaConf.create(obj)
    if isinstance(obj, dict):
        return OmegaConf.create(obj)
    return OmegaConf.create()


def to_tensordict(
    x,
    batch_size: Sequence[int] = None,
    find_batch_size_max: bool = False,
    force_same_device: bool = False,
    device=None,
) -> TensorDict:
    """Convert a dict of tensors to a TensorDict; pass-through for tensors and ndarrays."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    any_value = x[list(x.keys())[0]]
    device = any_value.device if force_same_device else device
    shared_batch_size = any_value.shape if batch_size is None else batch_size
    if find_batch_size_max:
        assert batch_size is None
        for t in x.values():
            if t.shape[: len(shared_batch_size)] != shared_batch_size:
                for i, (a, b) in enumerate(zip(t.shape, shared_batch_size)):
                    if a != b:
                        shared_batch_size = shared_batch_size[:i]
                        break
    return TensorDict(x, batch_size=shared_batch_size, device=device)


def raise_error_if_invalid_type(value: Any, possible_types: Sequence[Any], name: str = None):
    if all(not isinstance(value, t) for t in possible_types):
        name = name or (value.__name__ if hasattr(value, "__name__") else "value")
        raise ValueError(f"{name} must be an instance of either of {possible_types}, but was {type(value)}")
    return value


def enable_inference_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def disable_inference_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def subsample_preselected_indices(preselected_indices, max_num_samples: int):
    """Evenly subsample ``max_num_samples`` indices from a preselected list."""
    M = len(preselected_indices)
    if max_num_samples >= M:
        return preselected_indices
    spacing = (M - 1) / (max_num_samples - 1) if max_num_samples > 1 else 0
    positions = [int(round(i * spacing)) for i in range(max_num_samples)]
    if positions and positions[-1] >= M:
        positions[-1] = M - 1
    return [preselected_indices[pos] for pos in positions]


def _split_batch(x, start: int, end: int):
    if isinstance(x, (Tensor, TensorDictBase)):
        return x[start:end]
    return x


def run_func_in_sub_batches_and_aggregate(
    func: Union[Callable, Sequence[Callable]], *inputs, num_prediction_loops: int = 1, **kwargs
):
    """Run ``func`` in ``num_prediction_loops`` chunks along the batch axis and concatenate."""
    if isinstance(func, collections.abc.Sequence):
        assert len(func) == num_prediction_loops
    else:
        func = [func] * num_prediction_loops
    results: Any = defaultdict(list)
    full_batch_size = inputs[0].shape[0] if len(inputs) > 0 else kwargs[next(iter(kwargs))].shape[0]
    offset_factor = full_batch_size // num_prediction_loops
    for i in range(num_prediction_loops):
        start_i, end_i = i * offset_factor, (i + 1) * offset_factor
        inputs_i = [_split_batch(x, start_i, end_i) for x in inputs]
        kwargs_i = {k: _split_batch(v, start_i, end_i) for k, v in kwargs.items()}
        results_i = func[i](*inputs_i, **kwargs_i)
        if hasattr(results_i, "keys"):
            for k, v in results_i.items():
                results[k].append(v)
        else:
            results = [results_i] if i == 0 else results + [results_i]
    if hasattr(results, "keys"):
        results = {k: torch.cat(v, dim=0) for k, v in results.items()}
    return results
