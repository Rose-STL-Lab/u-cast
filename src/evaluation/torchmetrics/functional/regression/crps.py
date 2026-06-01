from typing import Union

import torch
from torch import Tensor

from src.evaluation.torchmetrics.utilities.checks import _check_same_shape


def _crps_update(preds: Tensor, target: Tensor, biased: bool, flatten: bool) -> tuple[Tensor, int]:
    """Update and returns variables required to compute CRPS.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds[0], target)
    if flatten:
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(-1)

    n_members = preds.shape[0]
    skill = (preds - target).abs().mean(dim=0)
    forecasts_diff = torch.unsqueeze(preds, 0) - torch.unsqueeze(preds, 1)
    denom = n_members**2 if biased else n_members * (n_members - 1)
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / denom
    crps = skill - 0.5 * spread
    return crps.sum(dim=0), target.shape[0]


def _crps_compute(sum_crps: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    return sum_crps / num_obs
