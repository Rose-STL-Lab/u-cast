import torch
from torch import Tensor

from src.utilities.utils import get_logger


log = get_logger(__name__)


def crps_ensemble(
    truth: Tensor,
    predicted: Tensor,
) -> Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch adaptation of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    but implementing the fair, unbiased CRPS as in Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7)

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is particularly important for small ensembles.
    Anecdotally, the unbiased CRPS leads to slightly smaller (i.e. "better") values than the biased version.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.
    """
    assert truth.ndim == predicted.ndim - 1, f"{truth.shape=}, {predicted.shape=}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    skill = (predicted - truth).abs().mean(dim=0)
    forecasts_diff = torch.unsqueeze(predicted, 0) - torch.unsqueeze(predicted, 1)
    # Using n_members * (n_members - 1) instead of n_members**2 is the fair, unbiased CRPS. Crucial for small ensembles.
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / (n_members * (n_members - 1))
    crps = skill - 0.5 * spread
    return crps


class AbstractWeightedLoss(torch.nn.Module):
    def __init__(
        self,
        weights: Tensor = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def weigh_loss(self, loss) -> Tensor:
        try:
            loss = self.weights * loss
        except RuntimeError as e:
            raise RuntimeError(f"Failed to multiply {self.weights.shape=} by {loss.shape=}.") from e
        return loss.mean()

    def forward(self, preds, targets) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method.")


class WeightedMSE(AbstractWeightedLoss):
    def forward(self, preds, targets) -> Tensor:
        return self.weigh_loss((preds - targets) ** 2)


class WeightedMAE(AbstractWeightedLoss):
    def forward(self, preds, targets) -> Tensor:
        return self.weigh_loss((preds - targets).abs())


class WeightedCRPS(AbstractWeightedLoss):
    require_ensembling: bool = True

    def forward(self, preds, targets) -> Tensor:
        return self.weigh_loss(crps_ensemble(predicted=preds, truth=targets))


def get_loss(name, reduction="mean", **kwargs):
    """Returns the loss function with the given name."""
    name = name.lower().strip().replace("-", "_")
    if name in ["wmse", "weighted_mse"]:
        loss = WeightedMSE(**kwargs)
    elif name in ["wmae", "weighted_mae"]:
        loss = WeightedMAE(**kwargs)
    elif name in ["wcrps", "weighted_crps"]:
        loss = WeightedCRPS(**kwargs)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss
