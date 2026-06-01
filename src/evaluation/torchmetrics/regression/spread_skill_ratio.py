# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

from torch import Tensor

from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.regression.mse import MeanSquaredError
from src.evaluation.torchmetrics.regression.variance import StdDeviation
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["StdDeviation.plot"]


class SpreadSkillRatio(Metric):
    r"""Compute standard deviation or variance."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        biased: bool = False,
        ensemble_dim: int = 0,
        weights: Optional[Tensor] = None,
        # error_shape: Tuple[int, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.biased = biased
        self.weights = weights
        self.ensemble_dim = ensemble_dim
        assert ensemble_dim == 0, "Only ensemble_dim=0 is supported"
        if self.weights is not None:
            # Compute dims of weights (-1 if 1D, -2, -1 if 2D, -3, -2, -1 if 3D)
            self.weight_dims = tuple(range(-len(self.weights.shape), 0))
        self.rmse_agg = MeanSquaredError(squared=False, weights=self.weights)  # , error_shape=error_shape)
        self.spread_agg = StdDeviation(squared=False, weights=self.weights, dim=ensemble_dim)
        self.ensemble_size = None

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.ensemble_size is None:
            self.ensemble_size = preds.shape[self.ensemble_dim]
        assert self.ensemble_size == preds.shape[self.ensemble_dim]
        self.rmse_agg.update(preds.mean(dim=self.ensemble_dim), target)
        self.spread_agg.update_single_argument(preds)

    def compute(self) -> Tensor:
        rmse = self.rmse_agg.compute()
        spread = self.spread_agg.compute()
        if not self.biased:
            spread *= ((self.ensemble_size + 1) / self.ensemble_size) ** 0.5
        return spread / rmse
