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
from collections.abc import Sequence
from typing import Any, Optional, Union

from torch import Tensor, tensor

from src.evaluation.metrics import weighted_mean
from src.evaluation.torchmetrics.functional.regression.crps import _crps_compute, _crps_update
from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from src.evaluation.torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ContinousRankedProbabilityScore.plot"]


class ContinuousRankedProbabilityScore(Metric):
    r"""`Compute ContinousRankedProbabilityScore (CRPS)"""

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_crps: Tensor
    total: Tensor

    def __init__(
        self,
        weights: Optional[Tensor] = None,
        biased: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.biased = biased
        self.weights = weights

        self.add_state("sum_crps", default=tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("sum_crps", default=torch.zeros(error_shape), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_crps, num_obs = _crps_update(preds, target, biased=self.biased, flatten=self.weights is None)
        self.sum_crps = sum_crps if self.total == 0 else self.sum_crps + sum_crps
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        sum_crps = self.sum_crps
        if self.weights is not None:
            sum_crps = weighted_mean(sum_crps, weights=self.weights)
        return _crps_compute(sum_crps, self.total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)
