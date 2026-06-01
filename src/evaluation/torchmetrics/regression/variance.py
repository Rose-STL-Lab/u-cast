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

import torch
from torch import Tensor, tensor

from src.evaluation.metrics import weighted_mean
from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["StdDeviation.plot"]


class StdDeviation(Metric):
    r"""Compute standard deviation or variance."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    var_sum: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = False,
        dim: Union[int, Sequence[int]] = (),  # If None, compute over all dims except first (batch)
        weights: Optional[Tensor] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared
        self.dim = dim
        self.weights = weights
        self.source = source
        if self.weights is not None:
            # Compute dims of weights (-1 if 1D, -2, -1 if 2D, -3, -2, -1 if 3D)
            self.weight_dims = tuple(range(-len(self.weights.shape), 0))
        self.add_state("var_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, data: Tensor, target: Tensor = None) -> None:
        assert self.source is not None, "source must be specified. Otherwise, use `update_single_argument`"
        if self.source == "pred":
            data_to_use = data
        elif self.source == "target":
            data_to_use = target
        else:
            raise ValueError("self.source must be either 'pred' or 'target'. Or use `update_single_argument`.")
        self.update_single_argument(data_to_use)

    def update_single_argument(self, data: Tensor) -> None:
        # The following assert only works when self.dim is a single int
        # assert data.shape[self.dim] > 1, f"Variance requires at least 2 elements in {self.dim=} ({data.shape=})"
        if self.dim is None:
            dim = tuple(range(1, len(data.shape)))
        elif isinstance(self.dim, str) and "except" in self.dim:
            dim = tuple(i for i in range(data.dim()) if str(i) not in self.dim.split("_")[1:])
        else:
            dim = self.dim

        new_var = data.var(dim=dim)
        if self.weights is not None:
            new_var = weighted_mean(new_var, weights=self.weights, dim=self.weight_dims)
        self.var_sum += new_var.sum()
        self.total += new_var.numel()

    def compute(self) -> Tensor:
        var = self.var_sum / self.total
        return var if self.squared else torch.sqrt(var)
