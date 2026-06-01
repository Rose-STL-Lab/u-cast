from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional

import torch

from src.evaluation.torchmetrics import (
    ContinuousRankedProbabilityScore,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    SpreadSkillRatio,
)
from src.utilities.utils import get_logger


log = get_logger(__name__)


class Aggregator(Metric):
    """Accumulates area-weighted, per-variable validation metrics over an epoch.

    For every output variable it tracks L1 and RMSE, plus spread-skill-ratio and CRPS when
    evaluating an ensemble. Call :meth:`update` once per batch with the per-variable target/prediction
    TensorDicts, then :meth:`compute` at the end of the epoch to get a flat ``{metric_key: value}`` dict.

    One instance is created per validation lead time; ``name`` (e.g. ``"t12"``) is prepended to every key.
    """

    def __init__(
        self,
        area_weights: Optional[torch.Tensor] = None,
        is_ensemble: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._area_weights = area_weights
        self.is_ensemble = is_ensemble
        self.name = name
        self._variable_metrics: Optional[Dict[str, Dict[str, Metric]]] = None

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, Metric]]:
        """Lazily build one metric object per (metric, variable), on first update."""
        if self._variable_metrics is not None:
            return self._variable_metrics
        self._variable_metrics = defaultdict(dict)
        weights = None if self._area_weights is None else self._area_weights.to(self.device)
        for var_name in gen_data.keys():
            self._variable_metrics["l1"][var_name] = MeanAbsoluteError(weights=weights)
            self._variable_metrics["rmse"][var_name] = MeanSquaredError(weights=weights, squared=False)
            if self.is_ensemble:
                self._variable_metrics["ssr"][var_name] = SpreadSkillRatio(weights=weights, ensemble_dim=0)
                self._variable_metrics["crps"][var_name] = ContinuousRankedProbabilityScore(weights=weights)
        for var_metrics in self._variable_metrics.values():
            for metric in var_metrics.values():
                metric.to(self.device)
        return self._variable_metrics

    @torch.inference_mode()
    def update(self, target_data: Mapping[str, torch.Tensor], gen_data: Mapping[str, torch.Tensor]) -> None:
        if len(gen_data) == 0 or len(target_data) == 0:
            raise ValueError("Empty target_data or gen_data passed to Aggregator.update.")
        for metric_name, var_metrics in self._get_variable_metrics(gen_data).items():
            for var_name, preds in gen_data.items():
                # SSR/CRPS consume the full ensemble; deterministic metrics use the ensemble mean.
                pred = preds if (metric_name in ("ssr", "crps") or not self.is_ensemble) else preds.mean(dim=0)
                var_metrics[var_name].update(pred, target_data[var_name])

    @torch.inference_mode()
    def compute(self, prefix: str = "") -> Dict[str, float]:
        if self._variable_metrics is None:
            raise ValueError("No batches were recorded before Aggregator.compute().")
        full_prefix = "/".join(p for p in (prefix, self.name) if p)
        logs = {}
        for metric_name, var_metrics in self._variable_metrics.items():
            for var_name, metric in var_metrics.items():
                logs[f"{full_prefix}/{metric_name}/{var_name}".strip("/")] = float(metric.compute().item())
        return logs

    def _apply(self, fn, **kwargs):
        # torchmetrics doesn't track our metrics as submodules, so move them explicitly on .to()/.cpu().
        if self._variable_metrics is not None:
            for var_metrics in self._variable_metrics.values():
                for metric in var_metrics.values():
                    metric._apply(fn, **kwargs)
        return super()._apply(fn, **kwargs)
