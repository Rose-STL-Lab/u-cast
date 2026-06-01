r"""Root package info."""

import logging as __logging
import os

from lightning_utilities.core.imports import package_available


_logger = __logging.getLogger("src.evaluation.torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

if package_available("numpy"):
    # compatibility for AttributeError: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead
    import numpy

    numpy.Inf = numpy.inf

from src.evaluation.torchmetrics import functional  # noqa: E402
from src.evaluation.torchmetrics.metric import Metric  # noqa: E402
from src.evaluation.torchmetrics.regression import (  # noqa: E402
    ContinuousRankedProbabilityScore,
    MeanAbsoluteError,
    MeanSquaredError,
    SpreadSkillRatio,
)


__all__ = [
    "ContinuousRankedProbabilityScore",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "Metric",
    "SpreadSkillRatio",
    "functional",
]
