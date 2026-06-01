"""Weights & Biases helpers: entity/project/run-id resolution + the Lightning logger subclass."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import wandb
from pytorch_lightning.utilities import rank_zero_only

from src.utilities.utils import get_logger


log = get_logger(__name__)

# Set by ``src.utilities.config_utils.extras`` from the resolved config.
_ENTITY: Optional[str] = None
PROJECT: str = "u-cast"


def get_entity(entity: Optional[str] = None) -> str:
    if entity is None:
        return _ENTITY or wandb.api.default_entity
    return entity


def _get_api(timeout: int = 100) -> wandb.Api:
    try:
        return wandb.Api(timeout=timeout)
    except wandb.errors.UsageError:
        wandb.login()
        return wandb.Api(timeout=timeout)


def get_run_api(
    run_id: Optional[str] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    run_path: Optional[str] = None,
) -> wandb.apis.public.Run:
    entity = get_entity(entity)
    project = project or PROJECT
    assert run_path is None or run_id is None, "Provide either run_path or run_id, not both."
    run_path = run_path or f"{entity}/{project}/{run_id}"
    api = _get_api()
    api._default_entity = entity
    return api.run(run_path)


def get_wandb_id_for_run() -> str:
    """Return a unique W&B run ID."""
    return wandb.sdk.lib.runid.generate_id()


class MyWandbLogger(pl.loggers.WandbLogger):
    """Thin ``WandbLogger`` wrapper that eagerly initializes the W&B run."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _ = self.experiment  # force wandb.init()

    @rank_zero_only
    def summary_update(self, summary_dict: dict):
        self.experiment.summary.update(summary_dict)
