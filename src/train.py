"""Training entry point used by ``run.py``."""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import src.utilities.config_utils as cfg_utils
from src.datamodules.era5 import ERA5DataModule
from src.experiment import UCastExperiment
from src.utilities.utils import get_logger
from src.utilities.wandb_utils import MyWandbLogger


log = get_logger(__name__)


def get_model_and_data(config: DictConfig) -> tuple[UCastExperiment, ERA5DataModule]:
    """Instantiate the (lightning module, datamodule) pair from the Hydra config."""
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        _recursive_=False,
    )
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False, model_config=config.model)
    return model, datamodule


def run_model(config: DictConfig) -> None:
    """Train U-Cast (or run validation if ``config.eval_mode == 'validate'``)."""
    pl.seed_everything(config.seed)
    torch.set_float32_matmul_precision("medium")

    # Apply env tweaks, wandb setup, checkpoint dirs, batch-size adjustment.
    config = cfg_utils.extras(config)

    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")

    uses_wandb = (
        config.get("logger") is not None
        and config.logger.get("wandb") is not None
        and config.logger.wandb.get("id") is not None
    )
    if uses_wandb:
        wandb_loggers = [lg for lg in loggers if isinstance(lg, MyWandbLogger)]
        assert len(wandb_loggers) == 1
        cfg_utils.save_hydra_config_to_wandb(config)

    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    ckpt_path = config.get("ckpt_path")
    eval_mode = config.get("eval_mode")
    if eval_mode is not None:
        assert eval_mode == "validate", f"Only eval_mode='validate' is supported, got {eval_mode!r}"
        assert ckpt_path is not None and os.path.isfile(
            ckpt_path
        ), f"ckpt_path must be a valid file when eval_mode='validate' (got {ckpt_path!r})"

    model, datamodule = get_model_and_data(config)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)

    cfg_utils.log_hyperparameters(
        config=config, model=model, data_module=datamodule, trainer=trainer, callbacks=callbacks
    )

    if eval_mode == "validate":
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model._default_global_step = ckpt.get("global_step", -1)
        model._default_epoch = ckpt.get("epoch", -1)
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Training finished successfully.")

    if uses_wandb:
        import wandb

        wandb.finish()
