import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

from src.utilities import wandb_utils as wandb_api
from src.utilities.utils import get_logger


log = get_logger(__name__)


@rank_zero_only
def print_config(
    config,
    fields: Union[str, Sequence[str]] = ("datamodule", "model", "trainer", "seed"),
    resolve: bool = True,
    rich_style: str = "magenta",
    max_width: int = 128,
) -> None:
    """Print DictConfig using Rich (if installed) or plain YAML."""
    import importlib

    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        log.info(OmegaConf.to_yaml(config, resolve=resolve))
        return
    import rich.console
    import rich.syntax
    import rich.tree

    tree = rich.tree.Tree(":gear: CONFIG", style=rich_style, guide_style=rich_style)
    if isinstance(fields, str):
        fields = config.keys() if fields.lower() == "all" else [fields]

    for field in fields:
        branch = tree.add(field, style=rich_style, guide_style=rich_style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.console.Console(width=max_width).print(tree)


def extras(config: DictConfig, allow_permission_error: bool = False) -> DictConfig:
    """Apply standard environment tweaks and wandb setup controlled by the config.

    Handles:
    - Resuming a W&B run when ``logger.wandb.id`` is explicitly set
    - Creating ``work_dir`` (with run-ID subfolder when using W&B)
    - Saving checkpoints under ``<ckpt_dir>/<wandb-run-id>/``
    - Batch-size / accumulate_grad_batches adjustment for multi-GPU
    - CUDA/MPS fallback, num_workers fixes
    - Syncing checkpoint monitor with module monitor
    """
    USE_WANDB = "logger" in config.keys() and config.logger.get("wandb") and hasattr(config.logger.wandb, "_target_")

    if USE_WANDB:
        os.environ["WANDB_HTTP_TIMEOUT"] = "200"
        wandb_cfg = config.logger.wandb
        wandb_api.PROJECT = config.logger.wandb.project = wandb_cfg.get("project", wandb_api.PROJECT)
        wandb_api._ENTITY = config.logger.wandb.entity = wandb_api.get_entity(wandb_cfg.get("entity"))

        # Resume an existing run when ``logger.wandb.id`` is set; otherwise start a fresh run.
        if wandb_cfg.get("id"):
            config.logger.wandb.resume = "allow"
            log.info(f"Resuming W&B run ID = `{wandb_cfg.id}`")
        else:
            config.logger.wandb.id = wandb_api.get_wandb_id_for_run()
            log.info(f"Starting new W&B run ID = `{config.logger.wandb.id}`")

    # ── work_dir: append wandb run ID if not already present ─────────────────
    if config.get("work_dir"):
        if USE_WANDB and config.logger.wandb.get("id"):
            run_id = str(config.logger.wandb.id)
            if run_id not in config.work_dir:
                with open_dict(config):
                    config.work_dir = os.path.join(config.work_dir, run_id)
                    config.ckpt_dir = os.path.join(config.work_dir, "checkpoints")
                    config.log_dir = os.path.join(config.work_dir, "logs")
                log.info(f"work_dir set to {config.work_dir}")
        try:
            os.makedirs(config.work_dir, exist_ok=True)
        except PermissionError as e:
            if allow_permission_error:
                log.warning(f"PermissionError creating work_dir: {e}")
            else:
                raise

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings (config.ignore_warnings=True)")
        warnings.filterwarnings("ignore")

    with open_dict(config):
        # ── Sync checkpoint monitor with module monitor ───────────────────────
        monitor = config.module.get("monitor") or ""
        if config.get("callbacks") and monitor:
            ckpt = config.callbacks.get("model_checkpoint")
            es = config.callbacks.get("early_stopping")
            if ckpt is not None and ckpt.get("monitor"):
                config.callbacks.model_checkpoint.monitor = monitor
            if es is not None and es.get("monitor"):
                config.callbacks.early_stopping.monitor = monitor

        # ── Checkpoint dirs: embed run ID ─────────────────────────────────────
        if USE_WANDB and config.get("callbacks") and not config.get("eval_mode"):
            run_id = str(config.logger.wandb.id)
            for cb_name in config.callbacks.keys():
                if "model_checkpoint" not in cb_name or config.callbacks.get(cb_name) is None:
                    continue
                d = config.callbacks[cb_name].dirpath
                if run_id not in d:
                    new_dir = os.path.join(d, run_id)
                    config.callbacks[cb_name].dirpath = new_dir
                    try:
                        os.makedirs(new_dir, exist_ok=True)
                    except PermissionError as e:
                        raise PermissionError(f"Cannot create checkpoint dir {new_dir}") from e
                    log.info(f"Checkpoints for '{cb_name}' → {os.path.abspath(new_dir)}")

        if not USE_WANDB:
            config.save_config_to_wandb = False

        # ── num_workers fixes ─────────────────────────────────────────────────
        if config.datamodule.num_workers == 0:
            if config.datamodule.get("prefetch_factor") is not None:
                log.warning("prefetch_factor ignored (num_workers=0); setting to None.")
                config.datamodule.prefetch_factor = None
            if config.datamodule.get("persistent_workers") is True:
                log.warning("persistent_workers ignored (num_workers=0); setting to False.")
                config.datamodule.persistent_workers = False

        # ── GPU count → world size ────────────────────────────────────────────
        n_gpus = config.trainer.get("devices", 1)
        n_nodes = int(config.trainer.get("num_nodes", 1))
        if n_gpus == "auto":
            n_gpus = int(torch.cuda.device_count())
        elif isinstance(n_gpus, str) and "," in n_gpus:
            n_gpus = len(n_gpus.split(","))
        elif isinstance(n_gpus, (list, omegaconf.ListConfig)):
            n_gpus = len(n_gpus)
        world_size = int(n_gpus * n_nodes) if n_gpus > 0 else 1

        # ── eval_batch_size: make divisible by world_size × eval_batch_size ──
        ebs = config.datamodule.get("eval_batch_size", config.datamodule.get("batch_size", 1))
        max_val_samples = config.datamodule.get("max_val_samples")
        if max_val_samples is not None and config.get("eval_mode") in [None, "validate"]:
            if max_val_samples % (ebs * world_size) != 0:
                max_ebs = max_val_samples // world_size
                ebs_new = next((i for i in range(max_ebs, 0, -1) if max_ebs % i == 0 and i <= ebs), None)
                if ebs_new is not None:
                    config.datamodule.eval_batch_size = ebs_new
                    log.info(
                        f"Scaled eval_batch_size {ebs}→{ebs_new} so {max_val_samples} samples "
                        f"divide evenly across {world_size} device(s)."
                    )
                else:
                    raise ValueError(
                        f"max_val_samples={max_val_samples} not divisible by "
                        f"eval_batch_size={ebs} × world_size={world_size}"
                    )

        # ── Batch size per GPU + accumulate_grad_batches ──────────────────────
        if not config.get("eval_mode"):
            batch_size = int(config.datamodule.get("batch_size", 1))
            bs_per_gpu_total = batch_size // world_size
            bs_per_gpu = config.datamodule.get("batch_size_per_gpu") or bs_per_gpu_total
            if bs_per_gpu > bs_per_gpu_total:
                bs_per_gpu = bs_per_gpu_total
            elif bs_per_gpu_total % bs_per_gpu != 0:
                bs_per_gpu = next((i for i in range(bs_per_gpu, 0, -1) if bs_per_gpu_total % i == 0), 1)
            acc = bs_per_gpu_total // bs_per_gpu
            effective_ebs = bs_per_gpu * acc * world_size
            if effective_ebs != batch_size:
                raise ValueError(f"Effective batch size {effective_ebs} ≠ target {batch_size} (within 10%)")
            config.n_gpus = n_gpus
            config.world_size = world_size
            config.effective_batch_size = effective_ebs
            config.datamodule.batch_size = bs_per_gpu
            config.trainer.accumulate_grad_batches = acc
            config.datamodule.pop("batch_size_per_gpu", None)

        # ── CUDA / MPS fallback ───────────────────────────────────────────────
        if not torch.cuda.is_available():
            if config.trainer.get("accelerator") == "gpu":
                if torch.backends.mps.is_available():
                    config.trainer.accelerator = "mps"
                    log.warning("CUDA unavailable, using MPS.")
                else:
                    config.trainer.accelerator = "cpu"
                    log.warning("CUDA unavailable, using CPU.")
                config.trainer.devices = 1

    return config


def get_all_instantiable_hydra_modules(config: DictConfig, module_name: str) -> list:
    """Instantiate all Hydra sub-configs that have a ``_target_`` key."""
    modules = []
    if config.get(module_name):
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                try:
                    modules.append(hydra.utils.instantiate(module_config))
                except omegaconf.errors.InterpolationResolutionError as e:
                    log.warning(f"Could not instantiate {module_config} for {module_name}")
                    raise e
    return modules


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
) -> None:
    """Log hyperparameters to all Lightning loggers."""

    def _to_dict(d):
        if d is None:
            return None
        return {k: _to_dict(v) if isinstance(v, DictConfig) else v for k, v in d.items()}

    log_params: Dict[str, Any] = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.get("seed"),
        "datamodule": _to_dict(config.get("datamodule")),
        "model": _to_dict(config.get("model")),
        "module": _to_dict(config.get("module")),
        "trainer": _to_dict(config.get("trainer")),
        "optim": _to_dict(config["module"].get("optimizer")) if config.get("module") else {},
        "scheduler": _to_dict(config["module"].get("scheduler")) if config.get("module") else {},
        "dirs/work_dir": config.get("work_dir"),
        "dirs/ckpt_dir": config.get("ckpt_dir"),
        "model/params_total": sum(p.numel() for p in model.parameters()),
    }
    if config.get("effective_batch_size"):
        log_params["optim/effective_batch_size"] = config.effective_batch_size
    if config.get("logger") and config.logger.get("wandb"):
        log_params["wandb"] = _to_dict(config.logger.wandb)

    if trainer.logger is not None:
        trainer.logger.log_hyperparams(log_params)


@rank_zero_only
def save_hydra_config_to_wandb(config: DictConfig) -> None:
    """Save the resolved config as a YAML file to the W&B run directory."""
    import wandb

    if not config.get("save_config_to_wandb", True):
        log.info("Hydra config will NOT be saved to W&B (save_config_to_wandb=False).")
        return
    filename = "config.yaml"
    # Avoid clobbering an existing upload, append version suffix if needed
    try:
        run_api = wandb_api.get_run_api(run_path=wandb.run.path)
        existing = [f.name for f in run_api.files()]
    except Exception:
        existing = []
    version = 2
    while filename in existing:
        filename = f"config-v{version}.yaml"
        version += 1

    filepath = os.path.join(wandb.run.dir, filename)
    with open(filepath, "w") as fp:
        OmegaConf.save(config, f=fp, resolve=True)
    wandb.save(filename)
    log.info(f"Config saved to W&B as {filename}")
