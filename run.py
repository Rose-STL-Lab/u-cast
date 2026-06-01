"""Training entry point for U-Cast.

Two configs are available (see ``configs/``); select one with Hydra's ``--config-name``:
    config_det   Stage 1: deterministic pre-training (default)
    config_prob  Stage 2: probabilistic CRPS fine-tuning

    # Stage 1: deterministic pre-training (default config):
    python run.py datamodule.data_dir=/my/data
    # Stage 2: probabilistic CRPS fine-tuning (inherits Stage 1 config):
    python run.py --config-name config_prob datamodule.data_dir=/my/data \
        module.from_pretrained_local_path=/path/to/stage1/checkpoints \
        module.from_pretrained_checkpoint_run_id=stage1_run_id

Any value can be overridden from the command line using Hydra dot-notation.
"""

import os

import hydra
from omegaconf import DictConfig

from src.train import run_model
from src.utilities.utils import get_logger


log = get_logger(__name__)


@hydra.main(config_path="configs/", config_name="config_det", version_base=None)
def main(config: DictConfig):
    """Train or evaluate U-Cast from the selected config (config_det / config_prob)."""
    os.environ["WORK_DIR"] = config.work_dir
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return run_model(config)


if __name__ == "__main__":
    if "WANDB_API_KEY" in os.environ:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"])

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
