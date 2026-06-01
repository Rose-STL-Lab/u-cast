"""Checkpoint (re)loading helpers used by fine-tuning support."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytorch_lightning
import torch

from src.utilities.utils import get_logger


log = get_logger(__name__)


def download_model_from_hf(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    hf_path: Optional[str] = None,
    cache_dir: str = "auto",
) -> str:
    """Download a checkpoint file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository id, e.g. ``"username/repo"``.
        filename: Name of the file inside the repo, e.g. ``"model.ckpt"``.
        hf_path: Combined ``"<repo_id>/<filename>"`` shorthand (used alone).
        cache_dir: Local cache directory; ``"auto"`` picks ``.cache/models/``.
    """
    from huggingface_hub import hf_hub_download

    if hf_path is not None:
        assert repo_id is None and filename is None
        repo_id, filename = hf_path.rsplit("/", 1)
    else:
        assert repo_id is not None and filename is not None

    if cache_dir == "auto":
        cache_dir = ".cache/models/"
    os.makedirs(cache_dir, exist_ok=True)
    log.info(f"Downloading from Hugging Face Hub: {repo_id}/{filename} (cache_dir={cache_dir!r})")
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def get_local_path_and_download_if_needed(
    path: Optional[str], filename: Optional[str] = None, abs_path: bool = False
) -> Optional[str]:
    """Resolve a checkpoint path; download from HF when ``path`` starts with ``hf:``."""
    if path is None:
        return None
    if not os.path.isfile(path) and filename is not None and not path.endswith(filename):
        path = os.path.join(path, filename)
    if path.startswith("hf:"):
        hf_path = path.replace("hf://", "").replace("hf:", "")
        path = download_model_from_hf(hf_path=hf_path)
    if abs_path:
        path = os.path.abspath(path)
    return path


def reload_model_from_config_and_ckpt(
    ckpt_path: str,
    *,
    model: pytorch_lightning.LightningModule,
    print_name: str = "",
) -> pytorch_lightning.LightningModule:
    """Load weights from ``ckpt_path`` into the given ``model``."""
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state["state_dict"], strict=True)

    file_size = os.path.getsize(ckpt_path)
    name = f"{print_name}: " if print_name else ""
    parts = [f"Reloaded {name}{ckpt_path}."]
    if model_state.get("epoch") is not None:
        parts.append(f"Epoch={model_state['epoch']}")
    if model_state.get("global_step") is not None:
        parts.append(f"Global_step={model_state['global_step']}")
    parts.append(f"File size [MB]: {file_size / 1e6:.2f}.")
    if model_state.get("wandb") is not None:
        parts.append(f"\nRun ID: {model_state['wandb']['id']}\t Name: {model_state['wandb']['name']}")
    log.info(" ".join(parts))

    return model


def reload_checkpoint(
    *,
    run_id: Optional[str] = None,
    ckpt_filename: Optional[str] = None,
    local_checkpoint_path: Optional[str] = None,
    **reload_kwargs,
) -> Dict[str, Any]:
    """Reload a checkpoint from a local path (or ``hf://`` URI) into ``reload_kwargs['model']``."""
    if run_id is not None:
        run_id = str(run_id).strip()
    assert isinstance(
        local_checkpoint_path, str
    ), f"local_checkpoint_path must be a string, got {local_checkpoint_path!r}"

    ckpt_path = get_local_path_and_download_if_needed(local_checkpoint_path, ckpt_filename, abs_path=True)
    if os.path.isdir(ckpt_path):
        if run_id and os.path.exists(os.path.join(ckpt_path, run_id, ckpt_filename)):
            ckpt_path = os.path.join(ckpt_path, run_id, ckpt_filename)
        else:
            ckpt_path = os.path.join(ckpt_path, ckpt_filename)
    assert os.path.isfile(ckpt_path), f"Could not find checkpoint at {ckpt_path}"
    log.info(f"Restoring model from local path: {ckpt_path}")

    reloaded_model = reload_model_from_config_and_ckpt(ckpt_path, **reload_kwargs)
    return {"model": reloaded_model, "ckpt_path": ckpt_path}
