from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from tqdm.auto import tqdm

from src.datamodules.era5 import get_dims_of_dataset
from src.evaluation.aggregator import Aggregator
from src.models._base_model import BaseModel
from src.models.modules import padding
from src.models.modules.ema import LitEma
from src.optimization.hybrid_optim import HybridOptim
from src.optimization.muon import Muon, get_muon_momentum
from src.utilities.checkpointing import reload_checkpoint
from src.utilities.utils import (
    get_logger,
    rrearrange,
    run_func_in_sub_batches_and_aggregate,
    subtract_if_present,
    to_DictConfig,
    to_tensordict,
    torch_to_numpy,
)


class UCastExperiment(pl.LightningModule):
    r"""U-Cast Lightning module: probabilistic next-step weather forecasting on ERA5.

    The model predicts a single next step per forward pass; during validation it is rolled out
    autoregressively up to ``prediction_horizon`` lead times.

    Args:
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW).
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR).
        monitor: The name of the metric to monitor, e.g. ``val/avg/crps_normed``.
        mode: The mode of the monitor (``min`` or ``max``).
        ema_decay: Decay of the exponential moving-average of the model weights.
        conv_padding_mode_global: Padding mode applied to every convolutional layer globally.
        num_predictions: Number of (ensemble) predictions to make per input sample during validation.
        num_predictions_in_memory: Sub-batch size for ensemble forward passes during validation.
        from_pretrained_checkpoint_run_id / from_pretrained_local_path / from_pretrained_checkpoint_filename:
            Optional knobs to warm-start training from a pre-trained checkpoint.
        verbose: Whether to log info messages.
    """

    def __init__(
        self,
        model_config: DictConfig,
        datamodule_config: DictConfig,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        ema_decay: float = None,
        conv_padding_mode_global: Optional[str] = None,
        from_pretrained_checkpoint_run_id: Optional[str] = None,
        from_pretrained_local_path: Optional[str] = None,
        from_pretrained_checkpoint_filename: Optional[str] = "last.ckpt",
        num_predictions: int = 1,
        num_predictions_in_memory: int = None,
        name: str = "",
        verbose: bool = True,
    ):
        super().__init__()
        # Save all constructor args to self.hparams.
        self.save_hyperparameters(ignore=["model_config", "datamodule_config", "verbose"])
        self.log_text = get_logger(name=self.__class__.__name__ if name == "" else name)
        self.name = name
        self._datamodule = None
        self.verbose = verbose
        if not self.verbose:
            self.log_text.setLevel(logging.WARN)

        if conv_padding_mode_global is not None:
            padding.set_global_padding_mode(padding_mode=conv_padding_mode_global)

        self.model_config = model_config
        self.datamodule_config = datamodule_config
        self.num_predictions = num_predictions
        if num_predictions_in_memory is None or num_predictions_in_memory > num_predictions:
            self.num_predictions_in_mem = num_predictions
        else:
            self.num_predictions_in_mem = num_predictions_in_memory
        assert (
            num_predictions % self.num_predictions_in_mem == 0
        ), f"{num_predictions_in_memory=} % {num_predictions=} != 0"

        # Infer input/output/spatial dimensions, instantiate model, optionally warm-start.
        self.dims = get_dims_of_dataset(self.datamodule_config)
        self.model = self.instantiate_model()
        # EMA wrapper around the model.
        self.model_ema = LitEma(model=self.model, decay=ema_decay)

        self.reload_weights_from_pretrained_checkpoint()   # Reload weights for stage 2 fine-tuning

        self._start_validation_epoch_time = self._start_epoch_time = None
        # Default epoch / global step values used when running validation without a Trainer.fit().
        self._default_epoch = self._default_global_step = 0
        assert self.horizon == 1, f"U-Cast assumes a single-step horizon, got horizon={self.horizon}."

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        if self._trainer and self.trainer.current_epoch != 0:
            return self.trainer.current_epoch
        return self._default_epoch

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs.

        If no Trainer is attached, this propery is 0.

        """
        if self._trainer and self.trainer.global_step != 0:
            return self.trainer.global_step

        return self._default_global_step

    # --------------------------------- Interface with model
    @property
    def window(self) -> int:
        return self.datamodule_config.get("window", 1)

    @property
    def horizon(self) -> int:
        return self.datamodule_config.get("horizon", 1)

    @property
    def num_prediction_loops(self):
        return self.num_predictions // self.num_predictions_in_mem

    @property
    def datamodule(self) -> pl.LightningDataModule:
        if self._datamodule is None:  # alt: set in ``on_fit_start``  method
            if self._trainer is None:
                return None
            self._datamodule = self.trainer.datamodule
            # Move the normalizer (its means/stds) onto the model's device.
            self._datamodule.normalizer.to(self.device)
        return self._datamodule

    @property
    def normalizer(self):
        """The datamodule's (single) data normalizer."""
        return self.datamodule.normalizer

    def instantiate_model(self) -> BaseModel:
        r"""Instantiate the model (a :class:`BaseModel` subclass) from ``self.model_config``."""
        return hydra.utils.instantiate(
            self.model_config,
            num_input_channels=self.dims["input"] * self.window,
            num_output_channels=self.dims["output"],
            num_conditional_channels=self.dims.get("conditional", 0),
            spatial_shape=self.dims["spatial"],
            _recursive_=False,
        )

    def reload_weights_from_pretrained_checkpoint(self) -> None:
        """Warm-start training from a pre-trained checkpoint, when configured."""
        if self.hparams.from_pretrained_checkpoint_run_id is None:
            return
        assert (
            self.hparams.from_pretrained_local_path is not None
        ), "from_pretrained_local_path must be set when from_pretrained_checkpoint_run_id is set"
        # Loads the checkpoint weights into ``self`` in place.
        reload_checkpoint(
            run_id=self.hparams.from_pretrained_checkpoint_run_id,
            local_checkpoint_path=self.hparams.from_pretrained_local_path,
            ckpt_filename=self.hparams.from_pretrained_checkpoint_filename,
            model=self,
            print_name="Pretrained model",
        )

    def forward(self, *args, **kwargs) -> Any:
        y = self.model(*args, **kwargs)
        return y

    # --------------------------------- Metrics
    def get_epoch_aggregators(self, split: str) -> dict:
        """Return the per-lead-time validation aggregators, keyed by name (e.g. ``t12``)."""
        assert split == "val", f"Invalid split {split}"
        aggregators = self.datamodule.get_epoch_aggregators(
            split=split,
            is_ensemble=self.use_ensemble_predictions(split),
            device=self.device,
        )
        for v in aggregators.values():
            v.to(self.device)
        return aggregators

    def get_dataset_attribute(self, attribute: str, split: str = "train") -> Any:
        """Return the attribute of the dataset."""
        split = "train" if split in ["fit", None] else split
        if hasattr(self, f"_dataset_{split}_{attribute}"):
            # Return the cached attribute
            return getattr(self, f"_dataset_{split}_{attribute}")

        if self.datamodule is None:
            raise ValueError("Cannot get dataset attribute if datamodule is None. Please set datamodule first.")

        dl = {
            "train": self.datamodule.train_dataloader(),
            "val": self.datamodule.val_dataloader(),
        }[split]
        if dl is None:
            return None

        # Try to get the attribute from the dataset
        ds = dl.dataset if isinstance(dl, torch.utils.data.DataLoader) else dl[0].dataset
        if isinstance(ds, torch.utils.data.ConcatDataset):
            ds = ds.datasets[0]
        attr_value = getattr(ds, attribute, getattr(ds, f"_{attribute}", None))
        if attr_value is not None:
            # Cache the attribute
            setattr(self, f"_dataset_{split}_{attribute}", attr_value)

        return attr_value

    @contextmanager
    def ema_scope(self, batch_idx: int = None):
        """Context manager to switch to EMA weights."""
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model, batch_idx=batch_idx)
        try:
            yield None
        finally:
            self.model_ema.restore(self.model.parameters())

    @contextmanager
    def inference_dropout_scope(self):
        """Context manager to switch to inference dropout mode."""
        self.model.enable_inference_dropout()
        try:
            yield None
        finally:
            self.model.disable_inference_dropout()

    def normalize_batch(self, batch: TensorDict | Tensor) -> TensorDict:
        """Normalize a per-variable TensorDict (or a packed tensor) using the datamodule's normalizer."""
        return to_tensordict(self.normalizer.normalize(batch))

    def denormalize_batch(self, x: TensorDict | Tensor) -> TensorDict:
        """Inverse of :meth:`normalize_batch`."""
        return to_tensordict(self.normalizer.denormalize(x))

    def predict_packed(self, inputs: Tensor, **kwargs) -> Dict[str, Tensor]:
        results = self.model.predict_forward(inputs, **kwargs)  # by default, just the forward method
        if torch.is_tensor(results):
            results = {"preds": results}
        return results

    def _predict(self, inputs: Tensor, num_predictions: Optional[int] = None, **kwargs) -> Dict[str, Tensor]:
        """Run the model on ``inputs`` (optionally in ensemble sub-batches) and post-process the predictions.

        Args:
            inputs: Input tensor of shape :math:`(B, *, C_{in})` (same as for :func:`forward`).
            num_predictions: Number of ensemble predictions to make. If None, use the default value.

        Returns:
            A dict ``{"preds": {var: tensor}, "preds_normed": {var: tensor}}`` of per-variable predictions.
        """
        base_num_predictions = self.num_predictions
        self.num_predictions = num_predictions or base_num_predictions
        results = run_func_in_sub_batches_and_aggregate(
            self.predict_packed, inputs, num_prediction_loops=self.num_prediction_loops, **kwargs
        )
        self.num_predictions = base_num_predictions
        return self.postprocess_predictions(results, inputs)

    def postprocess_predictions(self, results: Dict[str, Tensor], inputs: Tensor) -> Dict[str, Tensor]:
        results = self.unpack_predictions(results)  # {t{h}_preds: {var: residual_normed}}
        # Residual → normalized state: add back the latest input frame's channels.
        ndims = len(next(iter(results.values())).keys())  # number of output variables
        inputs = self.unpack_data(inputs[:, -ndims:], input_or_output="input")  # (B, C, H, W) → {var: tensor}
        for k in list(results.keys()):
            results[k] = self.normalizer.normalized_residual_to_normalized(results[k])
            results[k] = to_tensordict({kk: vv + inputs.get(kk, 0) for kk, vv in results[k].items()})

        results = self.reshape_predictions(results)
        # Keep both the normalized predictions (``t{h}_preds_normed``) and denormalized ones (``t{h}_preds``).
        for k in list(results.keys()):
            results[f"{k}_normed"] = results.pop(k)
            results[k] = self.denormalize_batch(results[f"{k}_normed"])
        return results

    def reshape_predictions(self, results: TensorDict) -> TensorDict:
        """Reshape ensemble predictions from (N*B, ...) back to (N, B, ...), in-place over all ``preds`` keys."""
        pred_keys = [k for k in results.keys() if "preds" in k]
        leading = results[pred_keys[0]].shape[0]
        if self.num_predictions > 1 and leading % self.num_predictions == 0:
            for k in pred_keys:
                results[k] = self._reshape_ensemble_preds(results[k])
        return results

    def _packer(self, input_or_output: str):
        assert input_or_output in ("input", "output"), f"Unknown input_or_output: {input_or_output}"
        return self.datamodule.in_packer if input_or_output == "input" else self.datamodule.out_packer

    def pack_data(self, data: Dict[str, Tensor], input_or_output: str) -> Tensor:
        """Pack a per-variable TensorDict into a single channel-stacked tensor."""
        return self._packer(input_or_output).pack(data)

    def unpack_data(self, results, input_or_output: str):
        """Unpack a channel-stacked tensor (or a ``{'preds': tensor}`` dict) into a per-variable TensorDict."""
        packer = self._packer(input_or_output)
        if torch.is_tensor(results):
            return packer.unpack(results)
        results["preds"] = packer.unpack(results["preds"])
        return results

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the packed channel axis of ``results['preds']`` into a per-variable TensorDict."""
        return self.unpack_data(results, input_or_output="output")

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        self._set_loss_weights()

    def _set_loss_weights(self, split: str = "fit") -> None:
        """Inject latitude weights from the datamodule into the criterion if it accepts them."""
        criterion = self.model.criterion
        if not hasattr(criterion, "weights") or criterion.weights is not None:
            return
        loss_weights = self.get_dataset_attribute("loss_weights_tensor", split=split)
        if loss_weights is None:
            self.log_text.warning(f"Criterion expects loss weights but dataset has none for split={split!r}.")
            return
        weights = loss_weights.to(self.device)
        self.log_text.info(f"Setting loss weights of shape {weights.shape} for weighted loss function.")
        assert (weights > 1e-6).all(), f"{weights=}"
        criterion.weights = weights

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the training loss for the given batch (residual-normalized, single-step targets)."""
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        inputs = self.get_inputs(batch, split=split, ensemble=False)

        # Single next-step target (horizon == 1): index the time axis to drop it.
        targets = dynamics[:, self.window, ...]
        prev_input = dynamics[:, self.window - 1, ...]
        extra_kwargs = self.get_extra_model_kwargs(batch, split=split, ensemble=False)
        # Residual target: predict the change from the previous step.
        targets = subtract_if_present(targets, prev_input)
        targets = self.normalizer.normalized_to_residual_normalized(targets)
        targets = self.pack_data(targets, input_or_output="output")

        return self.model.get_loss(inputs=inputs, targets=targets, **extra_kwargs)

    # --------------------------------- Horizon / autoregressive rollout helpers
    def get_horizon(self, split: str) -> int:
        return self.datamodule.get_horizon(split)

    def get_inputs_from_dynamics(self, dynamics: Tensor | Dict[str, Tensor]) -> Tensor | Dict[str, Tensor]:
        return dynamics[:, : self.window, ...]

    def get_condition_from_dynamica_cond(
        self, dynamics: Tensor | Dict[str, Tensor], **kwargs
    ) -> Tensor | Dict[str, Tensor]:
        assert dynamics.shape[1] == self.window + self.horizon, f"{dynamics.shape=}, {self.window=}, {self.horizon=}"
        dynamics_cond = dynamics[:, self.window].unsqueeze(1)
        return self.transform_inputs(dynamics_cond, **kwargs)

    def transform_inputs(self, inputs: Tensor, ensemble: bool = True, **kwargs) -> Tensor:
        inputs = rrearrange(inputs, "b window c ... -> b (window c) ...")
        if ensemble:
            inputs = self.get_ensemble_inputs(inputs, **kwargs)
        return inputs

    def get_extra_model_kwargs(
        self,
        batch: Dict[str, Tensor],
        split: str,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Dict[str, Any]:
        extra_kwargs = dict()
        ensemble_k = ensemble and not is_autoregressive
        for k, v in batch.items():
            if k == "dynamics":
                continue
            elif k == "metadata":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split) if ensemble_k else v
            elif k in ["static_condition"]:
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split) if ensemble else v
            elif k == "dynamical_condition":
                extra_kwargs[k] = self.get_condition_from_dynamica_cond(v, split=split, ensemble=ensemble)
            else:
                raise ValueError(f"Unsupported key {k} in batch")
        return extra_kwargs

    def get_inputs(
        self, batch: Dict[str, Tensor], split: str = None, ensemble: bool = False, is_autoregressive: bool = False
    ) -> Tensor:
        inputs = self.get_inputs_from_dynamics(batch["dynamics"])
        inputs = self.pack_data(inputs, input_or_output="input")
        return self.transform_inputs(inputs, split=split, ensemble=ensemble and not is_autoregressive)

    def get_inputs_and_extra_kwargs(
        self,
        batch: Dict[str, Tensor],
        split: str = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        inputs = self.get_inputs(batch, split, ensemble, is_autoregressive)
        extra_kwargs = self.get_extra_model_kwargs(
            batch, split=split, ensemble=ensemble, is_autoregressive=is_autoregressive
        )
        return inputs, extra_kwargs

    def training_step(self, batch: Any, batch_idx: int):
        r"""One step of training (backpropagation is done on the returned loss)."""
        # ERA5 yields ``batch["dynamics"]`` as a ``dict[str, Tensor]`` (one entry per variable).
        batch["dynamics"] = self.normalize_batch(to_tensordict(batch["dynamics"], find_batch_size_max=True))
        loss = self.get_loss(batch)
        self.log("train/loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_batch_end(self, outputs=None, batch=None, batch_idx: int = None):
        self.model_ema(self.model)

    def on_train_epoch_end(self) -> None:
        log_dict = {"epoch": float(self.current_epoch)}
        if self._start_epoch_time is not None:  # sometimes there's a weird issue in DDP mode where this is not set.
            log_dict["time/train"] = time.time() - self._start_epoch_time
        self.log_dict(log_dict, sync_dist=True)

    # --------------------- evaluation with PyTorch Lightning
    def evaluation_step(self, batch: Any, batch_idx: int, **kwargs) -> Dict[str, Tensor]:
        # ERA5 yields ``batch["dynamics"]`` as a ``dict[str, Tensor]`` (one entry per variable).
        batch["dynamics"] = to_tensordict(batch["dynamics"], find_batch_size_max=True)
        # Keep the (NaN cleaned) raw dynamics around so post-processing can compare against ground truth.
        batch["raw_dynamics"] = batch["dynamics"].clone()
        batch["raw_dynamics"] = self.normalizer.fill_nans(batch["raw_dynamics"])
        batch["dynamics"] = self.normalize_batch(batch["dynamics"])

        with self.ema_scope(batch_idx):
            with self.inference_dropout_scope():
                return self._evaluation_step(batch, batch_idx, **kwargs)

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        aggregators: Dict[str, Aggregator] = None,
    ):
        """Roll out the (optionally autoregressive) forecast for one batch and update aggregators."""
        split = "val"
        prediction_horizon = self.get_horizon(split)
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        target_dynamics_raw = batch.pop("raw_dynamics", None)
        dynamic_conds = batch.pop("dynamical_condition", None)
        if prediction_horizon > 20:
            target_dynamics_raw = target_dynamics_raw.to("cpu") if target_dynamics_raw is not None else None
            dynamic_conds = dynamic_conds.to("cpu") if dynamic_conds is not None else None

        n_outer_loops = prediction_horizon  # horizon == 1, so one autoregressive step per lead time
        dyn_any = target_dynamics_raw if target_dynamics_raw is not None else batch["dynamics"]
        if dyn_any.shape[1] < prediction_horizon:
            raise ValueError(f"Prediction horizon {prediction_horizon} is larger than {dyn_any.shape}[1]")

        batch["dynamics"] = batch["dynamics"][:, : self.window + self.horizon, ...]
        # Autoregressive rollout: each step predicts one frame and feeds it back as the newest input.
        pbar = tqdm(
            range(n_outer_loops),
            desc="Autoregressive Step",
            leave=True,
        )
        for ar_step in pbar:
            total_horizon = ar_step + 1  # horizon == 1

            # Seed the next input window with the most recent (window - 1) frames of the current window.
            ar_window_steps = []
            for i in reversed(range(self.window - 1)):
                frame = batch["dynamics"][:, self.window - i - 1, ...]
                if ar_step == 0:
                    frame = self.get_ensemble_inputs(frame, split=split)
                ar_window_steps.append(frame)

            if dynamic_conds is not None:
                current_slice = slice(ar_step, ar_step + 1 + self.window)
                batch["dynamical_condition"] = dynamic_conds[:, current_slice].to(self.device)

            # One forward pass for this step → {"preds": {var}, "preds_normed": {var}}.
            inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(
                batch, split=split, is_autoregressive=ar_step > 0, ensemble=True
            )
            results = self._predict(inputs, **extra_kwargs)

            ar_init = results["preds_normed"]
            if self.use_ensemble_predictions(split):
                ar_init = rrearrange(ar_init, "N B ... -> (N B) ...")
            ar_window_steps.append(ar_init)

            if not (no_aggregators or target_dynamics_raw is None):
                target_time = self.window + total_horizon - 1
                targets = target_dynamics_raw[:, target_time, ...].to(self.device).contiguous()
                aggregators[f"t{total_horizon}"].update(target_data=targets, gen_data=results["preds"])

            if ar_step < n_outer_loops - 1:
                autoregressive_inputs = torch.stack(ar_window_steps, dim=1)
                if not torch.is_tensor(autoregressive_inputs):
                    for k in list(autoregressive_inputs.keys()):
                        autoregressive_inputs[k.replace("preds", "inputs")] = autoregressive_inputs.pop(k)
                batch["dynamics"] = autoregressive_inputs

    def use_ensemble_predictions(self, split: str) -> bool:
        return self.num_predictions > 1 and split == "val"

    def get_ensemble_inputs(self, inputs_raw: Optional[Tensor], split: str) -> Optional[Tensor]:
        """Stack inputs ``num_predictions`` times along a new ensemble dimension."""
        if inputs_raw is None:
            return None
        if not self.use_ensemble_predictions(split):
            return inputs_raw

        num_predictions = self.num_predictions
        if isinstance(inputs_raw, (dict, TensorDictBase)):
            inputs = {k: self.get_ensemble_inputs(v, split) for k, v in inputs_raw.items()}
            if isinstance(inputs_raw, TensorDictBase):
                original_bs = inputs_raw.batch_size
                inputs = TensorDict(inputs, batch_size=[num_predictions * original_bs[0]] + list(original_bs[1:]))
            return inputs

        inputs = torch.stack([inputs_raw for _ in range(num_predictions)], dim=0)
        inputs = rrearrange(inputs, "N B ... -> (N B) ...")
        return inputs

    def _reshape_ensemble_preds(self, results: TensorDict) -> TensorDict:
        """Reshape (N*B, ...) predictions to (N, B, ...)."""
        batch_size = results.shape[0] // self.num_predictions
        return results.reshape(self.num_predictions, batch_size, *results.shape[1:])

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()
        self.aggregators_val = self.get_epoch_aggregators(split="val")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        results = self.evaluation_step(batch, batch_idx, aggregators=self.aggregators_val, **kwargs)
        return torch_to_numpy(results)

    def on_validation_epoch_end(self) -> None:
        val_stats, total_mean_metrics_all = self._on_eval_epoch_end(
            self.aggregators_val, time_start=self._start_validation_epoch_time
        )
        # If monitoring is enabled, check that it is one of the produced metrics.
        if self.trainer.sanity_checking:
            monitors = [self.monitor]
            for ckpt_callback in self.trainer.checkpoint_callbacks:
                if hasattr(ckpt_callback, "monitor") and ckpt_callback.monitor is not None:
                    monitors.append(ckpt_callback.monitor)
            for monitor in monitors:
                assert monitor in val_stats, (
                    f"Monitor metric {monitor} not found in {val_stats.keys()}. "
                    f"\nTotal mean metrics: {total_mean_metrics_all}"
                )
        return val_stats

    def _on_eval_epoch_end(
        self, aggregators: Dict[str, Callable], time_start: float
    ) -> Tuple[Dict[str, float], List[str]]:
        """Aggregate per-lead-time validation metrics into ``val/...`` and ``val/avg/...`` summaries."""
        label = "val"
        val_stats = {
            "time/val": time.time() - time_start,
            "num_predictions": self.num_predictions,
            "epoch": float(self.current_epoch),
            "global_step": self.global_step,
            "eval_batch_size": self.datamodule_config.get("eval_batch_size"),
            "world_size": self.trainer.world_size,
        }

        per_variable_mean_metrics = defaultdict(list)
        temporal_metrics_logged = 0
        for agg_name, agg in aggregators.items():
            # Determine the lead time from the aggregator name, e.g. "t12" -> 12.
            agg_name_part_with_t, lead_time = None, None
            for substr in (agg.name or "").split("/") + agg_name.split("/"):
                if substr.startswith("t") and substr[1:].isdigit():
                    agg_name_part_with_t, lead_time = substr, int(substr[1:])
                    break

            logs_metrics = agg.compute(prefix=label)
            val_stats.update(logs_metrics)
            if lead_time is None:
                continue

            # Log the temporal metrics with lead_time as the x-axis.
            logs_metrics_no_t = {
                k.replace(f"{agg_name_part_with_t}/", "").replace("//", "/"): v for k, v in logs_metrics.items()
            }
            if temporal_metrics_logged == 0:
                try:
                    wandb.define_metric("lead_time")
                    for k in logs_metrics_no_t.keys():
                        wandb.define_metric(k, step_metric="lead_time")
                except Exception as e:
                    self.log_text.warning(f"Could not define metric 'lead_time' in wandb: {e}.")
            if self.logger is not None:
                self.logger.experiment.log({"lead_time": lead_time, **logs_metrics_no_t})
            temporal_metrics_logged += 1

            for k, v in logs_metrics.items():
                k_base = re.sub(r"t\d+/", "", k.replace(f"{label}/", ""))  # strip label + /t{t} infix
                per_variable_mean_metrics[k_base].append(v)

        # Average each metric over lead times (per variable), then over variables.
        total_mean_metrics = defaultdict(list)
        for k, v in per_variable_mean_metrics.items():
            aggs_mean = np.mean(v)
            k_base = "/".join(k.split("/")[:-1])  # drop the variable name, e.g. "rmse/z500" -> "rmse"
            val_stats[f"{label}/avg/{k}"] = aggs_mean
            total_mean_metrics[f"{label}/avg/{k_base}"].append(aggs_mean)
        total_mean_metrics = {k: np.mean(v) for k, v in total_mean_metrics.items()}
        val_stats.update(total_mean_metrics)

        # sync_dist=False: aggregators are torchmetrics.Metric subclasses, already synced across devices.
        self.log_dict(val_stats, sync_dist=False, prog_bar=False)
        return val_stats, list(total_mean_metrics.keys())

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)

    def _get_optim(
        self,
        optim_name: str,
        model_handle=None,
        muon: DictConfig = None,
        **kwargs,
    ):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.

        Args:
            optim_name: Name of the optimizer to use
            model_handle: Optional model to optimize (defaults to self)
            **kwargs: Additional optimizer arguments
        """
        if optim_name.lower() == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"Unknown optimizer type: {optim_name}")

        model_handle = self if model_handle is None else model_handle
        assert all(k in ["lr", "momentum", "wd"] for k in muon.keys())
        muon_lr = muon.get("lr")
        self.use_muon = muon_lr is not None

        # Handle weight decay setup
        wd_orig = kwargs.get("weight_decay", 0)
        base_lr = kwargs["lr"]

        no_decay_params = {k for k, _ in model_handle.named_parameters() if "weight" not in k or "norm" in k}

        non_muon_weights = ["output_layer", "out_layer", "out_conv", "output_conv"]
        hidden_matrix_params = [
            n
            for n, p in model_handle.named_parameters()
            if p.ndim >= 2 and all(nw not in n for nw in non_muon_weights)
        ]

        # Initialize parameter groups
        param_groups = {}  #  start empty to ensure that only groups with parameters are created
        muon_groups = {}
        no_grad_params = 0

        # Process each parameter
        for name, param in model_handle.named_parameters():
            if not param.requires_grad:
                no_grad_params += 1
                continue

            # Determine learning rate multiplier
            is_muon = self.use_muon and name in hidden_matrix_params
            curr_lr = muon_lr if is_muon else base_lr

            # Determine weight decay group
            use_wd = not (wd_orig > 0 and any(nd in name for nd in no_decay_params))
            if is_muon:
                use_wd = use_wd and muon.wd > 0
                group_key = (curr_lr, use_wd, "muon")
            else:
                # Create group key based on lr and weight decay
                group_key = (curr_lr, use_wd)

            # Initialize group if needed
            param_groups_here = param_groups if not is_muon else muon_groups
            if group_key not in param_groups_here:
                group_kwargs = kwargs.copy()
                group_kwargs["lr"] = curr_lr
                if not use_wd:
                    group_kwargs["weight_decay"] = 0
                if is_muon:
                    group_kwargs["momentum"] = muon.momentum
                    group_kwargs["weight_decay"] = muon.wd
                param_groups_here[group_key] = {"params": [], **group_kwargs}

            param_groups_here[group_key]["params"].append(param)

        # Create optimizer with all parameter groups
        optim = optimizer(list(param_groups.values()))
        if self.use_muon:
            assert len(muon_groups) == 1, f"{muon_groups}"
            optim = HybridOptim([optim, Muon(**list(muon_groups.values())[0])])
        return optim

    def configure_optimizers(self):
        """Build the AdamW(+Muon) optimizer and the per-step warmup+cosine LR scheduler."""
        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k != "name"}
        optimizer = self._get_optim(self.hparams.optimizer.name, **optim_kwargs)

        scheduler_params = to_DictConfig(self.hparams.scheduler)

        # The scheduler counts in optimizer steps, so convert ``max_epochs`` to a step count.
        n_steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        max_epochs = scheduler_params.pop("max_epochs", None)
        if max_epochs is not None:
            assert scheduler_params.get("max_steps") in [None, -1]
            scheduler_params["max_steps"] = max_epochs * n_steps_per_epoch

        scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)
        lr_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": self.monitor}
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @property
    def monitor(self):
        return self.hparams.monitor

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        super().lr_scheduler_step(scheduler, *args, **kwargs)
        if self.use_muon:
            wup = self.hparams.scheduler.get("warmup_steps", 1000)
            cooldown = wup // 10
            mom_max = self.hparams.optimizer.get("muon", {}).get("momentum", 0.95)
            mom_min = mom_max - 0.1
            num_iterations = self.trainer.estimated_stepping_batches
            mom = get_muon_momentum(
                self.global_step,
                num_iterations,
                muon_warmup_steps=wup,
                muon_cooldown_steps=cooldown,
                momentum_min=mom_min,
                momentum_max=mom_max,
            )
            # optimizer will be a HybridOptim
            for group in scheduler.optimizer.optimizers[-1].param_groups:
                assert "momentum" in group.keys()
                group["momentum"] = mom
