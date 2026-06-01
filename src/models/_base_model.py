from __future__ import annotations

from typing import Sequence, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from src.optimization.losses import get_loss
from src.utilities.utils import (
    disable_inference_dropout,
    enable_inference_dropout,
    get_logger,
)


class BaseModel(LightningModule):
    r"""Base class for the forecasting network. Concrete models implement :func:`__init__` and :func:`forward`.

    Handles the shared plumbing: channel/conditioning bookkeeping, the (optionally ensembled) loss,
    and inference-time dropout toggling.
    """

    def __init__(
        self,
        num_input_channels: int = None,
        num_output_channels: int = None,
        num_conditional_channels: int = 0,
        spatial_shape: Union[Sequence[int], int] = None,
        loss_function: str = "mean_squared_error",
        num_training_ensemble_members: int = 1,
    ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        self.save_hyperparameters()
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__)
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_conditional_channels = num_conditional_channels
        self.spatial_shape = spatial_shape

        print_text = f"Model: {self.__class__.__name__} with {self.num_input_channels=}, {self.num_output_channels=}"
        print_text += f", {self.spatial_shape=}"
        print_text += f", {self.num_conditional_channels=}" if self.num_conditional_channels > 0 else ""
        self.log_text.info(print_text)
        self.criterion = get_loss(self.hparams.loss_function)
        assert isinstance(
            self.criterion, torch.nn.Module
        ), f"Expected a ``nn.Module`` criterion, got {type(self.criterion)}."

    def forward(self, inputs: Tensor, dynamical_condition: Tensor = None, static_condition: Tensor = None, **kwargs):
        r"""Standard ML model forward pass (implemented by the concrete model).

        Args:
            inputs: Input tensor of shape :math:`(B, C_{in}, H, W)`.
            dynamical_condition / static_condition: Optional conditioning fields concatenated onto the inputs.
        """
        raise NotImplementedError("Base model is an abstract class!")

    def concat_condition_if_needed(
        self,
        inputs: Tensor,
        dynamical_condition: Tensor = None,
        static_condition: Tensor = None,
    ):
        """Concatenate the (dynamical, static) conditioning fields onto the input channels."""
        conditions = [c for c in (dynamical_condition, static_condition) if c is not None]
        if self.num_conditional_channels == 0:
            assert not conditions, "Conditions were provided but num_conditional_channels is 0."
            return inputs
        if not conditions:
            raise ValueError(f"No conditions provided but {self.num_conditional_channels=}")
        return torch.cat([inputs, *conditions], dim=1)

    def get_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        """Get the loss for the given inputs and targets.

        Args:
            inputs: Input tensor of shape :math:`(B, *, C_{in})`.
            targets: Target tensor of shape :math:`(B, *, C_{out})`.
        """
        if getattr(self.criterion, "require_ensembling", False):
            predictions = self.ensemble_forward(
                inputs, n_ens_mems=self.hparams.num_training_ensemble_members, **kwargs
            )
        else:
            predictions = self(inputs, **kwargs)

        return self.criterion(predictions, targets)

    def ensemble_forward(self, inputs: Tensor, n_ens_mems: int = None, **kwargs):
        """Run ``n_ens_mems`` ensemble members in a single batched forward pass.

        The input and tensor kwargs are tiled along a new leading ensemble axis (flattened into the
        batch dim for the forward), and the output is reshaped back to (ens, batch, ...).
        """

        def tile(x: Tensor) -> Tensor:
            return x.unsqueeze(0).expand(n_ens_mems, *x.shape).reshape(-1, *x.shape[1:])

        inputs_ens = tile(inputs)
        kwargs_ens = {k: (tile(v) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        predictions_raw = self(inputs_ens, **kwargs_ens)
        return predictions_raw.reshape(n_ens_mems, -1, *predictions_raw.shape[1:])  # (ens, batch, ...)

    def predict_forward(self, inputs: Tensor, **kwargs):
        """Forward pass for prediction. By default identical to ``forward``."""
        return self(inputs, **kwargs)

    def enable_inference_dropout(self):
        """Set all dropout layers to training mode"""
        enable_inference_dropout(self)

    def disable_inference_dropout(self):
        """Set all dropout layers to eval mode"""
        disable_inference_dropout(self)
