from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utilities.utils import get_logger


log = get_logger(__name__)

PADDING_MODE_SET = None  # Global variable to store the padding mode so that it is set only once.


def pad_before_conv2d_circular_height_only(
    input: Tensor, weight: Tensor, padding: int, torch_func, **kwargs
) -> Tensor:
    """
    Args:
        torch_func: F.conv2d
        input: (B, C_in, H, W)
        weight: (C_out, C_in, H_k, W_k)
        padding: will be equally applied to left, right, top, bottom. Mode is circular for height, zero for width.
    """
    if padding == 0:
        return torch_func(input, weight, **kwargs)
    if padding < 0:
        raise ValueError("padding should be a non-negative integer")
    # Pad circularly around height
    input = F.pad(input, (0, 0, padding, padding), mode="circular")
    # Conv2d, using zero-padding on width only
    return torch_func(input, weight, **kwargs, padding=(0, padding))


def set_global_padding_mode(padding_mode: str):
    # Check if already set by other modules
    global PADDING_MODE_SET
    if PADDING_MODE_SET is not None:
        if PADDING_MODE_SET != padding_mode:
            raise ValueError(f"Padding mode is already set to ``{PADDING_MODE_SET}``.")
        return

    PADDING_MODE_SET = padding_mode  # Set the global padding mode

    log.info(f"Setting padding mode of Conv's to ``{padding_mode}`` globally.")
    if padding_mode == "circular_height_only":
        torch.nn.functional.conv2d = partial(
            pad_before_conv2d_circular_height_only, torch_func=torch.nn.functional.conv2d
        )
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode!r}. Only 'circular_height_only' is supported.")
