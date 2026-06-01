from typing import Optional

import torch


def weighted_mean(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim=(),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    try:
        return (tensor * weights).sum(dim=dim, keepdim=keepdim) / weights.expand(tensor.shape).sum(
            dim=dim, keepdim=keepdim
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Error computing weighted mean. tensor.shape={tensor.shape}, weights.shape={weights.shape}, dim={dim}"
        ) from e
