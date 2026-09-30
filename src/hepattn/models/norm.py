from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class CustomLayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype through normalization operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        dtype = x.dtype
        return super().forward(x).to(dtype)


class FastLayerNorm(nn.LayerNorm):
    """Slightly faster LayerNorm by setting elementwise_affine=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)

    def forward(self, x):
        dtype = x.dtype
        return super().forward(x).to(dtype)


class CustomRMSNorm(nn.Module):
    """Custom RMSNorm implementation from https://arxiv.org/abs/1910.07467."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.weight


class SimpleRMSNorm(nn.Module):
    """From X-transformers."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x):
        dtype = x.dtype
        return (F.normalize(x, dim=-1) * self.scale).to(dtype)


class DyT(nn.Module):
    """2503.10622."""

    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


def get_hybrid_norm_config(
    norm: str | None, depth: int, hybrid_norm: bool, qkv_norm: bool, dense_norm_placement: Literal["hybridnorm", "legacy"] = "hybridnorm"
) -> tuple[str | None, bool, bool]:
    """Get the normalization configuration for HybridNorm.

    Args:
        norm: The normalization type.
        depth: The layer depth.
        hybrid_norm: Whether to use HybridNorm.
        qkv_norm: Whether to use QKV normalization.
        dense_norm_placement: Where the dense layer's norm goes. "hybridnorm" follows 2503.04598: pre-norm in the
            first layer and post-norm in the layers after it. "legacy" is the placement the model used at the
            clic-paper tag, the other way round: post-norm in the first layer (and in every layer without
            hybrid_norm), pre-norm in the layers after it.

    Returns:
        attn_norm: The normalization to use before attention.
        dense_post_norm: Whether to use post-normalization for the dense layer.
        qkv_norm: Whether to use QKV normalization.

    Raises:
        ValueError: If dense_norm_placement is not a known placement.
    """
    if dense_norm_placement not in {"hybridnorm", "legacy"}:
        raise ValueError(f"Unsupported dense_norm_placement: {dense_norm_placement}. Must be 'hybridnorm' or 'legacy'.")

    qkv_norm = qkv_norm or hybrid_norm
    is_hybrid_subsequent = hybrid_norm and depth > 0
    attn_norm = None if is_hybrid_subsequent else norm
    dense_post_norm = is_hybrid_subsequent if dense_norm_placement == "hybridnorm" else not is_hybrid_subsequent

    return attn_norm, dense_post_norm, qkv_norm


# Mapping of normalization type strings to their corresponding nn.Module classes
# Includes both PyTorch built-ins and custom implementations
NORM_TYPES: dict[str, type[nn.Module]] = {
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": nn.RMSNorm,
    "CustomLayerNorm": CustomLayerNorm,
    "FastLayerNorm": FastLayerNorm,
    "CustomRMSNorm": CustomRMSNorm,
    "SimpleRMSNorm": SimpleRMSNorm,
    "DyT": DyT,
}
