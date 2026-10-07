"""Keras/HGQ2 twin of hepattn.models.encoder.{EncoderLayer, Encoder}.

Residual wrapping (norms, LayerScale, DropPath) and the hybrid-norm schedule are
REUSED from the torch implementation — norms are the float boundary of the
quantized model. Only the attention and feed-forward leaves are factory-built.

The keras encoder implements the dense-mask path (torch-SDPA semantics): sequence
sorting, sliding windows, score mods, and the flash/flex backends are torch-side
performance features with identical math to the dense path, and are rejected here.
A torch-model attn_type of flash/flash-varlen is coerced to "torch" with a warning
so configs written for the GPU model can be ported directly.
"""

import warnings
from functools import partial

import torch
from torch import Tensor, nn

from hepattn.keras.attention import build_keras_attention
from hepattn.keras.dense import KerasDense
from hepattn.keras.factory import LayerFactory
from hepattn.models.encoder import Residual
from hepattn.models.norm import get_hybrid_norm_config

ALLOWED_ATTN_TYPES = {"torch", "flash", "flash-varlen", "linformer", "masked-linformer"}
# flash/flash-varlen are coerced to torch (identical math on the dense-mask path);
# linformer is a distinct backend (KerasLinformerAttention) and is kept as-is.
COERCE_TO_TORCH = {"flash", "flash-varlen"}


class KerasEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str = "LayerNorm",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        attn_type: str = "torch",
        factory: LayerFactory | None = None,
        name: str = "encoder_layer",
    ) -> None:
        """Mirror of hepattn EncoderLayer: residual self-attention + residual feed-forward."""
        super().__init__()
        factory = factory or LayerFactory()

        attn_kwargs = dict(attn_kwargs or {})
        dense_kwargs = dict(dense_kwargs or {})

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs["value_residual"] = value_residual
        attn_kwargs["is_first_layer"] = depth == 0

        self.dim = dim
        residual = partial(Residual, dim=dim, layer_scale=layer_scale, drop_path=drop_path)
        attn = build_keras_attention(dim, attn_type=attn_type, qkv_norm=qkv_norm, norm=norm, factory=factory, name=f"{name}_attn", **attn_kwargs)
        self.attn = residual(attn, norm=attn_norm)
        self.dense = residual(KerasDense(dim, factory=factory, name=f"{name}_ffn", **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class KerasEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        score_mod: str | None = None,
        value_residual: bool = False,
        num_register_tokens: int | None = None,
        factory: LayerFactory | None = None,
        **layer_kwargs,
    ) -> None:
        """Mirror of hepattn Encoder for the dense-mask (torch-SDPA) path.

        Raises:
            ValueError: If windowed attention, score mods, or an inexpressible attn_type is requested.
        """
        super().__init__()
        factory = factory or LayerFactory()

        if attn_type not in ALLOWED_ATTN_TYPES:
            raise ValueError(f"KerasEncoder cannot express attn_type='{attn_type}'")
        resolved_attn_type = "torch" if attn_type in COERCE_TO_TORCH else attn_type
        if attn_type in COERCE_TO_TORCH:
            warnings.warn(f"KerasEncoder coerces attn_type='{attn_type}' to 'torch' (identical math on the dense-mask path)", stacklevel=2)
        if window_size is not None or window_wrap:
            raise ValueError("windowed attention is not supported by KerasEncoder")
        if score_mod is not None:
            raise ValueError("score mods are not supported by KerasEncoder")

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = resolved_attn_type
        self.window_size = None
        self.value_residual = value_residual
        self.num_register_tokens = num_register_tokens

        if self.num_register_tokens is not None:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, dim))
        else:
            self.register_tokens = None

        layer_kwargs = dict(layer_kwargs or {})
        attn_kwargs = dict(layer_kwargs.pop("attn_kwargs", None) or {})
        attn_kwargs.pop("attn_type", None)
        attn_kwargs.pop("window_size", None)
        layer_kwargs["value_residual"] = self.value_residual
        layer_kwargs["attn_kwargs"] = attn_kwargs
        layer_kwargs["attn_type"] = resolved_attn_type

        self.layers = nn.ModuleList([
            KerasEncoderLayer(dim=dim, depth=i, factory=factory, name=f"encoder_l{i}", **layer_kwargs) for i in range(num_layers)
        ])

    def set_backend(self, attn_type: str) -> None:
        if attn_type != "torch":
            raise ValueError(f"KerasEncoder cannot switch to backend '{attn_type}'")

    def forward(self, x: Tensor, x_sort_value: Tensor | None = None, kv_mask: Tensor | None = None, **kwargs) -> Tensor:
        if x_sort_value is not None:
            raise NotImplementedError("input sorting is a windowed-attention feature, not supported by KerasEncoder")

        batch_size = x.shape[0]

        # Add register tokens at the beginning of the sequence
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([register_tokens, x], dim=1)

            if kv_mask is not None:
                register_mask = torch.full((1, self.num_register_tokens), fill_value=True, device=kv_mask.device, dtype=kv_mask.dtype)
                kv_mask = torch.cat([register_mask.expand(batch_size, -1), kv_mask], dim=1)

        initial_values = {} if self.value_residual else None
        for layer in self.layers:
            x = layer(x, attn_mask=None, initial_values=initial_values, kv_mask=kv_mask, **kwargs)

        # Remove register tokens
        if self.register_tokens is not None:
            x = x[:, self.num_register_tokens :]

        return x
