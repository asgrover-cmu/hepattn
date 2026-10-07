"""Keras/HGQ2 twin of the MaskFormer decoder.

KerasMaskFormerDecoder subclasses MaskFormerDecoder and replaces ONLY layer
construction: the whole forward pass — per-layer task execution, mask-attention
thresholding, attn-mask scatter/detach/unmask_all_false, bidirectional updates,
dynamic queries — is inherited torch glue operating on tensors produced by the
keras layers, so those semantics are reused rather than re-derived.
"""

from functools import partial
from typing import Literal

from torch import nn

from hepattn.keras.attention import build_keras_attention
from hepattn.keras.dense import KerasDense
from hepattn.keras.factory import LayerFactory
from hepattn.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer
from hepattn.models.encoder import Residual
from hepattn.models.norm import get_hybrid_norm_config


class KerasMaskFormerDecoderLayer(MaskFormerDecoderLayer):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        cross_attn_mode: Literal["softmax", "kmeans"] = "softmax",
        kmeans_kwargs: dict | None = None,
        factory: LayerFactory | None = None,
        name: str = "decoder_layer",
    ) -> None:
        """Mirror of MaskFormerDecoderLayer with factory-built attention/dense leaves.

        forward() and set_backend() are inherited from the torch implementation.

        Raises:
            NotImplementedError: If cross_attn_mode="kmeans" is requested.
        """
        if cross_attn_mode != "softmax":
            raise NotImplementedError("KerasMaskFormerDecoderLayer supports cross_attn_mode='softmax' only")

        # skip MaskFormerDecoderLayer.__init__ (it would build torch Attention/Dense); replicate its body
        nn.Module.__init__(self)
        factory = factory or LayerFactory()

        self.dim = dim
        self.bidirectional_ca = bidirectional_ca
        self.cross_attn_mode = cross_attn_mode

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = dict(attn_kwargs or {})
        attn_type = attn_kwargs.pop("attn_type", "torch")
        if attn_type in {"flash", "flash-varlen"}:  # identical math on the dense-mask path
            attn_type = "torch"
        attn_kwargs.pop("window_size", None)
        self.attn_type = attn_type
        dense_kwargs = dict(dense_kwargs or {})

        def attn(part: str):
            return build_keras_attention(
                dim, attn_type=attn_type, qkv_norm=qkv_norm, norm=norm, factory=factory, name=f"{name}_{part}", **attn_kwargs
            )

        residual = partial(Residual, dim=dim)
        self.q_ca = residual(attn("q_ca"), norm=attn_norm)
        self.q_sa = residual(attn("q_sa"), norm=attn_norm)
        self.q_dense = residual(KerasDense(dim, factory=factory, name=f"{name}_q_ffn", **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(attn("kv_ca"), norm=attn_norm)
            self.kv_dense = residual(KerasDense(dim, factory=factory, name=f"{name}_kv_ffn", **dense_kwargs), norm=norm, post_norm=dense_post_norm)


class KerasMaskFormerDecoder(MaskFormerDecoder):
    def __init__(self, *, factory: LayerFactory | None = None, **kwargs) -> None:
        """Mirror of MaskFormerDecoder; only decoder_layers construction is replaced.

        Accepts exactly the MaskFormerDecoder configuration (plus the layer factory).

        Raises:
            NotImplementedError: If local strided attention or kmeans cross-attention is configured.
        """
        # the parent builds throwaway torch layers before they are replaced below, and the torch
        # Attention has no "masked-linformer"; hand it the layer config with a type it accepts
        base_kwargs = dict(kwargs)
        layer_attn_kwargs = kwargs["decoder_layer_config"].get("attn_kwargs") or {}
        if layer_attn_kwargs.get("attn_type") == "masked-linformer":
            base_kwargs["decoder_layer_config"] = {**kwargs["decoder_layer_config"], "attn_kwargs": {**layer_attn_kwargs, "attn_type": "torch"}}
        super().__init__(**base_kwargs)
        factory = factory or LayerFactory()

        if self.local_strided_attn:
            raise NotImplementedError("local_strided_attn is not supported by KerasMaskFormerDecoder")

        decoder_layer_config = dict(kwargs["decoder_layer_config"])
        num_decoder_layers = kwargs["num_decoder_layers"]
        self.decoder_layers = nn.ModuleList([
            KerasMaskFormerDecoderLayer(depth=i, factory=factory, name=f"decoder_l{i}", **decoder_layer_config) for i in range(num_decoder_layers)
        ])
        self.attn_type = "torch"
