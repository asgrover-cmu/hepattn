"""Lightning module for the Keras/HGQ2 CLIC pflow model.

Kept separate from lightning_module.py so torch-only runs never import keras
(importing keras pins its backend process-wide).
"""

import os

import torch
from lion_pytorch import Lion
from torch import Tensor
from torch.optim import AdamW

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer

# Recompile budget per compiled HGQ2 function. The decoder-scope CLIC model needs up to 13 cache
# entries for one function (one per distinct quantizer shape/rank); past torch's default of 8,
# dynamo silently runs the remaining call sites eagerly.
HGQ_COMPILE_RECOMPILE_LIMIT = 256


def enable_hgq_train_compile() -> None:
    """Compile HGQ2's training-mode quantizer functions (HGQ2's own set_train_compile).

    Needs the HGQ2 build pinned in pyproject.toml, which provides
    hgq.quantizer.internal.fixed_point_quantizer.set_train_compile. HGQ2 still owns the
    quantization code; only its pure per-quantizer functions run through torch.compile, and
    the quantizer state is still assigned eagerly by HGQ2.

    Process-wide effects:
    - torch._dynamo's recompile limits are raised to HGQ_COMPILE_RECOMPILE_LIMIT;
    - hepattn's torch.compile-wrapped loss / matching-cost functions are unwrapped to run
      eagerly. That is how the HGQ runs already execute them (their launchers set
      TORCHDYNAMO_DISABLE=1) and how this path was benchmarked and parity-checked.

    Raises:
        RuntimeError: If TORCHDYNAMO_DISABLE is set (torch.compile would be a silent no-op).
        ImportError: If the installed HGQ2 does not provide set_train_compile.
    """
    if os.environ.get("TORCHDYNAMO_DISABLE", "0") not in {"", "0"}:
        raise RuntimeError("hgq_train_compile=True needs torch.compile, but TORCHDYNAMO_DISABLE is set; unset it in the launcher")
    from hgq.quantizer.internal import fixed_point_quantizer as fpq  # noqa: PLC0415

    if not hasattr(fpq, "set_train_compile"):
        raise ImportError(f"hgq_train_compile=True needs the HGQ2 build pinned in pyproject.toml (no set_train_compile in {fpq.__file__})")
    from hepattn.models import loss as loss_mod  # noqa: PLC0415

    for table in (loss_mod.cost_fns, loss_mod.loss_fns):
        for key, fn in table.items():
            table[key] = getattr(fn, "_torchdynamo_orig_callable", fn)
    cfg = torch._dynamo.config  # noqa: SLF001
    cfg.recompile_limit = max(cfg.recompile_limit, HGQ_COMPILE_RECOMPILE_LIMIT)
    cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, 8 * HGQ_COMPILE_RECOMPILE_LIMIT)
    fpq.set_train_compile(True)


class MPflowHGQ(MPflow):
    """MPflow driving a KerasMaskFormer (float reference or HGQ2 quantization-aware).

    Adds on top of MPflow:
    - the HGQ2 EBOPs regularization term in the aggregated loss,
    - materialization of lazily-built quantized layers before the optimizer is
      created and before checkpoint state is restored (HGQ2 layers size their
      bitwidth variables from the first real batch's static shapes),
    - a quantizer parameter group without weight decay (decaying learned bitwidths
      would silently shrink precision), with the non-trainable beta excluded,
    - registration of the keras weights on the torch module tree, without which they
      are absent from state_dict() and therefore from every checkpoint
      (see KerasMaskFormer.register_keras_parameters),
    - migration of already-materialized keras Variables onto Lightning's device, which
      nn.Module.to() cannot do for the ones no layer registered
      (see KerasMaskFormer.move_keras_variables_to).
    """

    def __init__(
        self,
        name: str,
        model: KerasMaskFormer,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        quantizer_grad_clip: str = "global",
        hgq_train_compile: bool = False,
    ):
        """quantizer_grad_clip: how Trainer(gradient_clip_val=...) treats the quantizer group.

        "global" (default, the historical behaviour): one norm over every parameter. The
        weight gradients (norm ~400-1100 on CLIC) set the clip factor (~1e-4 at 0.1), which
        pushes the bitwidth gradients (~1e-8..1e-6) far below AdamW's eps, so bitwidths
        effectively never update. "separate": the weight group and the quantizer group are
        each clipped to gradient_clip_val by their OWN norm, so weight clipping is
        unchanged and bitwidths keep AdamW-sized steps.

        hgq_train_compile: compile HGQ2's training-mode quantizer functions with
        torch.compile (see enable_hgq_train_compile). Numerically identical to eager HGQ2
        (outputs, losses and state bit-identical; gradients within ~1e-7); on the CLIC
        decoder-scope model at batch 32 it measured 1.65x the step rate and 0.44x the peak
        memory. Requires the pinned HGQ2 build and TORCHDYNAMO_DISABLE unset. Default off.

        Raises:
            ValueError: If quantizer_grad_clip is not "global" or "separate".
        """
        super().__init__(name, model, lrs_config, optimizer, mtl)
        if quantizer_grad_clip not in {"global", "separate"}:
            raise ValueError(f"quantizer_grad_clip must be 'global' or 'separate', got {quantizer_grad_clip!r}")
        self.quantizer_grad_clip = quantizer_grad_clip
        self.hgq_train_compile = hgq_train_compile
        # per-group gradient norms before/after clipping, last step (set track_clip_stats)
        self.track_clip_stats = False
        self.clip_stats: dict[str, float] = {}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None) -> None:
        groups = optimizer.param_groups

        def norms(tag):
            if self.track_clip_stats:
                for gi, g in enumerate(groups):
                    gs = [p.grad for p in g["params"] if p.grad is not None]
                    per_tensor = [torch.linalg.vector_norm(x.float()) for x in gs]
                    self.clip_stats[f"g{gi}_norm_{tag}"] = float(torch.linalg.vector_norm(torch.stack(per_tensor))) if gs else 0.0
                    self.clip_stats[f"g{gi}_absmax_{tag}"] = float(max(x.abs().max() for x in gs)) if gs else 0.0

        norms("pre")
        if self.quantizer_grad_clip == "global" or gradient_clip_val is None:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        else:
            if gradient_clip_algorithm not in {None, "norm"}:
                raise ValueError("quantizer_grad_clip='separate' supports gradient_clip_algorithm='norm' only")
            for g in groups:
                params = [p for p in g["params"] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, gradient_clip_val)
        norms("post")

    def setup(self, stage: str) -> None:
        super().setup(stage)
        assert isinstance(self.model, KerasMaskFormer), "MPflowHGQ requires a KerasMaskFormer model"
        if self.hgq_train_compile:
            enable_hgq_train_compile()
        # Create keras variables on the RANK'S device, not cpu.
        #
        # The old comment here said variables are "created on cpu and moved with the module
        # by Lightning". Measured (polaris/10_ddp_device_probe.py): they are NOT. Module.to()
        # moves all 2027 registered parameters and buffers, and leaves the keras Variables
        # behind on cpu. Single-device runs survive that because keras reads them wherever
        # they are; DDP does not, because torch's _sync_module_states walks a wider set than
        # named_parameters()+named_buffers() and hands NCCL 684 cpu tensors, which fails with
        # "No backend type associated with device type cpu" (measured, run 7598422).
        #
        # self.device is still cpu at setup() -- Lightning has not moved the module yet -- so
        # take the device from the strategy, which setup_environment() has already resolved.
        set_keras_default_device(self._target_device())
        self._materialize_keras_layers(stage)
        # Every lazy keras Variable now exists. Publish them to the torch module tree here,
        # before anything can save a checkpoint or introspect the module: HGQ2 layers never
        # do it themselves, and keras' lazy recovery would otherwise make the state_dict key
        # set depend on whether something happened to traverse submodule .parameters(). It
        # also has to precede on_load_checkpoint(), which checks a restored state against
        # exactly the key set this call fixes.
        self.model.register_keras_parameters()

    def _target_device(self) -> str:
        strategy = getattr(self.trainer, "strategy", None)
        root = getattr(strategy, "root_device", None)
        return str(root) if root is not None else str(self.device)

    def _sync_keras_device(self) -> None:
        # Two distinct things have to follow Lightning's device:
        # 1. where keras creates NEW tensors -- quantizer internals (STE rounding, LUT
        #    domains) otherwise mix cpu constants with cuda activations;
        # 2. where the ALREADY-materialized keras Variables live. set_keras_default_device
        #    does not touch those, and nn.Module.to() only reaches the ones a layer
        #    registered on the torch module tree.
        #
        # setup() now builds on the strategy's root device, so both are normally no-ops.
        # Kept because self.device is authoritative once Lightning has moved the module, and
        # because test/predict can run without a fit having gone through setup() first --
        # in which case the Variables really are somewhere else and have to be migrated.
        set_keras_default_device(str(self.device))
        self.model.move_keras_variables_to(self.device)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._sync_keras_device()

    def on_validation_start(self) -> None:
        self._sync_keras_device()

    def on_test_start(self) -> None:
        self._sync_keras_device()

    def on_predict_start(self) -> None:
        self._sync_keras_device()

    def _materialize_keras_layers(self, stage: str) -> None:
        datamodule = self.trainer.datamodule
        loader_fn = {
            "fit": datamodule.train_dataloader,
            "validate": datamodule.val_dataloader,
        }.get(stage, datamodule.test_dataloader)
        inputs, _ = next(iter(loader_fn()))
        # the loader yields cpu tensors; the layers are being built on the target device
        device = self._target_device()
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.model.to(device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)
        self.model.train(was_training)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Refuse a checkpoint that predates the keras-weight registration fix.

        Such a checkpoint carries the quantizer state, the norms and the optimizer moments
        but none of the network's kernels, so Lightning would restore it onto freshly
        initialized weights and report a successful resume. Fail here, with the reason,
        rather than let strict=True print thousands of missing keys or -- worse -- let a
        non-strict load through.

        Runs after setup(), so register_keras_parameters() has already fixed the key set
        this compares against.

        Raises:
            RuntimeError: If the checkpoint does not carry every key the model expects.
        """
        state = checkpoint.get("state_dict")
        if state is None:
            return
        missing = self.model.missing_state_keys({k.removeprefix("model."): v for k, v in state.items()})
        if missing:
            raise RuntimeError(
                f"this checkpoint is missing {len(missing)} of the model's state keys, including "
                f"{sum(1 for k in missing if k.endswith(('/kernel', '/bias')))} keras kernel/bias tensors "
                f"(e.g. {missing[:3]}). It was written before KerasMaskFormer.register_keras_parameters "
                "existed, so it does not contain the trained network weights and cannot be resumed "
                "faithfully -- only its quantizer state and norms are recoverable."
            )

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str | None = None) -> Tensor:
        total_loss = super().aggregate_losses(losses, stage=stage)
        quant_loss = self.model.quant_losses()
        self.log(f"{stage}/quant_ebops_loss", quant_loss, sync_dist=True)
        return total_loss + quant_loss

    def configure_optimizers(self):
        # Mirrors ModelWrapper.configure_optimizers with quantizer-aware param groups.
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # NOT self.model.named_parameters(): keras layer weights are not registered on
        # the nn.Module, so that list omits every Dense kernel in the model. See
        # KerasMaskFormer.trainable_parameter_groups.
        decay_params, quantizer_params = self.model.trainable_parameter_groups()

        # A tensor in two groups would take its update twice, with two weight decays.
        # Cheap, and the two groups are built from overlapping traversals.
        ids = [id(p) for p in decay_params + quantizer_params]
        assert len(ids) == len(set(ids)), "duplicate parameter objects across optimizer groups"

        param_groups = [{"params": decay_params}, {"params": quantizer_params, "weight_decay": 0.0}]
        opt = optimizer(param_groups, lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
