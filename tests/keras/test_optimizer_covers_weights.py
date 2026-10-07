"""The optimizer must receive the model's Dense kernels, not just quantizer state.

Keras 3 on the torch backend keeps layer weights as keras `Variable`s whose `.value` is
an `nn.Parameter` that is never registered on the `nn.Module`. Building an optimizer from
`named_parameters()` therefore silently omits every kernel in the model -- measured on the
CLIC config: 282 kernel/bias tensors, 11.6M elements, all receiving gradients, none
optimized. Training then only moves quantizer state and a handful of norm/query tensors.

Discriminating properties, each with the wrong implementation it catches:
- covers: every trainable kernel/bias is in a group -- catches the original bug exactly.
- classified: kernels land in the non-quantizer group, bitwidths in the quantizer group --
  catches a fix that dumps everything into one group and applies weight decay to bitwidths.
- deduplicated: no tensor appears twice -- `keras_layers()` aggregates sublayer weights, so
  a naive concatenation double-registers and double-applies each update.
- excludes beta: the regularization strength is never optimized -- its 'gradient' is just
  the EBOPs magnitude.
- steps: an optimizer built from these groups actually changes a kernel.
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from parity_utils import make_padded_batch  # noqa: F401  # ty: ignore [unresolved-import]  (pins test conftest paths)
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}


@pytest.fixture(scope="module")
def built():
    model = make_keras_model(seed=0, quant=QUANT)
    inputs, _ = clic_dummy_batch(2)
    with torch.no_grad():
        model.eval()(inputs)
    model.train()
    return model


def _kernels(model):
    return [w for layer in model.keras_layers() for w in layer.weights if ("kernel" in w.path or "bias" in w.path) and w.trainable]


def test_every_kernel_reaches_the_optimizer(built):
    decay, quant = built.trainable_parameter_groups()
    covered = {id(p) for p in decay + quant}
    kernels = _kernels(built)
    assert kernels, "fixture built no kernels — the test would be vacuous"
    missing = [w.path for w in kernels if id(w.value) not in covered]
    assert not missing, f"{len(missing)} of {len(kernels)} kernels absent from the optimizer, e.g. {missing[:3]}"


def test_kernels_are_not_in_named_parameters(built):
    """Pins the underlying keras-on-torch behaviour this fix exists for."""
    registered = {id(p) for p in built.parameters()}
    kernels = _kernels(built)
    assert not [w for w in kernels if id(w.value) in registered], (
        "keras weights now appear in named_parameters() — upstream behaviour changed, and trainable_parameter_groups can be simplified"
    )


def test_groups_are_classified_and_deduplicated(built):
    decay, quant = built.trainable_parameter_groups()
    ids = [id(p) for p in decay + quant]
    assert len(ids) == len(set(ids)), "a tensor appears in more than one group — updates would be applied twice"

    kernel_ids = {id(w.value) for w in _kernels(built)}
    assert kernel_ids <= {id(p) for p in decay}, "kernels must be in the non-quantizer group (they take weight decay)"
    # only TRAINABLE bitwidths: 182 of 528 ship with requires_grad=False and are
    # correctly excluded from any optimizer group.
    bit_ids = {id(p) for n, p in built.named_parameters() if n.rsplit("/", 1)[-1] in {"b", "f", "i"} and p.requires_grad}
    assert bit_ids and bit_ids <= {id(p) for p in quant}, "bitwidths must be in the quantizer group (no weight decay)"


def test_beta_is_never_optimized(built):
    decay, quant = built.trainable_parameter_groups()
    covered = {id(p) for p in decay + quant}
    betas = [p for n, p in built.named_parameters() if n.endswith("/beta")]
    assert not [p for p in betas if id(p) in covered], "beta is a hyperparameter, not a trainable weight"


def test_an_optimizer_step_moves_a_kernel(built):
    decay, quant = built.trainable_parameter_groups()
    kernel = max(_kernels(built), key=lambda w: w.value.numel()).value
    before = kernel.detach().clone()
    opt = torch.optim.SGD([{"params": decay}, {"params": quant}], lr=0.1)
    kernel.grad = torch.ones_like(kernel)
    opt.step()
    assert not torch.equal(before, kernel.detach()), "kernel unchanged after an optimizer step"
