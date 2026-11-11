"""The vanilla-Pion baseline, and the properties the comparison depends on."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ngd_pion.linalg import skew
from ngd_pion.pion_baseline import Pion, _truncated_exp
from ngd_pion.reference import generators as ref_generators

DT = torch.float64


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DT)


def orth(n, seed=0):
    return torch.linalg.qr(rand(n, n, seed=seed))[0]


def test_generators_are_the_published_ones():
    """`g_in = W^T G - G^T W`, `g_out = G W^T - W G^T`, as in their pion.py."""
    from ngd_pion.direction import generators

    W, G = rand(9, 6, seed=1), rand(9, 6, seed=2)
    gi, go = generators(W, G)
    ri, ro = ref_generators(W.numpy(), G.numpy())
    assert np.abs(gi.numpy() - ri).max() < 1e-13
    assert np.abs(go.numpy() - ro).max() < 1e-13


def test_truncated_exp_is_the_taylor_series():
    A = skew(rand(7, 7, seed=3)) * 0.1
    want = torch.eye(7, dtype=DT) + A + A @ A / 2
    assert torch.allclose(_truncated_exp(A, 2), want)


def test_truncated_exp_only_approximates_orthogonality():
    """The reason the spectrum drifts rather than holding."""
    X = skew(rand(16, 16, seed=4))
    X = X / torch.linalg.matrix_norm(X, 2)
    eye = torch.eye(16, dtype=DT)
    small = _truncated_exp(0.05 * X, 2)
    large = _truncated_exp(2.0 * X, 2)
    assert 0 < (small.T @ small - eye).abs().max() < 1e-5
    assert (large.T @ large - eye).abs().max() > 0.5


@pytest.mark.parametrize("retraction,limit", [("cayley", 1e-13), ("trunc", None)])
def test_spectrum_holds_only_under_cayley(retraction, limit):
    W0 = orth(20, seed=5)
    s0 = torch.linalg.svdvals(W0)
    p = nn.Parameter(W0.clone())
    opt = Pion([p], lr=1e-2, scaling="rms", momentum="none", retraction=retraction, alternate=False)
    for i in range(200):
        p.grad = rand(20, 20, seed=100 + i)
        opt.step()
    drift = float((torch.linalg.svdvals(p.detach()) - s0).abs().max() / s0.max())
    if limit is None:
        assert drift > 1e-10, "the truncated exponential is expected to drift"
    else:
        assert drift < limit


def test_unscaled_truncated_exponential_diverges():
    """Why Pion cannot be run without its RMS scaling on its own retraction.

    RMS scaling is load-bearing for Pion, not cosmetic: it keeps the rotation
    angle small enough that the truncated exponential's inflation does not
    compound. Switch the scaling off and the same retraction blows up within
    tens of steps. This is the finding that made an "ablated Pion" a different
    optimizer rather than Pion with a part removed (ADR 0008).
    """
    W0 = orth(24, seed=6)
    diverged = {}
    for retraction in ("trunc", "cayley"):
        p = nn.Parameter(W0.clone())
        opt = Pion([p], lr=0.1, scaling="none", momentum="none", retraction=retraction, alternate=False)
        blew_up = False
        for i in range(200):
            p.grad = rand(24, 24, seed=200 + i)
            opt.step()
            if not torch.isfinite(p.detach()).all():
                blew_up = True
                break
        diverged[retraction] = blew_up
    assert diverged["trunc"], "expected the truncated exponential to blow up unscaled"
    assert not diverged["cayley"], "Cayley must survive what the truncation cannot"


def test_rms_scaling_fixes_the_ambient_update_size():
    """`alpha` normalises so the per-element RMS of the ambient update is lr * rms."""
    W = orth(12, seed=7)
    p = nn.Parameter(W.clone())
    lr, rms = 1e-2, 0.2
    opt = Pion([p], lr=lr, scaling="rms", rms=rms, momentum="none", retraction="cayley", alternate=False)
    p.grad = rand(12, 12, seed=8)
    opt.step()
    delta = p.detach() - W
    got = float(delta.pow(2).mean().sqrt())
    assert 0.2 * lr * rms < got < 5 * lr * rms


def test_momentum_none_uses_the_raw_gradient():
    W = orth(8, seed=9)
    outs = {}
    for momentum in ("none", "ambient"):
        p = nn.Parameter(W.clone())
        opt = Pion([p], lr=1e-2, scaling="none", momentum=momentum, retraction="cayley", alternate=False)
        p.grad = rand(8, 8, seed=10)
        opt.step()
        outs[momentum] = p.detach().clone()
    assert not torch.allclose(outs["none"], outs["ambient"])


def test_alternate_moves_one_side_at_a_time():
    W = orth(10, seed=11)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, scaling="none", momentum="none", retraction="cayley", alternate=True)
    p.grad = rand(10, 10, seed=12)
    opt.step()
    # step 0 is the out-side: W <- R W, so W^T W is unchanged
    assert (p.detach().T @ p.detach() - W.T @ W).abs().max() < 1e-12


def test_invalid_options_are_refused():
    p = nn.Parameter(rand(4, 4, seed=13))
    for kwargs in (dict(scaling="x"), dict(momentum="x"), dict(retraction="x")):
        with pytest.raises(ValueError):
            Pion([p], lr=1e-3, **kwargs)
    with pytest.raises(ValueError, match="2-D"):
        Pion([nn.Parameter(torch.zeros(3))], lr=1e-3)


def test_lie_momentum_keeps_buffers_on_the_generators():
    """Their `lie_lie`: separate buffers on g_in and g_out, both staying skew.

    This is the variant their only published 60M numbers come from, so the
    anchor run needs it even though their 60M script sets the ambient one.
    """
    W = orth(10, seed=14)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, scaling="none", momentum="lie", retraction="cayley", alternate=False)
    for i in range(3):
        p.grad = rand(10, 10, seed=300 + i)
        opt.step()
    state = opt.state[p]
    assert {"m_in", "m_out", "v_in", "v_out"} <= set(state)
    for key in ("m_in", "m_out"):
        assert (state[key] + state[key].T).abs().max() < 1e-14, "generator momentum must stay skew"


def test_the_three_momentum_variants_differ():
    W = orth(9, seed=15)
    outs = {}
    for momentum in ("none", "ambient", "lie"):
        p = nn.Parameter(W.clone())
        opt = Pion([p], lr=1e-2, scaling="none", momentum=momentum, retraction="cayley", alternate=False)
        for i in range(3):
            p.grad = rand(9, 9, seed=400 + i)
            opt.step()
        outs[momentum] = p.detach().clone()
    keys = list(outs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not torch.allclose(outs[keys[i]], outs[keys[j]])


def _stepped(W0, G, **kw):
    """One Pion step on a single weight, returned."""
    p = nn.Parameter(W0.clone())
    opt = Pion([p], lr=1e-2, momentum="none", retraction="cayley", **kw)
    p.grad = G.clone()
    opt.step()
    return p.detach()


def test_alternate_scales_the_side_it_applies():
    """Their `_scale_update_matrix_rms` takes `update_side`; ours must too.

    Normalising against `W @ g_in + g_out @ W` while applying only one side
    calibrates the RMS target against a step twice the size of the one taken.
    Left unfixed it produced a bilateral-to-alternate gap of 0.0355 against
    their published 0.0079 -- the anchor's sharpest test, missed by 4.5x, with
    every entry in KNOWN_DIFFERENCES unable to explain it because they act on
    both arms alike.
    """
    W, G = rand(6, 4, seed=1), rand(6, 4, seed=2)
    g_in, g_out = ref_generators(W.numpy(), G.numpy())
    g_in, g_out = torch.as_tensor(g_in), torch.as_tensor(g_out)
    group = dict(scaling="rms", rms=0.2, lr=1e-2)

    one = Pion._scale(W, g_in, g_out, group, "out")
    both = Pion._scale(W, g_in, g_out, group, "both")
    assert one != pytest.approx(both), "the side must reach the scale"

    m, n = W.shape
    expected = 1e-2 * 0.2 * np.sqrt(m * n) / torch.linalg.matrix_norm(g_out @ W, "fro")
    assert one == pytest.approx(float(expected), rel=1e-9)


def test_alternate_takes_one_side_per_step_starting_with_out():
    """`"in" if step % 2 == 1 else "out"`, their `_effective_update_side`.

    Step 0 rotates on the left and leaves the row space alone; step 1 rotates
    on the right. Getting the phase backwards would swap which arm of the
    anchor is which without changing anything a test of norms could see.
    """
    W, G = rand(5, 5, seed=3), rand(5, 5, seed=4)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, momentum="none", retraction="cayley",
               scaling="none", alternate=True)
    p.grad = G.clone()
    opt.step()
    # left multiplication only: W^T W is unchanged, W W^T is not
    assert torch.allclose(p.detach().T @ p.detach(), W.T @ W, atol=1e-10)
    p.grad = G.clone()
    opt.step()
    # right multiplication only, now against the weight the first step left
    after_one = _stepped(W, G, scaling="none", alternate=True)
    assert torch.allclose(p.detach() @ p.detach().T, after_one @ after_one.T, atol=1e-10)


def test_row_blocks_rotate_each_block_on_its_own():
    """Their per-head Q: each block gets its own generators and its own scale.

    A block-split weight is not the same update as the whole matrix rotated
    once, and each block keeps its own singular values because each is
    multiplied by its own orthogonal factor.
    """
    W, G = rand(8, 5, seed=5), rand(8, 5, seed=6)
    whole = _stepped(W, G, scaling="rms", alternate=False)
    split = _stepped(W, G, scaling="rms", alternate=False, row_blocks=None)
    assert torch.allclose(whole, split)

    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, momentum="none", retraction="cayley",
               scaling="rms", alternate=False, row_blocks={id(p): 4})
    p.grad = G.clone()
    opt.step()
    blocked = p.detach()
    assert not torch.allclose(blocked, whole, atol=1e-8)
    for b in range(4):
        sl = slice(b * 2, (b + 1) * 2)
        assert torch.allclose(
            torch.linalg.svdvals(blocked[sl]), torch.linalg.svdvals(W[sl]), atol=1e-10
        )


def test_row_blocks_keep_separate_momentum_but_share_the_step():
    """Blocks must not share momentum buffers, and must share the phase.

    Their code keys momentum by a per-block prefix while `state["step"]` is
    incremented once per parameter, so every block of one weight alternates
    together. Sharing the buffers would mix unrelated gradients; splitting the
    counter would put blocks of one matrix on opposite sides of the update.
    """
    W, G = rand(8, 5, seed=7), rand(8, 5, seed=8)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, momentum="lie", retraction="cayley",
               scaling="rms", alternate=True, row_blocks={id(p): 4})
    p.grad = G.clone()
    opt.step()
    state = opt.state[p]
    assert state["step"] == 1
    assert {f"b{b}_m_in" for b in range(4)} <= set(state)
    assert "m_in" not in state


def test_second_moment_is_off_when_beta2_is_none():
    """Their `--pion-use-second-momentum` defaults off; ours divided regardless.

    `Pion.__init__` defaulted `beta2=0.95` and `build_optimizers` passed no
    value, so every Pion run here normalised the Lie momentum by `sqrt(v)`
    where their published runs do not.
    """
    W, G = rand(5, 5, seed=9), rand(5, 5, seed=10)
    p = nn.Parameter(W.clone())
    opt = Pion([p], lr=1e-2, momentum="lie", retraction="cayley",
               scaling="none", alternate=False, beta2=None)
    p.grad = G.clone()
    opt.step()
    assert "v_in" not in opt.state[p]
    assert torch.allclose(opt.state[p]["m_in"], 0.1 * skew(
        torch.as_tensor(ref_generators(W.numpy(), G.numpy())[0])), atol=1e-10)
