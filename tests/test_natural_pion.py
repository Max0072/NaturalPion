"""Natural Pion: Fast is Algorithm 1 with its speed-ups off, every form keeps each matrix's spectrum, and the
ablation switches and faster tiers behave as documented."""

from __future__ import annotations

import pytest
import torch

from ngd_pion.natural_pion import NaturalPion, NaturalPionFast


SHAPES = [(12, 8), (8, 8), (8, 8), (8, 12), (12, 8)]  # repeated shapes and sizes exercise the batching


# The tests below were written when NaturalPionFast's defaults were the angle rule in fp32 with the floor 1e-4 (before
# 2026-11-11); they compare it with the angle-rule reference NaturalPion, so they keep those settings unless they set
# their own. test_defaults_are_the_method checks the new defaults.
LEGACY = dict(length="spectral", tf32="off", fisher_dtype="fp32", eps=1e-4)


def _fast(params, *args, **kw):
    return NaturalPionFast(params, *args, **{**LEGACY, **kw})


def _weights(seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.3) for s in SHAPES]


def _drive(opt, params, steps: int, seed: int, schedule):
    """Feed identical statistics and gradients, step with a scheduled angle; return the weights per step."""
    g = torch.Generator().manual_seed(seed)
    trace = []
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = schedule(t)
        for p in params:
            m, n = p.shape
            opt.observe(p, torch.randn(64, n, generator=g))
            opt.observe_backward(p, torch.randn(64, m, generator=g) / 64.0)
            p.grad = torch.randn(m, n, generator=g)
        opt.step()
        trace.append([p.detach().clone() for p in params])
    return trace


def _cosine(t, peak=0.2, steps=60):
    import math
    return 1e-5 + 0.5 * (peak - 1e-5) * (1 + math.cos(math.pi * min(1.0, t / steps)))


@pytest.mark.parametrize("t_fac,window", [(5, 2), (25, 6)])
def test_fast_is_algorithm_1_bit_for_bit(t_fac, window):
    """With its speed-ups off and the same cadence, NaturalPionFast takes NaturalPion's steps exactly."""
    a, b = _weights(0), _weights(0)
    ref = NaturalPion(a, lr=0.2, t_fac=t_fac, stats_window=window, beta_a=0.5, beta_d=0.5)
    ours = _fast(b, lr=0.2, t_fac=t_fac, stats_window=window, retraction="cayley", stats_dtype="fp32",
                           batch_sigma=False, share_root=False)
    for t, (wa, wb) in enumerate(zip(_drive(ref, a, 60, 1, _cosine), _drive(ours, b, 60, 1, _cosine))):
        for x, y in zip(wa, wb):
            assert torch.equal(x, y), f"step {t}: max diff {(x - y).abs().max().item():.3e}"


def test_exact_form_keeps_the_spectrum_and_moves_the_weights():
    params = _weights(3)
    before = [torch.linalg.svdvals(p.detach().double()) for p in params]
    start = [p.detach().clone() for p in params]
    _drive(NaturalPion(params, lr=0.2), params, 40, 4, _cosine)
    for p, s0, w0 in zip(params, before, start):
        s1 = torch.linalg.svdvals(p.detach().double())
        assert torch.allclose(s0, s1, rtol=1e-4), (s0 - s1).abs().max()
        assert (p.detach() - w0).abs().max() > 1e-3


def test_statistics_window():
    params = _weights(5)
    opt = _fast(params, lr=0.1, t_fac=10, stats_window=3)
    _drive(opt, params, 1, 6, lambda t: 0.1)          # first step builds the basis
    st = opt.state[params[0]]
    assert st["since_refactor"] == 1
    assert not opt._collecting(params[0])              # 1 < 10 - 3
    st["since_refactor"] = 7
    assert opt._collecting(params[0])                  # 7 >= 10 - 3


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        NaturalPion([torch.nn.Parameter(torch.zeros(3))])
    with pytest.raises(ValueError):
        NaturalPion(_weights(0), t_fac=5, stats_window=6)


def test_row_blocks_rotate_each_block_and_keep_its_spectrum():
    params = _weights(7)
    q = params[1]                                       # 8 x 8, split into 4 blocks of 2 rows
    before = [torch.linalg.svdvals(q.detach()[2 * i:2 * i + 2].double()) for i in range(4)]
    opt = _fast(params, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(q): 4})
    _drive(opt, params, 30, 8, _cosine)
    for i in range(4):
        after = torch.linalg.svdvals(q.detach()[2 * i:2 * i + 2].double())
        assert torch.allclose(before[i], after, rtol=1e-4), i
    assert "blocks" in opt.state[q] and len(opt.state[q]["blocks"]) == 3


def test_one_block_is_the_unsplit_optimizer():
    a, b = _weights(9), _weights(9)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2)
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(b[1]): 1})
    for x, y in zip(_drive(oa, a, 20, 10, _cosine)[-1], _drive(ob, b, 20, 10, _cosine)[-1]):
        assert torch.equal(x, y)


def test_newton_schulz_retraction_matches_cayley_at_small_angles():
    a, b = _weights(11), _weights(11)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley")
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, retraction="ns", ns_iters=2)
    ta, tb = _drive(oa, a, 30, 12, _cosine), _drive(ob, b, 30, 12, _cosine)
    for x, y in zip(ta[-1], tb[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    before = [torch.linalg.svdvals(p.detach().double()) for p in _weights(11)]
    for p, s0 in zip(b, before):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-4)


def test_truncated_exponential_is_close_to_cayley_and_inflates_the_spectrum():
    a, b = _weights(13), _weights(13)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley")
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, retraction="trunc")
    ta, tb = _drive(oa, a, 30, 14, _cosine), _drive(ob, b, 30, 14, _cosine)
    w0 = _weights(13)
    for x, y, z in zip(ta[-1], tb[-1], w0):
        assert (x - y).abs().max() < 5e-2 * (x - z).abs().max()       # third-order differences, 30 steps at 0.2
        s0 = torch.linalg.svdvals(z.detach().double())
        assert (torch.linalg.svdvals(y.double()) >= s0 * (1 - 1e-6)).all()  # T^T T = I + A^4/4 >= I: never shrinks


def _drive_shared(opt, params, steps, seed, share):
    """q, k, v-like: params 1 and 2 (both 8 x 8) read the same input tensor, as a fused pre-norm does."""
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = _cosine(t)
        xs = {}
        for i, p in enumerate(params):
            m, n = p.shape
            if i == 2:
                x = xs[1] if share else xs[1].clone()
                torch.randn(64, n, generator=g)            # keep the generator in step with the other branch
            else:
                x = torch.randn(64, n, generator=g)
            xs[i] = x
            opt.observe(p, x)
            opt.observe_backward(p, torch.randn(64, m, generator=g) / 64.0)
            p.grad = torch.randn(m, n, generator=g)
        opt.step()
    return [p.detach().clone() for p in params]


def test_shared_input_moment_is_bit_identical():
    a, b = _weights(15), _weights(15)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, share_inputs=False)
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2)           # shares by default
    wa, wb = _drive_shared(oa, a, 30, 16, share=False), _drive_shared(ob, b, 30, 16, share=True)
    for x, y in zip(wa, wb):
        assert torch.equal(x, y)
    assert ob.state[b[1]]["cov"] is ob.state[b[2]]["cov"]
    assert oa.state[a[1]]["cov"] is not oa.state[a[2]]["cov"]


def test_bf16_statistics_are_the_gram_of_rounded_inputs():
    from ngd_pion.covariance import CovarianceAccumulator
    x = torch.randn(512, 16, generator=torch.Generator().manual_seed(17)) * 3
    acc = CovarianceAccumulator(beta=0.5, input_dtype=torch.bfloat16)
    acc.observe(x)
    q = x.to(torch.bfloat16).float()
    assert acc.matrix.dtype == torch.float32
    assert torch.allclose(acc.matrix, q.T @ q / 512, rtol=1e-6, atol=1e-6)
    assert torch.linalg.eigvalsh(acc.matrix.double()).min() > 0


def test_bf16_statistics_keep_the_spectrum_and_track_fp32():
    a, b = _weights(19), _weights(19)
    s0 = [torch.linalg.svdvals(p.detach().double()) for p in a]
    wa = _drive(_fast(a, lr=0.2, t_fac=5, stats_window=2), a, 30, 20, _cosine)[-1]
    wb = _drive(_fast(b, lr=0.2, t_fac=5, stats_window=2, stats_dtype="bf16"), b, 30, 20, _cosine)[-1]
    for x, y, z, s in zip(wa, wb, _weights(19), s0):
        assert (x - y).abs().max() < 0.1 * (x - z).abs().max()
        assert torch.allclose(torch.linalg.svdvals(y.double()), s, rtol=1e-4)


def test_newton_schulz_in_correction_form_matches_cayley_and_keeps_the_norm():
    a, b = _weights(21), _weights(21)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, retraction="cayley", stats_dtype="fp32")
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, stats_dtype="fp32")
    ta, tb = _drive(oa, a, 30, 22, _cosine), _drive(ob, b, 30, 22, _cosine)
    for x, y in zip(ta[-1], tb[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    for p, s0 in zip(b, [torch.linalg.svdvals(p.detach().double()) for p in _weights(21)]):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-5)


def test_batched_sigma_matches_the_loop():
    a, b = _weights(25), _weights(25)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, batch_sigma=False, row_blocks={id(a[1]): 4})
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(b[1]): 4})
    for x, y in zip(_drive(oa, a, 30, 26, _cosine)[-1], _drive(ob, b, 30, 26, _cosine)[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()   # batched reductions: round-off only


# --- ablation switches of NaturalPionFast (Section 4.4); defaults are the method -------------------------------

def test_ablated_length_path_with_default_rule_is_the_method():
    """The per-matrix length path, run with the default rule, takes the method's step (round-off only)."""
    a, b = _weights(40), _weights(40)
    oa = _fast(a, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(a[1]): 4})
    ob = _fast(b, lr=0.2, t_fac=5, stats_window=2, row_blocks={id(b[1]): 4})
    ob._ablated = True
    for x, y in zip(_drive(oa, a, 30, 41, _cosine)[-1], _drive(ob, b, 30, 41, _cosine)[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()


@pytest.mark.parametrize("opts", [dict(direction="raw"), dict(length="before"), dict(length="rms"),
                                  dict(length="fisher"), dict(length="quad"), dict(length="fixed"),
                                  dict(sides="each"), dict(planes="equal"), dict(attn_mult=2.0),
                                  dict(a_window=3, d_window=1), dict(stats_window=None, a_every=2)])
def test_ablations_keep_the_spectrum_and_move(opts):
    params = _weights(42)
    spectra = lambda: [torch.linalg.svdvals(b.double()) for p in params                 # the split matrix: per block
                       for b in (p.detach().chunk(4) if p is params[1] else [p.detach()])]
    before, start = spectra(), [p.detach().clone() for p in params]
    _drive(_fast(params, lr=0.05, t_fac=5, row_blocks={id(params[1]): 4}, **{"stats_window": 3, **opts}),
           params, 25, 43, lambda t: 0.05)
    for s0, s1 in zip(before, spectra()):
        assert torch.allclose(s0, s1, rtol=1e-4), (s0 - s1).abs().max()
    for p, w0 in zip(params, start):
        assert (p.detach() - w0).abs().max() > 1e-4


def test_realised_angle_under_each_rule():
    """Spectral and polar steps turn every matrix by theta; the fixed rule does not."""
    for opts, even in ((dict(sides="each"), True), (dict(planes="equal"), True), (dict(length="fixed"), False)):
        params = _weights(44)
        opt = _fast(params, lr=0.05, t_fac=5, stats_window=3, **opts)
        _drive(opt, params, 3, 45, lambda t: 0.05)
        angles = list(opt.angles.values())
        assert len(angles) == len(params)
        if even:
            assert all(abs(a - 0.05) < 1e-3 for a in angles), angles
        else:
            assert max(angles) / min(angles) > 1.5, angles


def test_attention_multiplier_scales_square_matrices_only():
    params = _weights(46)
    opt = _fast(params, lr=0.05, t_fac=5, stats_window=3, attn_mult=2.0)
    _drive(opt, params, 3, 47, lambda t: 0.05)
    for p in params:
        want = 0.1 if p.shape[0] == p.shape[1] else 0.05
        assert abs(opt.angles[id(p)] - want) < 2e-3, (p.shape, opt.angles[id(p)])


def test_separate_statistics_windows():
    params = _weights(48)
    opt = _fast(params, lr=0.1, t_fac=10, stats_window=None, beta_a=0.9, d_window=3, a_every=2)
    _drive(opt, params, 1, 49, lambda t: 0.1)
    p, st = params[0], opt.state[params[0]]
    st["since_refactor"], st["step"] = 2, 3
    assert not opt._collecting(p, "a") and not opt._collecting(p, "d")    # odd step; outside D's window
    st["since_refactor"], st["step"] = 8, 4
    assert opt._collecting(p, "a") and opt._collecting(p, "d")


def test_ablation_arguments_are_checked():
    params = _weights(50)
    for bad in (dict(direction="x"), dict(length="x"), dict(sides="each", length="rms"),
                dict(planes="equal", length="fixed"), dict(attn_mult=0.0), dict(a_every=0)):
        with pytest.raises(ValueError):
            _fast(params, **bad)


def test_newton_schulz_iterations_follow_the_angle():
    from ngd_pion.natural_pion import ns_iterations
    assert ns_iterations(0.05, 2) == 2 and ns_iterations(0.2, 2) == 2
    assert ns_iterations(0.4, 2) == 3 and ns_iterations(0.8, 2) == 4
    assert ns_iterations(1.9, 2) is None


@pytest.mark.parametrize("angle", [0.4, 0.8])
def test_newton_schulz_at_large_angles_keeps_the_spectrum(angle):
    params = _weights(54)
    before = [torch.linalg.svdvals(p.detach().double()) for p in params]
    _drive(_fast(params, lr=angle, t_fac=5, stats_window=3), params, 20, 55, lambda t: angle)
    for p, s0 in zip(params, before):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-5)


def test_bf16_fisher_products_keep_the_spectrum_and_track_fp32():
    a, b = _weights(56), _weights(56)
    before = [torch.linalg.svdvals(p.detach().double()) for p in b]
    ta = _drive(_fast(a, lr=0.1, t_fac=5, stats_window=3), a, 20, 57, lambda t: 0.1)
    tb = _drive(_fast(b, lr=0.1, t_fac=5, stats_window=3, fisher_dtype="bf16"), b, 20, 57, lambda t: 0.1)
    for p, s0 in zip(b, before):
        assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-5)
    for x, y in zip(ta[-1], tb[-1]):
        assert (x - y).abs().max() < 0.05 * x.abs().max()


def test_polar_factor_from_the_eigenproblem_is_the_svd_one():
    from ngd_pion.natural_pion import _polar_skew
    from ngd_pion.linalg import skew
    g = torch.Generator().manual_seed(58)
    X = skew(torch.randn(12, 12, generator=g, dtype=torch.float64))
    U, S, Vh = torch.linalg.svd(X)
    assert torch.allclose(_polar_skew(X), skew(U @ Vh), atol=1e-8)
    s = torch.linalg.svdvals(_polar_skew(X))
    assert torch.allclose(s, torch.ones_like(s), atol=1e-8)


def test_angle_multiplier_by_matrix():
    params = _weights(60)
    opt = _fast(params, lr=0.05, t_fac=5, stats_window=3, angle_mult={id(params[0]): 2.0})
    _drive(opt, params, 3, 61, lambda t: 0.05)
    assert abs(opt.angles[id(params[0])] - 0.1) < 2e-3 and abs(opt.angles[id(params[2])] - 0.05) < 2e-3


def test_batched_rms_rule_is_the_per_matrix_one():
    """Pion's RMS rule on the batched path takes the per-matrix path's steps (round-off only) and keeps the spectrum."""
    a, b = _weights(62), _weights(62)
    oa = _fast(a, lr=6e-4, t_fac=5, stats_window=3, length="rms", row_blocks={id(a[1]): 4})
    ob = _fast(b, lr=6e-4, t_fac=5, stats_window=3, length="rms", row_blocks={id(b[1]): 4})
    assert not oa._ablated
    ob._ablated = True                                    # force the per-matrix path
    for x, y in zip(_drive(oa, a, 20, 63, lambda t: 6e-4)[-1], _drive(ob, b, 20, 63, lambda t: 6e-4)[-1]):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (x - y).abs().max()
    ra = [oa.realised_angles()[id(p)] for p in a]
    rb = [ob.realised_angles()[id(p)] for p in b]
    assert all(abs(x - y) < 1e-4 * max(1.0, y) for x, y in zip(ra, rb)), (ra, rb)


def test_scaling_and_squaring_cayley():
    """Up to angle 0.2 the step is the two-iteration Newton-Schulz one; above, squared rotations stay orthogonal and
    turn the largest plane by close to the requested angle."""
    from ngd_pion.natural_pion import squarings
    assert squarings(0.05) == 0 and squarings(0.2) == 0 and squarings(0.4) == 1 and squarings(0.8) == 2
    for angle in (0.05, 0.4, 1.6, 2.5):
        params = _weights(64)
        before = [torch.linalg.svdvals(p.detach().double()) for p in params]
        _drive(_fast(params, lr=angle, t_fac=5, stats_window=3, stats_dtype="fp32"), params, 10, 65,
               lambda t: angle)
        for p, s0 in zip(params, before):
            assert torch.allclose(torch.linalg.svdvals(p.detach().double()), s0, rtol=1e-5), angle


@pytest.mark.parametrize("tf32", ["upstream", "all"])
def test_tf32_option_runs_and_keeps_the_spectrum(tf32):
    """On a CPU TF32 does not exist, so the trajectory is the fp32 one; the option must only change the arithmetic."""
    a, b = _weights(52), _weights(52)
    ref = NaturalPionFast(a, lr=0.05, t_fac=5, stats_window=3, tf32="off")
    opt = NaturalPionFast(b, lr=0.05, t_fac=5, stats_window=3, tf32=tf32)
    for x, y in zip(_drive(ref, a, 15, 53, lambda t: 0.05)[-1], _drive(opt, b, 15, 53, lambda t: 0.05)[-1]):
        assert torch.equal(x, y)
    with pytest.raises(ValueError):
        NaturalPionFast(_weights(52), tf32="x")


def test_defaults_are_the_method():
    """NaturalPionFast without arguments is the method of the preprint (ADR 0017)."""
    o = NaturalPionFast(_weights(52))
    g = o.param_groups[0]
    assert (o.length, o.tf32, o.fisher_dtype, g["eps"], g["t_fac"], g["stats_window"]) == ("rms", "upstream", "bf16", 1e-3, 25, 6)
