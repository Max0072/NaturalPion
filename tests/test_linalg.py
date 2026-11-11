"""Primitives: each property they are relied upon for, checked directly."""

import numpy as np
import pytest
import torch

from ngd_pion.linalg import (
    cayley,
    cayley_newton_schulz,
    exact_fp32,
    floor_eigenvalues,
    floor_spectrum,
    is_identity,
    skew,
)

torch.manual_seed(0)


def rand_skew(n, batch=()):
    M = torch.randn(*batch, n, n, dtype=torch.float64)
    return skew(M)


def test_skew_is_a_projection():
    M = torch.randn(7, 7, dtype=torch.float64)
    S = skew(M)
    assert torch.allclose(S, -S.transpose(-1, -2))
    assert torch.allclose(skew(S), S)


def test_floor_is_identity_above_the_threshold():
    """The defining property: a floor must not move well-determined eigenvalues."""
    w = torch.tensor([1.0, 1e-2, 1e-5, 1e-9, 0.0], dtype=torch.float64)
    out = floor_eigenvalues(w, 1e-4)
    assert out[0] == w[0] and out[1] == w[1]           # above the floor, untouched
    assert torch.all(out[2:] == 1e-4)                  # below it, raised to it


def test_floor_clamps_negative_rounding():
    w = torch.tensor([1.0, -1e-18], dtype=torch.float64)
    assert torch.all(floor_eigenvalues(w, 0.0) >= 0.0)


def test_floor_rejects_negative_eps():
    with pytest.raises(ValueError):
        floor_eigenvalues(torch.ones(3, dtype=torch.float64), -1e-8)


def test_floor_batches_per_matrix():
    """Each matrix in a batch is floored against its own lam_max, not a shared one."""
    w = torch.tensor([[1.0, 1e-9], [1e-6, 1e-15]], dtype=torch.float64)
    out = floor_eigenvalues(w, 1e-3)
    assert out[0, 1] == pytest.approx(1e-3)
    assert out[1, 1] == pytest.approx(1e-9)


def test_floor_spectrum_matches_eigenvalue_floor():
    U = torch.linalg.qr(torch.randn(9, 9, dtype=torch.float64))[0]
    w = torch.tensor([1.0, 0.5, 1e-3, 1e-7, 1e-9, 0.0, 2.0, 0.3, 1e-11], dtype=torch.float64)
    M = (U * w) @ U.T
    got = torch.linalg.eigvalsh(floor_spectrum(M, 1e-4))
    want = torch.sort(floor_eigenvalues(w, 1e-4)).values
    assert torch.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("c", [1e-6, 0.1, 1.0, 50.0])
def test_cayley_is_exactly_orthogonal_at_any_step(c):
    """The property Pion's truncated exponential only has to third order."""
    for n in (5, 16):
        R = cayley(rand_skew(n), c)
        eye = torch.eye(n, dtype=torch.float64)
        assert (R.T @ R - eye).abs().max() < 1e-12
        assert abs(torch.linalg.matrix_norm(R, 2).item() - 1.0) < 1e-12


def test_cayley_never_inflates_where_truncated_exp_does():
    """At a large angle the degree-2 exponential blows up; Cayley cannot."""
    X = rand_skew(32)
    X = X / torch.linalg.matrix_norm(X, 2)
    c = 5.0
    trunc = torch.eye(32, dtype=torch.float64) - c * X + 0.5 * (c * X) @ (c * X)
    assert torch.linalg.matrix_norm(trunc, 2) > 5.0
    assert abs(torch.linalg.matrix_norm(cayley(X, c), 2).item() - 1.0) < 1e-12


def test_cayley_preserves_singular_values():
    W = torch.randn(6, 11, dtype=torch.float64)
    s0 = torch.linalg.svdvals(W)
    W2 = cayley(rand_skew(6), 0.7) @ W @ cayley(rand_skew(11), 0.7)
    assert (torch.linalg.svdvals(W2) - s0).abs().max() < 1e-10


def test_cayley_batches():
    X = rand_skew(8, batch=(4,))
    R = cayley(X, 0.3)
    for i in range(4):
        assert torch.allclose(R[i], cayley(X[i], 0.3))


def test_cayley_agrees_with_reference():
    from ngd_pion.reference import cayley as ref_cayley

    X = rand_skew(12)
    got = cayley(X, 0.37).numpy()
    want = ref_cayley(X.numpy(), 0.37)
    assert np.abs(got - want).max() < 1e-13


def test_is_identity():
    assert is_identity(torch.eye(5, dtype=torch.float64))
    assert not is_identity(2.0 * torch.eye(5, dtype=torch.float64))
    assert not is_identity(torch.randn(4, 6, dtype=torch.float64))


def test_exact_fp32_restores_what_it_changed():
    """TF32 is off inside the guard and the caller's settings survive it.

    What the guard is for cannot be tested here: TF32 exists only on the GPU,
    and this suite runs on CPU. Measured on an RTX PRO 6000 Blackwell, leaving
    it on moves the singular values of a weight by a relative 1.0 over 200
    two-sided steps. So this pins the mechanism, and `scripts/gpu_smoke.py`
    plus `scripts/preflight.py` check the consequence where it exists.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        with exact_fp32():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


# --- Newton-Schulz: measured, tested, and deliberately not wired in ---------
#
# `ngd_pion.legacy.fast` does not use this. It was written, measured and then left
# out, because the rotation angles this method produces are far past the range
# where it is usable -- one to two radians per step at `lr = 1e-3` on small
# problems, against the `||A|| = angle/2 < 0.4` it needs. The primitive stays
# because the measurement behind that decision is worth keeping executable: if
# a real run turns out to have small angles, wiring it back in is one line.


@pytest.mark.parametrize("iters,angle,limit", [
    # residual follows ||A||^(2^(k+1)) with ||A|| = angle/2; limits from that
    (1, 1e-2, 1e-7), (2, 1e-2, 1e-12), (2, 1e-1, 1e-10), (3, 0.5, 1e-9),
])
def test_newton_schulz_matches_its_error_law(iters, angle, limit):
    X = rand_skew(48)
    X = X / torch.linalg.matrix_norm(X, 2)
    R = cayley_newton_schulz(X, angle, iters)
    eye = torch.eye(48, dtype=torch.float64)
    assert (R.T @ R - eye).abs().max() < limit


def test_newton_schulz_agrees_with_cayley_where_it_converges():
    X = rand_skew(32)
    X = X / torch.linalg.matrix_norm(X, 2)
    got = cayley_newton_schulz(X, 0.05, 3)
    assert torch.allclose(got, cayley(X, 0.05), atol=1e-12)


def test_newton_schulz_degrades_quietly_past_its_range():
    """It returns a worse matrix rather than raising, which is why any use of
    it needs a bound on `||A||` rather than trust."""
    X = rand_skew(32)
    X = X / torch.linalg.matrix_norm(X, 2)
    R = cayley_newton_schulz(X, 1.6, 1)          # ||A|| = 0.8
    eye = torch.eye(32, dtype=torch.float64)
    assert (R.T @ R - eye).abs().max() > 1e-2


def test_newton_schulz_batches():
    X = rand_skew(8, batch=(3,))
    R = cayley_newton_schulz(X, 0.1, 2)
    for i in range(3):
        assert torch.allclose(R[i], cayley_newton_schulz(X[i], 0.1, 2))


def test_newton_schulz_rejects_zero_iterations():
    with pytest.raises(ValueError):
        cayley_newton_schulz(rand_skew(8), 0.1, 0)
