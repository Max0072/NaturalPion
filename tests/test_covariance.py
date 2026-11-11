"""§3: the covariance accumulator, including the precision guard."""

import pytest
import torch

from ngd_pion.covariance import CovarianceAccumulator

torch.manual_seed(0)


def test_single_batch_is_the_plain_gram():
    x = torch.randn(64, 8)
    acc = CovarianceAccumulator(dtype=torch.float64)
    acc.observe(x)
    assert torch.allclose(acc.matrix, (x.T @ x).double() / 64)


def test_first_observation_initialises_rather_than_blending_into_zero():
    """Otherwise `A` is scaled by `1 - beta` early, and we divide by its eigenvalues."""
    x = torch.randn(32, 5)
    acc = CovarianceAccumulator(beta=0.99, dtype=torch.float64)
    acc.observe(x)
    assert torch.allclose(acc.matrix, (x.T @ x).double() / 32)


def test_ema_follows_the_recursion():
    a = CovarianceAccumulator(beta=0.9, dtype=torch.float64)
    x1, x2 = torch.randn(16, 4), torch.randn(16, 4)
    a.observe(x1)
    a.observe(x2)
    g1, g2 = (x1.T @ x1).double() / 16, (x2.T @ x2).double() / 16
    assert torch.allclose(a.matrix, 0.9 * g1 + 0.1 * g2)


def test_result_is_psd_and_symmetric():
    acc = CovarianceAccumulator(dtype=torch.float64)
    for _ in range(5):
        acc.observe(torch.randn(128, 12))
    A = acc.matrix
    assert torch.allclose(A, A.T)
    assert torch.linalg.eigvalsh(A).min() > 0


def test_leading_dimensions_are_flattened():
    """Activations arrive as (batch, seq, d); the accumulator must not care."""
    x = torch.randn(4, 7, 6)
    a, b = CovarianceAccumulator(dtype=torch.float64), CovarianceAccumulator(dtype=torch.float64)
    a.observe(x)
    b.observe(x.reshape(-1, 6))
    assert torch.allclose(a.matrix, b.matrix)


def test_low_precision_is_refused():
    """A guard, not a preference: bf16 statistics give a step wrong by 10^3-10^4."""
    for dt in (torch.bfloat16, torch.float16):
        with pytest.raises(ValueError, match="fp32 or fp64"):
            CovarianceAccumulator(dtype=dt)


def test_bad_beta_is_refused():
    for beta in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="beta"):
            CovarianceAccumulator(beta=beta)


def test_bf16_activations_are_upcast_not_rejected():
    """Inputs may be low precision -- the accumulator is what must not be."""
    acc = CovarianceAccumulator(dtype=torch.float32)
    acc.observe(torch.randn(32, 6).bfloat16())
    assert acc.matrix.dtype is torch.float32


def test_empty_batch_is_a_no_op():
    acc = CovarianceAccumulator(dtype=torch.float64)
    acc.observe(torch.randn(0, 5))
    assert not acc.ready


def test_unobserved_accumulator_raises_rather_than_returning_zeros():
    with pytest.raises(RuntimeError, match="no activations"):
        _ = CovarianceAccumulator().matrix


def test_state_round_trips():
    a = CovarianceAccumulator(beta=0.8, dtype=torch.float64)
    a.observe(torch.randn(20, 4))
    b = CovarianceAccumulator(dtype=torch.float64)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.matrix, b.matrix)
    assert b.beta == 0.8 and b.count == a.count
