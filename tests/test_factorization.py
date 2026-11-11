"""§4: the bases, their defining identities, and which path each shape takes."""

import numpy as np
import pytest
import torch

from ngd_pion.factorization import basis_congruence, basis_identity_anchor, build_bases
from ngd_pion.reference import congruent_pencil, isotropic_pencil

torch.manual_seed(0)
DT = torch.float64


def spd(n, cond=1e3, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    w = torch.logspace(0, -np.log10(cond), n, dtype=DT)
    return (U * w) @ U.T


def semi_ortho(d_out, d_in, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(max(d_out, d_in), min(d_out, d_in), generator=g, dtype=DT))[0]
    return Q if d_out >= d_in else Q.T


def test_identity_anchor_gives_an_orthogonal_basis():
    C = spd(12)
    b = basis_identity_anchor(C, eps=0.0)
    assert b.orthogonal
    assert (b.P.T @ b.P - torch.eye(12, dtype=DT)).abs().max() < 1e-12
    assert (b.P.T @ C @ b.P - torch.diag(b.lam)).abs().max() < 1e-10


def test_congruence_satisfies_its_defining_identities():
    B, C = spd(11, seed=1), spd(11, seed=2)
    b = basis_congruence(B, C, eps=0.0)
    assert not b.orthogonal
    assert (b.P.T @ B @ b.P - torch.eye(11, dtype=DT)).abs().max() < 1e-9
    assert (b.P.T @ C @ b.P - torch.diag(b.lam)).abs().max() < 1e-9


def test_denominator_is_the_fisher_spectrum():
    b = basis_identity_anchor(spd(6), eps=0.0)
    den = b.denominator
    for i in range(6):
        for j in range(6):
            assert den[i, j] == pytest.approx(0.5 * (b.lam[i] + b.lam[j]).item())


@pytest.mark.parametrize("eps", [0.0, 1e-8, 1e-4])
def test_matches_numpy_reference(eps):
    B, C = spd(9, cond=1e5, seed=3), spd(9, seed=4)
    got = basis_congruence(B, C, eps)
    _, lam_ref = congruent_pencil(B.numpy(), C.numpy(), eps=eps)
    # eigenvalues here span several orders, so the tolerance has to be relative
    assert np.allclose(np.sort(got.lam.numpy()), np.sort(lam_ref), rtol=1e-9, atol=0.0)
    iso = basis_identity_anchor(C, eps)
    _, l_ref = isotropic_pencil(C.numpy(), eps=eps)
    assert np.allclose(np.sort(iso.lam.numpy()), np.sort(l_ref), rtol=1e-11, atol=0.0)


def test_batched_equals_per_item():
    """A batched eigh must give exactly what the loop gives, or fusing is unsafe."""
    Bs = torch.stack([spd(8, seed=s) for s in range(4)])
    Cs = torch.stack([spd(8, seed=10 + s) for s in range(4)])
    batched = basis_congruence(Bs, Cs, eps=1e-8)
    for i in range(4):
        one = basis_congruence(Bs[i], Cs[i], eps=1e-8)
        assert (batched.lam[i] - one.lam).abs().max() < 1e-12


@pytest.mark.parametrize(
    "shape,expect_cheap_in",
    [((16, 16), True), ((24, 12), True), ((12, 24), False)],
)
def test_path_selection_follows_shape_under_semi_orthogonal_init(shape, expect_cheap_in):
    """`W^T W = I` for square and tall `W`; a wide one leaves a projector."""
    d_out, d_in = shape
    W = semi_ortho(d_out, d_in)
    b_in, b_out = build_bases(W, spd(d_in), eps=1e-8)
    assert b_in.orthogonal is expect_cheap_in
    assert b_out.orthogonal, "the out-side anchor is I under S = I, always"


def test_non_orthogonal_init_forces_congruence_in_side():
    """The initialisation actually chosen: nothing is identity, so no shortcut."""
    g = torch.Generator().manual_seed(7)
    W = torch.randn(16, 16, generator=g, dtype=DT) * 0.02
    b_in, b_out = build_bases(W, spd(16), eps=1e-8)
    assert not b_in.orthogonal
    assert b_out.orthogonal


def test_wide_w_leaves_exactly_the_expected_number_of_floored_modes():
    d_out, d_in, eps = 12, 24, 1e-10
    W = semi_ortho(d_out, d_in)
    b_in, _ = build_bases(W, spd(d_in), eps=eps)
    floored = (b_in.lam <= b_in.lam.max() * eps * 1.001).sum().item()
    assert floored == d_in - d_out
