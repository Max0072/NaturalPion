"""docs/history/ALGORITHM.md §4 -- the bases that diagonalise the Fisher operator.

The operator on each side is `F(X) = (B X C + C X B)/2`, and it is symmetric in
`B` and `C`. Diagonalising it means finding `P` with `P^T B P = I` and
`P^T C P = diag(lam)`; then `F` acts elementwise as `(lam_i + lam_j)/2`.

Two paths, and which one applies is a property of the *shapes*, not a tuning
choice:

* **identity anchor** -- when one of the pair is `I`, `eigh` of the other is
  already the answer and `P` comes out orthogonal. Under `S = I` this is the
  out-side, always. It is also the in-side whenever `W^T W = I`, which holds
  for square and tall `W` under a semi-orthogonal initialisation.
* **congruence** -- otherwise. Costs an extra `eigh` plus an inverse square
  root, and yields a `P` that is merely invertible.

Everything batches over leading dimensions, so layers of equal shape are
factorised in one `eigh` call rather than one per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .linalg import floor_eigenvalues, is_identity, safe_eigh

__all__ = ["Basis", "basis_identity_anchor", "basis_congruence", "build_bases"]


@dataclass(frozen=True)
class Basis:
    """`P` and the spectrum `lam` that together diagonalise one side.

    `orthogonal` records which path produced it -- useful for diagnostics and
    for asserting in tests that the cheap path was taken when it should be.
    """

    P: torch.Tensor
    lam: torch.Tensor
    orthogonal: bool
    # `None` means `lam` arrives already floored, which is what
    # `basis_identity_anchor` and `basis_congruence` do. A float means the
    # opposite convention: `lam` is raw and the floor belongs on the operator's
    # own spectrum, applied here. See `damp_op.py` for why that is not the same
    # thing.
    eps: float | None = None
    # Exponent on the operator's eigenvalues. `1.0` is the natural gradient,
    # `0.5` is what Adam does to its second moment, `0.0` is no
    # preconditioning at all. See `powered.py`.
    power: float = 1.0
    # Additive Tikhonov damping. Adding a constant here is not an approximation
    # to `(F + lam I)^-1 G`, and it is not the same thing either. With
    # `X = P Y P^T`, adding `lam` to the denominator solves
    #
    #     F(X) + lam * (P P^T)^-1 X (P P^T)^-1  =  G
    #
    # exactly. `basis_congruence` builds `P = B^-1/2 Q`, so `P P^T = B^-1` and
    # the damped operator is `F(X) + lam B X B` -- Tikhonov in the metric of
    # the anchor, which for the in-side is `A`, the input covariance. Checked
    # to machine precision against a dense solve, including at a 1e5 mismatch
    # between the two factors, where damping against the *Euclidean* identity
    # instead is wrong by a factor of 7e3.
    #
    # An orthogonal basis has `P P^T = I` and this reduces to plain `F + lam I`.
    lam_tikhonov: float = 0.0

    @property
    def denominator(self) -> torch.Tensor:
        """`(lam_i + lam_j)/2`, the eigenvalues of `F` in this basis.

        They are the eigenvalues of `F` as a linear operator, and eigenvalues
        are invariant under any change of basis, so a floor applied *here* is a
        property of `F` alone. A floor applied to `lam` beforehand is not:
        `lam` is the spectrum of a pencil, and which pencil depends on how the
        pair was split.

        The `eps` floor and `lam_tikhonov` differ in kind, not in degree.
        `max(d, eps * d_max)` is *relative*: multiply every `d` in a layer by a
        constant and it does not move, while the step it permits grows by that
        constant. It can only ever act on the spread of the spectrum. Measured
        on this model, spread is not what drives the step --
        `rho(angle, nullfrac) = -0.92`, and the largest angle in the model came
        from a layer with no degeneracy at all. `d + lam` acts on every
        eigenvalue including the largest, which is what bounds the step at
        `||G|| / lam` whatever the layer's overall curvature.
        """
        d = 0.5 * (self.lam.unsqueeze(-1) + self.lam.unsqueeze(-2))
        if self.eps is not None:
            d = torch.maximum(d, self.eps * d.amax(dim=(-2, -1), keepdim=True))
        if self.lam_tikhonov:
            d = d + self.lam_tikhonov
        return d if self.power == 1.0 else d.pow(self.power)


def basis_identity_anchor(C: torch.Tensor, eps: float) -> Basis:
    """`F(X) = 2(X C + C X)`. A plain symmetric eigenproblem; `P` orthogonal."""
    lam, Q = safe_eigh(C)
    return Basis(P=Q, lam=floor_eigenvalues(lam, eps), orthogonal=True)


def floored_inverse_root(B: torch.Tensor, eps: float) -> torch.Tensor:
    """`B^{-1/2}` with B's eigenvalues floored at `eps * max`, the first factor of `basis_congruence`."""
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, eps)
    return (Ub * wb.rsqrt().unsqueeze(-2)) @ Ub.transpose(-1, -2)


def basis_congruence_from_root(B_inv_half: torch.Tensor, C: torch.Tensor, eps: float) -> Basis:
    """`basis_congruence` given `B^{-1/2}` already, so one root can serve every matrix that shares `B`."""
    wc, Uc = safe_eigh(C)
    C_floored = (Uc * floor_eigenvalues(wc, eps).unsqueeze(-2)) @ Uc.transpose(-1, -2)
    M = B_inv_half @ C_floored @ B_inv_half
    lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
    return Basis(P=B_inv_half @ Q, lam=floor_eigenvalues(lam, eps), orthogonal=False)


def basis_congruence(B: torch.Tensor, C: torch.Tensor, eps: float) -> Basis:
    """`F(X) = (B X C + C X B)/2` with neither factor the identity.

    Goes through the symmetric root `B^{-1/2}` rather than a Cholesky factor:
    congruence needs only *some* `B = K K^T`, and the symmetric root exposes
    the eigenvalues so the floor applies to them directly. Both spectra are
    floored, which is the whole of the regularisation -- the denominators come
    out positive with nothing further added.
    """
    wb, Ub = safe_eigh(B)
    wb = floor_eigenvalues(wb, eps)
    B_inv_half = (Ub * wb.rsqrt().unsqueeze(-2)) @ Ub.transpose(-1, -2)

    wc, Uc = safe_eigh(C)
    C_floored = (Uc * floor_eigenvalues(wc, eps).unsqueeze(-2)) @ Uc.transpose(-1, -2)

    M = B_inv_half @ C_floored @ B_inv_half
    lam, Q = safe_eigh(0.5 * (M + M.transpose(-1, -2)))
    # The floor goes on every spectrum that ends up in a denominator, not only
    # on the ones being inverted. Flooring `B` and `C` does not make `lam`
    # positive in finite precision: the pencil can spread them over more orders
    # than the working dtype resolves, and the small end then rounds to zero.
    return Basis(P=B_inv_half @ Q, lam=floor_eigenvalues(lam, eps), orthogonal=False)


def build_bases(W: torch.Tensor, A: torch.Tensor, eps: float) -> tuple[Basis, Basis]:
    """Both sides for a weight matrix, taking the cheap path wherever it is valid.

    In-side pair is `(A, W^T W)`, out-side is `(S, W A W^T)` with `S = I`.

    In a batch the cheap in-side path is taken only if *every* member has
    `W^T W = I`; a mixed batch falls back to congruence throughout, which is
    correct and merely slower.
    """
    Wt = W.transpose(-1, -2)
    gram_in = Wt @ W
    gram_out = W @ A @ Wt

    if is_identity(gram_in):
        basis_in = basis_identity_anchor(A, eps)
    else:
        basis_in = basis_congruence(A, gram_in, eps)
    return basis_in, basis_identity_anchor(gram_out, eps)
