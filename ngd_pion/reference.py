"""Spec-faithful reference implementation of NGD-Pion.

Follows `natural_gradient_pion.md` section by section, in dense exact
arithmetic: every operation is the textbook O(d^3) one, no sketching, no
Krylov, no low-rank. Nothing here is meant to be fast -- this module is the
oracle that the optimised implementations get checked against.

Section numbers in the docstrings refer to that document.

Sign convention (sec. 7): `X_in` and `X_out` come out of `natural_gradient`
unsigned, and the descent direction is `Cayley(-eta * alpha * X)`. With that
choice `quad`, `curv` and `alpha_t` are all positive and the step decreases
the loss by `eta * alpha_t * quad` to first order (the half went with the
2026-08-28 generator convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "skew",
    "generators",
    "fisher_apply",
    "fisher_matrix",
    "congruent_pencil",
    "isotropic_pencil",
    "floor_spectrum",
    "natural_gradient",
    "trust_region_alpha",
    "cayley",
    "NGDPionReference",
]


def skew(M: np.ndarray) -> np.ndarray:
    """Project onto the skew-symmetric part, `(M - M^T) / 2`."""
    return 0.5 * (M - M.T)


# --- sec. 1: Pion generators -------------------------------------------------


def generators(W: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`G_in = skew(W^T G)`, `G_out = skew(G W^T)` (sec. 1).

    Convention changed 2026-08-28 together with `direction.generators`: this
    used to return twice the above. See that docstring for what moves and what
    does not.
    """
    return 0.5 * (W.T @ G - G.T @ W), 0.5 * (G @ W.T - W @ G.T)


# --- sec. 2: the Fisher operator ---------------------------------------------


def fisher_apply(B: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """`F(X) = (B X C + C X B)/2`.

    In-side is called with `B = A`, `C = S' = W^T S W`; out-side mirrors it
    with `B = S`, `C = A' = W A W^T`.
    """
    return 0.5 * (B @ X @ C + C @ X @ B)

def fisher_matrix(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """`F` as an explicit `d^2 x d^2` matrix acting on row-major `vec(X)`.

    Uses `vec(B X C) = kron(B, C^T) vec(X)` for C-ordered flattening. Only
    usable for small `d`; it exists so tests can invert `F` by a route that
    shares no code with `congruent_pencil` / `natural_gradient`.
    """
    return 0.5 * (np.kron(B, C.T) + np.kron(C, B.T))


# --- sec. 4: generalised congruent diagonalisation ---------------------------


def floor_spectrum(M: np.ndarray, eps: float) -> np.ndarray:
    """Raise every eigenvalue below `eps * lam_max` to that floor.

    The one regularisation primitive of the method. A floor is the identity on
    every eigenvalue above it, so well-determined directions keep their exact
    scale; a shift (`M + eps I`) perturbs those too, paying accuracy
    everywhere for protection only the small end needs. Measured on a
    reachable least-squares target at `eps = 1e-4`, floor against shift is
    `5.8e-9` vs `7.7e-7` final loss on a wide `W`.
    """
    w, U = eigh(M)
    w = np.clip(w, 0.0, None)
    return (U * np.maximum(w, eps * max(w.max(), np.finfo(float).tiny))) @ U.T


def congruent_pencil(
    B: np.ndarray, C: np.ndarray, eps: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the pencil `(B, C)` by congruence.

    Returns `(P, lam)` with `P^T B_f P = I` and `P^T C_f P = diag(lam)`, where
    both spectra are floored first.

    Goes through the symmetric root `B^{-1/2}` rather than a Cholesky factor:
    congruence only needs *some* factor `B = K K^T`, and the symmetric one
    exposes the eigenvalues so the floor can be applied to them directly.

    Flooring both sides is all the regularisation this needs -- it protects
    `B^{-1/2}` and keeps the pencil eigenvalues positive, so the denominators
    of sec. 5 come out positive with nothing further added.
    """
    Bf = floor_spectrum(B, eps)
    w, U = eigh(Bf)
    B_inv_half = (U / np.sqrt(np.clip(w, np.finfo(float).tiny, None))) @ U.T
    M = B_inv_half @ floor_spectrum(C, eps) @ B_inv_half
    M = 0.5 * (M + M.T)
    lam, Q = eigh(M)
    lam = np.clip(lam, 0.0, None)
    # The floor belongs on every spectrum that reaches a denominator, not only
    # on the ones being inverted. Flooring `B` and `C` does not keep `lam`
    # positive in finite precision: the pencil spreads it over more orders than
    # a low working dtype resolves, and the small end rounds to zero.
    lam = np.maximum(lam, eps * max(lam.max(), np.finfo(float).tiny))
    return B_inv_half @ Q, lam


def isotropic_pencil(C: np.ndarray, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The `S = I` special case: no congruence needed at all.

    When the output-side covariance is taken isotropic, the out-side operator
    is `F(X) = 2(X C + C X)` and diagonalising it is a plain symmetric
    eigenproblem, so `P` comes out orthogonal (`P^-1 = P^T`) instead of merely
    invertible. Much better conditioned than the general pencil.
    """
    lam, Q = eigh(C)
    lam = np.clip(lam, 0.0, None)
    return Q, np.maximum(lam, eps * max(lam.max(), np.finfo(float).tiny))


# --- sec. 5: the natural gradient --------------------------------------------


def natural_gradient(
    G_skew: np.ndarray, P: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    """Solve `F(X) = G_skew` in the congruent basis of sec. 4.

    With `P^T B P = I` and `P^T C P = diag(lam)`, substituting `X = P Y P^T`
    turns `(BXC + CXB)/2 = G` into `(lam_i + lam_j)/2 Y_ij = (P^T G P)_ij`.

    No damping here. `lam` comes from floored spectra, so the denominators are
    positive by construction, and regularisation lives in exactly one place --
    `floor_spectrum`, applied to the three matrices whose spectra can go
    singular: `A`, `W^T W`, `W A W^T`.
    """
    den = 0.5 * (lam[:, None] + lam[None, :])
    return skew(P @ ((P.T @ G_skew @ P) / den) @ P.T)


# --- sec. 6: trust-region step size ------------------------------------------


def trust_region_alpha(
    quad: float, curv: float, alpha_max: float = 1.0, eps: float = 1e-12
) -> float:
    """`alpha_t = min(alpha_max, quad / (curv + eps))`.

    This is identically 1 whenever the basis is fresh: `X = F^+ G` gives
    `F(X) = G`, hence `curv = <X, F(X)> = <X, G> = quad` as an algebraic
    identity, not an approximation. What makes it move is *staleness* -- `X`
    is built from a basis `T_fac` steps old while `curv` is measured against
    the current statistics, so `alpha_t` reads out how far the factorisation
    has drifted behind `W`. Measured on a reachable least-squares target it
    decays monotonically with steps-since-refactor and reaches ~0.47 at
    `T_fac = 50`.

    `alpha_max = 1` makes the mechanism one-sided: a stale basis can only
    shorten the step, never lengthen it. A larger cap lets staleness *amplify*
    (1.23 was observed at `T_fac = 50`), which is the opposite of a safeguard.
    """
    return float(min(alpha_max, quad / (curv + eps)))


# --- sec. 7: Cayley update ---------------------------------------------------


def cayley(X: np.ndarray, c: float) -> np.ndarray:
    """`(I + c/2 X)^{-1} (I - c/2 X)`, i.e. `Cayley(-c X)` for skew `X`.

    Exactly orthogonal for skew `X` and any real `c`; `I + c/2 X` is never
    singular because a real skew matrix has purely imaginary eigenvalues.
    """
    n = X.shape[0]
    half = 0.5 * c * X
    return np.linalg.solve(np.eye(n) + half, np.eye(n) - half)


# --- the full step -----------------------------------------------------------


@dataclass
class NGDPionReference:
    """One dense NGD-Pion step, statistics included.

    `isotropic_delta` is the `S = I` reading of sec. 2: the backward
    covariance is taken isotropic, which collapses the out-side pencil to a
    symmetric eigenproblem and removes the need to accumulate `S` at all --
    only forward activations are observed. The general-`S` path is kept so the
    two can be compared.
    """

    d_out: int
    d_in: int
    beta: float = 0.95
    eps: float = 1e-8  # spectral floor, relative to lambda_max
    eta: float = 1e-2
    alpha_max: float = 1.0
    isotropic_delta: bool = True

    def __post_init__(self) -> None:
        self.A = np.zeros((self.d_in, self.d_in))
        self.S = np.eye(self.d_out) if self.isotropic_delta else np.zeros((self.d_out,) * 2)
        self._basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def observe(self, x: np.ndarray, delta: np.ndarray | None = None) -> None:
        """Sec. 3. `delta` is ignored (and unnecessary) when `S = I`."""
        x = np.atleast_2d(x)
        self.A = self.beta * self.A + (1 - self.beta) * (x.T @ x) / x.shape[0]
        if not self.isotropic_delta:
            if delta is None:
                raise ValueError("general-S mode needs the backward signal")
            delta = np.atleast_2d(delta)
            self.S = self.beta * self.S + (1 - self.beta) * (delta.T @ delta) / delta.shape[0]

    def _pairs(self, W: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """The two `(B, C)` operator pairs of sec. 4, as `F(X) = (BXC + CXB)/2`.

        Neither `C` is damped here. The document damps `S` before forming
        `W^T S W`, but that cannot do what it looks like it does: the rank of
        `W^T(S + eps I)W` is still bounded by `rank(W)`, so the structural
        zeros survive any `eps`. And `C` is never inverted -- the anchor `B`
        is. Damping belongs in `congruent_pencil` (protecting `B^{-1/2}`) and
        in `natural_gradient` (handling the zeros), and nowhere else.
        """
        return (
            (self.A, W.T @ self.S @ W),
            (self.S, W @ self.A @ W.T),
        )

    def refactor(self, W: np.ndarray) -> None:
        """Sec. 4, both sides, rebuilt from the current `W`.

        Whenever a side's pencil has an identity anchor the congruence is
        skipped: `eigh` alone gives an orthogonal basis, no inverse square
        root, and therefore nothing for `eps_A` to protect. That covers the
        out-side always (under `S = I`) and the in-side for every `W` with
        `W^T W = I` -- square and tall matrices under a semi-orthogonal
        initialisation, which Pion then preserves for the whole run. Only a
        wide `W` (`d_out < d_in`) leaves `W^T W` a rank-deficient projector
        and needs the general path.
        """
        (A_in, S_in), (S_out, A_out) = self._pairs(W)
        if np.abs(S_in - np.eye(self.d_in)).max() < 1e-8:
            P_in, lam_in = isotropic_pencil(A_in, eps=self.eps)
        else:
            P_in, lam_in = congruent_pencil(A_in, S_in, eps=self.eps)
        if self.isotropic_delta:
            P_out, lam_out = isotropic_pencil(A_out, eps=self.eps)
        else:
            P_out, lam_out = congruent_pencil(S_out, A_out, eps=self.eps)
        self._basis = (P_in, lam_in, P_out, lam_out)

    def direction(
        self, W: np.ndarray, G: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Secs. 5-6: preconditioned directions plus the trust-region scalar."""
        if self._basis is None:
            self.refactor(W)
        P_in, lam_in, P_out, lam_out = self._basis
        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, P_in, lam_in)
        X_out = natural_gradient(G_out, P_out, lam_out)

        (A_in, S_in), (S_out, A_out) = self._pairs(W)
        quad = float(np.sum(G_in * X_in) + np.sum(G_out * X_out))
        curv = float(
            np.sum(X_in * fisher_apply(A_in, S_in, X_in))
            + np.sum(X_out * fisher_apply(S_out, A_out, X_out))
        )
        return X_in, X_out, quad, curv, trust_region_alpha(quad, curv, self.alpha_max)

    def step(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Sec. 7. Returns the updated `W`; the spectrum of `W` is preserved."""
        X_in, X_out, _quad, _curv, alpha = self.direction(W, G)
        c = self.eta * alpha
        return cayley(X_out, c) @ W @ cayley(X_in, c)
