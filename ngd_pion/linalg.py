"""Primitives shared by the rest of the package.

Nothing here knows about NGD-Pion. Every function is pure, batched over any
leading dimensions, and preserves the dtype and device it is given -- callers
choose precision, this module never converts.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

__all__ = ["skew", "floor_eigenvalues", "floor_spectrum", "cayley", "is_identity",
           "exact_fp32", "spectral_norm", "cayley_newton_schulz",
           "safe_eigh", "EIGH_FALLBACKS"]


@contextmanager
def exact_fp32():
    """Make fp32 mean fp32 for the duration, then put the settings back.

    On an Ampere-or-newer card torch performs fp32 matrix operations in TF32
    by default -- eight bits of exponent, **ten** of mantissa -- and that
    includes the solve inside `cayley`. Measured on an RTX PRO 6000 Blackwell,
    the orthogonality error of the retraction goes from `3e-06` to `4e-03`, and
    the consequence is not cosmetic: over 200 two-sided steps at an angle of
    1e-2 the singular values of `W` move by a **relative 1.0** with TF32 and by
    `2.6e-04` without. A full run is 73242 steps.

    Preserving the spectrum exactly is the whole reason this method uses Cayley
    rather than a truncated exponential, so the guarantee has to be defended
    where it is produced rather than assumed from the caller's environment. A
    CPU test suite cannot see any of this, because TF32 does not exist there.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(precision)


@contextmanager
def tf32_matmul():
    """fp32 matrix products in TF32 (ten mantissa bits) for the duration, then the settings back. Used only where the
    result is upstream of the Cayley gate (natural_pion.NaturalPionFast, tf32="upstream"), or to measure."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.set_float32_matmul_precision(precision)


def skew(M: torch.Tensor) -> torch.Tensor:
    """The skew-symmetric part, `(M - M^T) / 2`.

    Used both as a projection and as hygiene: the algebra guarantees several
    intermediate results are skew, and rounding does not, so the projection is
    applied where that guarantee is relied upon.
    """
    return 0.5 * (M - M.transpose(-1, -2))


# How many times each rung of `safe_eigh`'s ladder was needed. Zero is the
# expected reading; anything else says the pencil handed to `eigh` has become
# badly enough conditioned that the primary solver gives up, which is a fact
# about the algorithm's conditioning and not something to be quietly absorbed.
EIGH_FALLBACKS = {"jitter": 0, "backend": 0, "cpu": 0}


def safe_eigh(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`torch.linalg.eigh` with fallbacks, because its iteration can fail.

    cusolver's symmetric eigensolver is iterative and has an iteration cap. It
    gave up on one 512x512 pencil at step 41 500 of an otherwise healthy run --
    `error code: 59`, "the input matrix is ill-conditioned or has too many
    repeated eigenvalues" -- and took seventeen hours of work with it. The
    congruence path is where this happens: it forms
    `M = B^-1/2 C B^-1/2` from a `B` whose spectrum has been floored at
    `eps * lam_max`, so `B^-1/2` already carries a condition number of `100` at
    `eps = 1e-4`, and two matrix products then add rounding on top.

    Three rungs, in increasing order of desperation, and **none of them runs
    unless the one above it raised**. When `eigh` succeeds -- which is the
    normal case -- this function is exactly `torch.linalg.eigh` and the
    trajectory is bit-identical.

    1. **Jitter.** Add `1e-7` of the mean diagonal along the diagonal and retry
       on the GPU. That moves every eigenvalue by that much, which is far below
       the floor `eps` applies anyway, and it is often enough for the iteration
       to converge.
    2. **The other backend.** cusolver and magma use different algorithms --
       divide-and-conquer against Jacobi -- and one converges where the other
       does not.
    3. **CPU.** LAPACK is a third implementation again. Measured at 18 ms for
       512x512 and 142 ms for 1376x1376 on eight threads, against a
       refactorisation every 25 steps, so even falling back on *every*
       refactorisation for *every* wide layer costs 5.3% of wall clock. It is
       cheap; it is last only because needing it often is a symptom.

    `EIGH_FALLBACKS` counts the rungs so that "needing it often" is visible
    rather than inferred.
    """
    try:
        return torch.linalg.eigh(M)
    except Exception:
        pass

    scale = M.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    jittered = M + (1e-7 * scale)[..., None, None] * eye
    try:
        out = torch.linalg.eigh(jittered)
        EIGH_FALLBACKS["jitter"] += 1
        return out
    except Exception:
        pass

    if M.is_cuda:
        try:
            previous = torch.backends.cuda.preferred_linalg_library()
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
                out = torch.linalg.eigh(jittered)
                EIGH_FALLBACKS["backend"] += 1
                return out
            finally:
                torch.backends.cuda.preferred_linalg_library(previous)
        except Exception:
            pass

    w, Q = torch.linalg.eigh(jittered.cpu())
    EIGH_FALLBACKS["cpu"] += 1
    return w.to(M.device), Q.to(M.device)


def floor_eigenvalues(w: torch.Tensor, eps: float) -> torch.Tensor:
    """Raise eigenvalues below `eps * lam_max` to that floor (docs/history/ALGORITHM.md §4).

    The single regularisation primitive of the method. A floor is the identity
    on every eigenvalue above it, so well-determined directions keep their
    exact scale; a shift (`w + eps`) perturbs those too and pays accuracy
    everywhere for protection only the small end needs.

    `w` is clamped non-negative first: the spectra fed here are PSD by
    construction and any negative value is rounding.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    w = w.clamp_min(0.0)
    lam_max = w.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    return torch.maximum(w, eps * lam_max)


def floor_spectrum(M: torch.Tensor, eps: float) -> torch.Tensor:
    """`floor_eigenvalues` applied to a symmetric matrix, rebuilt in place."""
    w, U = torch.linalg.eigh(M)
    return (U * floor_eigenvalues(w, eps).unsqueeze(-2)) @ U.transpose(-1, -2)


def spectral_norm(
    X: torch.Tensor, iters: int, v: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest singular value of `X` by power iteration on `X^T X`.

    Returns `(sigma, v)`; feeding `v` back on the next call is the whole point.
    Cost is `iters` matrix-vector products, `O(n^2)`, against `O(n^3)` for a
    decomposition -- measured at 4830x cheaper than `matrix_norm(X, 2)` across
    the 56 weights of LLaMA-60M, where the exact call was 95% of the entire
    optimizer step because cusolver falls back to a slow path on 1376x1376.

    **Cold, this converges badly, and that is a property of the input rather
    than of the implementation.** A skew matrix has its singular values in
    equal pairs and the large ones bunch together, so the ratio governing
    convergence sits near 1. Measured on a random 512x512 skew: 21% relative
    error after one iteration, 6% after five, 0.5% after twenty. Warm, from a
    vector converged on the previous step's `X`, one iteration gives `1e-4`.
    So callers must cache `v` and spend iterations generously exactly twice --
    on the first step, and again whenever `X` moves discontinuously.

    The estimate is a Rayleigh quotient and therefore always a *lower* bound.
    For a diagnostic asking whether an angle stays bounded that is the less
    comfortable direction to err in, which is the reason the warm-start
    discipline above is a requirement and not an optimisation.
    """
    if iters < 0:
        raise ValueError(f"iters must be non-negative, got {iters}")
    tiny = torch.finfo(X.dtype).tiny

    def unit(z: torch.Tensor) -> torch.Tensor:
        """Normalise, reseeding any vector that has collapsed to zero.

        Without the reseed this function has an absorbing state: a cached `v`
        that reaches zero once stays zero for the rest of the run, because
        `0 / tiny` is `0`, and `sigma` is then reported as exactly `0` forever
        for that layer.

        Measured on a 1376x1376 skew in fp32, sweeping `||X||_2`: correct down
        to `1e-14`; at `1e-16` the norm of `v` underflows to a denormal, the
        `clamp_min(tiny)` below then inflates `v` to `7e11` and `sigma` comes
        out **eleven orders too large**; at `1e-20` it is zero and stays zero.
        So the failure is silent corruption before it is silence.

        What this accounts for in job 246666 is the *shape* of the
        `angle_max = 0` readings -- irreversible, and spreading layer by layer
        between step 450 and step 700, with no layer ever recovering, which is
        what an absorbing state looks like and what a genuinely small rotation
        does not. It does not account for the trigger: the step's `X` runs
        around `1e-3`, far above the threshold, so something had to push a
        layer transiently below it. With the reseed the estimate recovers
        instead of latching, so the next run distinguishes the two.
        """
        n = z.norm(dim=-1, keepdim=True)
        alive = n > tiny
        # Branchless on purpose. The obvious `if dead.any(): ...` costs a
        # device-to-host synchronisation on every call, and this runs five
        # times per `spectral_norm`, twice per layer, 56 layers -- 560 syncs a
        # step against the roughly 170 the step already had. Measured: 0.72
        # s/step became upwards of 26, and jobs 252299 and 252302 were
        # cancelled for it. Removing it bought a factor of six, to 4 s/step,
        # which is still five times the baseline -- the remainder is elsewhere,
        # and is not in this function: step 0 makes 101 `unit()` calls per
        # `spectral_norm` against 5 on an ordinary step and is faster than the
        # old baseline. The reseed vector is all-ones, whose norm is sqrt(n),
        # so the replacement needs no second reduction either.
        root_n = float(z.shape[-1]) ** 0.5
        z = torch.where(alive, z, torch.ones_like(z))
        n = torch.where(alive, n, torch.full_like(n, root_n))
        return z / n.clamp_min(tiny)

    if v is None:
        v = torch.ones(*X.shape[:-2], X.shape[-1], dtype=X.dtype, device=X.device)
    v = unit(v)
    Xt = X.transpose(-1, -2)
    # Normalising between the two products rather than after the pair. The
    # underflow above is not a property of a degenerate direction: `X^T X v`
    # scales as `||X||^2`, and the step's `X` runs at `1e-2` and below, so in
    # fp32 the intermediate reaches `1e-40` and rounds to zero while `X`
    # itself is perfectly well scaled. Splitting the product keeps both
    # intermediates at unit norm and removes the failure at its source; the
    # iterate's direction, and therefore the estimate, is unchanged.
    for _ in range(iters):
        u = unit((X @ v.unsqueeze(-1)).squeeze(-1))
        v = unit((Xt @ u.unsqueeze(-1)).squeeze(-1))
    return (X @ v.unsqueeze(-1)).squeeze(-1).norm(dim=-1), v


def cayley(X: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """`(I + c/2 X)^-1 (I - c/2 X)`, i.e. `Cayley(-c X)` (docs/history/ALGORITHM.md §7).

    Exactly orthogonal for skew `X` and any real `c`, and never singular: a
    real skew matrix has purely imaginary eigenvalues, so `I + c/2 X` cannot
    lose rank. Unlike a truncated exponential it also cannot inflate -- its
    spectral norm is exactly 1 at every step size, which is what keeps the
    singular values of `W` fixed rather than merely nearly fixed.

    That last sentence is only true if fp32 means fp32, so this guards itself
    rather than trusting the caller: with TF32 left on, the solve below carries
    ten bits of mantissa and the spectrum moves by a relative 1.0 over 200
    steps. See `exact_fp32`.
    """
    with exact_fp32():
        return _cayley(X, c)


def _cayley(X: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    n = X.shape[-1]
    eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
    half = 0.5 * (c if isinstance(c, torch.Tensor) else torch.as_tensor(c, dtype=X.dtype, device=X.device))
    half = half.reshape(*([1] * (X.dim() - 2)), 1, 1) if isinstance(half, torch.Tensor) and half.dim() else half
    A = half * X
    return torch.linalg.solve(eye + A, eye - A)


def cayley_newton_schulz(
    X: torch.Tensor, c: float | torch.Tensor, iters: int = 2
) -> torch.Tensor:
    """`cayley`, with the inverse by Newton-Schulz instead of a solve.

    `Cayley(-cX) = (I + A)^-1 (I - A) = 2(I + A)^-1 - I` for `A = (c/2)X`, and
    that second form is used here because it costs one matmul less.

    The iteration is `Z <- Z(2I - MZ)`, whose residual `R = I - MZ` **squares**
    every step. Starting from `Z0 = I - A` gives `M Z0 = I - A^2`, so the
    residual begins at `A^2` and runs `A^4`, `A^8`, `A^16`. Rotation angles in
    this method sit near `1e-2`, where two iterations land below the fp32
    epsilon.

    Why bother, when `torch.linalg.solve` computes the same thing exactly: the
    solve is latency-bound, not flop-bound. A 512x512 solve with 512
    right-hand sides is 0.36 GFLOP and takes 1 ms on an RTX PRO 6000
    Blackwell -- 0.36 TFLOPS, where matmuls on the same card run at 18. LU with
    pivoting is sequential and does not fill a GPU at this size, and the step
    does 112 of them. Newton-Schulz costs `4k n^3` of matmul against the
    solve's `2.7 n^3`: three times the arithmetic on hardware fifty times
    faster at it.

    **This converges only for `||A|| < 1`**, and unlike the solve it degrades
    rather than failing loudly -- at `||A|| = 0.5` a single iteration leaves an
    orthogonality error of `3.8e-2`. Callers must bound `(|c|/2) * ||X||_2`,
    which `spectral_norm` already produces for the angle diagnostic, and either
    raise `iters` or fall back to `cayley`. Measured orthogonality error in
    fp32, against `1.1e-6` for the exact solve at the same angle:

    | angle | 1 iter | 2 iters | 3 iters |
    |---|---|---|---|
    | `1e-2` | `1.1e-5` | `6.1e-6` | `6.1e-6` |
    | `1e-1` | `8.3e-6` | `3.7e-6` | `3.7e-6` |
    | `5e-1` | `2.7e-3` | `8.6e-6` | `3.1e-6` |
    | `1.0`  | `3.8e-2` | `1.2e-3` | `4.6e-6` |
    """
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    with exact_fp32():
        n = X.shape[-1]
        eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
        half = c if isinstance(c, torch.Tensor) else torch.as_tensor(
            c, dtype=X.dtype, device=X.device)
        half = 0.5 * half
        if half.dim():
            half = half.reshape(*([1] * (X.dim() - 2)), 1, 1)
        A = half * X
        M = eye + A
        Z = eye - A                       # residual A^2, and it squares
        for _ in range(iters):
            Z = Z @ (2.0 * eye - M @ Z)
        return 2.0 * Z - eye


def is_identity(M: torch.Tensor, atol: float = 1e-6) -> bool:
    """Whether `M` is the identity to `atol`, used to pick the cheap basis path."""
    n = M.shape[-1]
    if M.shape[-2] != n:
        return False
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    return bool(torch.all((M - eye).abs().amax(dim=(-2, -1)) <= atol))
