"""Natural Pion: spectrum-preserving rotations along the natural gradient (docs/paper/latex/main.pdf).

* `NaturalPionFast` -- the method (the paper's Algorithm 1) with its defaults: length="rms" (Pion's rule: c = theta sqrt(mn)
  / ||W X_in + X_out W||_F), tf32="upstream", fisher_dtype="bf16", eps 1e-3. With switches, every ablation. (Until
  2026-11-11 the defaults were spectral / off / fp32 / 1e-4.) The paper's exact form is this class with t_fac=1,
  stats_window=None and retraction="cayley" (scripts/paper/v2.sbatch rows c-alg1-*).
* `NaturalPion` -- the earlier exact reference with the angle rule below, kept for the tests and as the base class.

For each rotated weight `W` (m x n) with gradient `G`, one step of the reference is

    G_in  = skew(W^T G),  G_out = skew(G W^T)                 generators (Riemannian gradients)
    M     = beta1 M + (1 - beta1) G                           Lie momentum on each side
    X     = F^-1(M)                                           F(X) = (B X C + C X B) / 2 on each side:
                                                              in (A, W^T D W), out (D, W A W^T)
    sigma = max(||X_in||_2, ||X_out||_2)                      one scale for both sides
    c     = theta_t / sigma                                   the largest plane turns by theta_t (angle rule)
    W    <- Cay(X_out, c) W Cay(X_in, c)                      Cayley: W's singular values are kept

`lr` is `theta_t` (an angle in radians under the angle rule, an RMS scale under length="rms"); the caller schedules it.
`A = E[x x^T]` (layer input) and `D = E[d d^T]` (gradient at the layer output) are fed by hooks:
`ngd_pion.hooks.attach` for `observe` and `ngd_pion.hooks.attach_backward` for `observe_backward`.
Built from the primitives in `covariance`, `direction`, `factorization` and `linalg`.
"""

from __future__ import annotations

from collections import defaultdict
import math

import torch

from .covariance import CovarianceAccumulator
from .direction import generators
from .factorization import (Basis, basis_congruence, basis_congruence_from_root, basis_identity_anchor,
                            floored_inverse_root)
from .linalg import floor_eigenvalues, safe_eigh
from .linalg import exact_fp32, is_identity, skew, spectral_norm, tf32_matmul
from contextlib import nullcontext

__all__ = ["NaturalPion", "NaturalPionFast", "BY_NAME", "default_of"]


def ns_iterations(angle: float, minimum: int) -> int | None:
    """Newton-Schulz iterations for (I + Psi)^-1, ||Psi||_2 = angle / 2, that make the Cayley step exact to fp32
    round-off: after k iterations from Z0 = I - Psi the error is about (angle/2)^(2^(k+1)), so the smallest k >= minimum
    with that below 2^-24 (2 up to angle 0.2, 3 up to 0.4, 4 up to 0.8). None where the iteration converges too
    slowly to be worth it (angle / 2 >= 0.9): the exact solve then."""
    x = float(angle) / 2.0
    if x >= 0.9:
        return None
    k = minimum
    while x ** (2 ** (k + 1)) > 2.0 ** -24:
        k += 1
    return k


def squarings(angle: float, base: float = 0.2) -> int:
    """How many times to halve a step's largest rotation angle so it is at most `base` (where two Newton-Schulz
    iterations are exact to fp32), for scaling and squaring."""
    a = float(angle)
    return max(0, math.ceil(math.log2(a / base))) if a > base else 0


def _lmax(C: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Largest eigenvalue of symmetric PSD `C` (batched) by power iteration, shape (..., 1)."""
    v = torch.ones(*C.shape[:-1], 1, dtype=C.dtype, device=C.device)
    for _ in range(iters):
        v = C @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    return (v.transpose(-1, -2) @ C @ v).squeeze(-1)


class NaturalPion(torch.optim.Optimizer):
    """Natural Pion, exact reference with the angle rule: fresh Fisher basis and statistics every step.

    Args:
        params: 2-D weights to rotate. Embeddings, head, norms and biases belong to another optimizer.
        lr: rotation angle `theta` in radians for the largest plane of every matrix. Scheduled by the caller.
        beta1: Lie momentum on the generators (as in Pion). 0 turns it off.
        beta_a: EMA factor of `A = E[x x^T]`.
        beta_d: EMA factor of `D = E[d d^T]`.
        t_fac: rebuild the Fisher basis every `t_fac` steps.
        stats_window: collect `A` and `D` only in the last `stats_window` steps before a rebuild;
            `None` collects every step. Between rebuilds nothing reads them.
        eps: spectral floor, relative to the largest eigenvalue, on every spectrum that is inverted.
        angle_iters, angle_warmup: power-iteration steps for `||X||_2`, warm-started from the previous
            step's vector; `angle_warmup` right after a rebuild, where `X` jumps.
        damping: Tikhonov-style damping relative to each matrix's largest Fisher eigenvalue: the denominator
            (lam_i + lam_j)/2 becomes (lam_i + lam_j)/2 + damping * max. In the basis this is F + delta B X B,
            damping in the metric of the anchor factor. 0 (default) leaves only the floor `eps`.
        retraction: "cayley" solves (I + c/2 X) R = (I - c/2 X) exactly; "ns" computes the inverse by
            Newton-Schulz iterations (matrix products only), at least `ns_iters` and as many as the step's largest
            angle needs to be exact to fp32 round-off (`ns_iterations`); the exact solve above `ns_guard`. It rounds
            next to 1 with a positive bias in fp32 (kept to reproduce that measurement); "ns_delta" is the same
            iteration carried in Z - I, which removes the bias; "trunc" is
            Pion's retraction, the degree-2 truncated exponential I + A + A^2/2 with A = -c X, which is not
            orthogonal: (T^T T = I + A^4/4), so it lets the spectrum grow.
        row_blocks: `{id(weight): k}` splits that weight's rows into `k` equal blocks, each rotated as a
            separate matrix with its own Fisher (input moment `A` shared, output moment the block's
            diagonal block of `D`). Pion rotates the query projection per attention head this way.
        share_inputs: weights fed the very same input tensor (q, k, v; gate, up) share one accumulator
            for `A` instead of computing the identical gram once each. Bit-identical to not sharing.
        stats_dtype: "fp32" forms the grams of `A` and `D` from fp32 inputs; "bf16" rounds the inputs
            (activations, output gradients) to bf16 and still accumulates and returns the gram in fp32, so
            it stays a PSD gram (see `CovarianceAccumulator.observe`).
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        *,
        beta1: float = 0.9,
        beta_a: float = 0.95,
        beta_d: float = 0.5,
        t_fac: int = 1,
        stats_window: int | None = None,
        eps: float = 1e-4,
        angle_iters: int = 2,
        angle_warmup: int = 50,
        row_blocks: dict[int, int] | None = None,
        damping: float = 0.0,
        retraction: str = "cayley",
        ns_iters: int = 2,
        ns_guard: float = 1.8,
        share_inputs: bool = False,
        stats_dtype: str = "fp32",
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must lie in [0, 1), got {beta1}")
        for name, b in (("beta_a", beta_a), ("beta_d", beta_d)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {b}")
        if t_fac < 1:
            raise ValueError(f"t_fac must be at least 1, got {t_fac}")
        if retraction not in ("cayley", "ns", "ns_delta", "trunc"):
            raise ValueError(f"retraction must be cayley, ns, ns_delta or trunc, got {retraction!r}")
        if stats_dtype not in ("fp32", "bf16"):
            raise ValueError(f"stats_dtype must be fp32 or bf16, got {stats_dtype!r}")
        if stats_window is not None and not 1 <= stats_window <= t_fac:
            raise ValueError(f"stats_window must lie in [1, t_fac], got {stats_window}")
        params = list(params)
        for p in params:
            if p.dim() != 2:
                raise ValueError(f"NaturalPion takes 2-D weights, got shape {tuple(p.shape)}")
        self.row_blocks = dict(row_blocks or {})
        for p in params:
            k = self.row_blocks.get(id(p), 1)
            if k < 1 or p.shape[0] % k:
                raise ValueError(f"row_blocks {k} does not divide {p.shape[0]} rows")
        self.share_inputs = share_inputs
        self._input_dtype = torch.bfloat16 if stats_dtype == "bf16" else None
        self._last_input = None                    # (input tensor, its accumulator) for share_inputs
        super().__init__(params, dict(
            lr=lr, beta1=beta1, beta_a=beta_a, beta_d=beta_d, t_fac=t_fac, stats_window=stats_window,
            eps=eps, angle_iters=angle_iters, angle_warmup=angle_warmup, compute_dtype=torch.float32,
            damping=damping, retraction=retraction, ns_iters=ns_iters, ns_guard=ns_guard,
        ))

    # --- statistics, fed by hooks -----------------------------------------------------------------

    def _group_of(self, param: torch.Tensor) -> dict:
        for group in self.param_groups:
            for p in group["params"]:
                if p is param:
                    return group
        raise KeyError("parameter is not held by this optimizer")

    def _collecting(self, param: torch.Tensor, kind: str = "a") -> bool:
        """Whether this step's statistic `kind` ("a": input moment, "d": output-gradient moment) is ever read:
        always before the first basis, then only inside the window that precedes a rebuild."""
        group = self._group_of(param)
        window = group["stats_window"]
        state = self.state[param]
        return window is None or "bases" not in state or state["since_refactor"] >= group["t_fac"] - window

    def _stat(self, param: torch.Tensor, key: str, beta: float) -> CovarianceAccumulator:
        state = self.state[param]
        if key not in state:
            state[key] = CovarianceAccumulator(beta=beta, dtype=torch.float32, input_dtype=self._input_dtype)
            state.setdefault("step", 0)
            state.setdefault("since_refactor", 0)
        return state[key]

    def observe(self, param: torch.Tensor, x: torch.Tensor) -> None:
        """Activations entering the layer that owns `param` (forward pre-hook)."""
        if not self._collecting(param, "a"):
            return
        beta = self._group_of(param)["beta_a"]
        last = self._last_input
        if self.share_inputs and last is not None and last[0] is x and last[1].beta == beta:
            self.state[param]["cov"] = last[1]     # same tensor, already folded in: same gram, computed once
            self.state[param].setdefault("step", 0)
            self.state[param].setdefault("since_refactor", 0)
            return
        acc = self._stat(param, "cov", beta)
        acc.observe(x)
        if self.share_inputs:
            self._last_input = (x, acc)

    def observe_backward(self, param: torch.Tensor, delta: torch.Tensor) -> None:
        """Gradient at the layer's output (backward hook). Scaled by the token count squared: the loss is a
        mean, so autograd's `delta` carries `1/N`, and the Fisher of a mean loss is the mean of per-token
        outer products."""
        if not self._collecting(param, "d"):
            return
        flat = delta.detach().reshape(-1, delta.shape[-1])
        self._stat(param, "cov_backward", self._group_of(param)["beta_d"]).observe(
            flat, scale=float(flat.shape[0]) ** 2)

    def _ready(self, param: torch.Tensor) -> bool:
        state = self.state[param]
        return "cov" in state and state["cov"].ready and "cov_backward" in state and state["cov_backward"].ready

    def _units(self, p: torch.Tensor) -> list[slice]:
        """The row blocks of `p` that rotate as separate matrices: one block unless `row_blocks` splits it."""
        k = self.row_blocks.get(id(p), 1)
        h = p.shape[0] // k
        return [slice(i * h, (i + 1) * h) for i in range(k)]

    @staticmethod
    def _unit_state(state: dict, i: int) -> dict:
        """Per-block state. Block 0 lives in the parameter's own state, so an unsplit weight keeps the layout
        it always had; further blocks get their own dicts."""
        if i == 0:
            return state
        return state.setdefault("blocks", {}).setdefault(i, {})

    # --- the Fisher basis ---------------------------------------------------------------------------

    def _floor(self, group: dict) -> float:
        """The relative spectral floor for a rebuild now: `eps` on the reference path."""
        return group["eps"]

    def _refactor(self, params: list[torch.Tensor], group: dict) -> None:
        """Bases with `P^T B P = I`, `P^T C P = diag(lam)` for in (A, W^T D W) and out (D, W A W^T),
        batched over matrices of equal shape."""
        dt, eps = group["compute_dtype"], self._floor(group)
        by_shape: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for p in params:
            units = self._units(p)
            for i, rows in enumerate(units):
                shape = (rows.stop - rows.start, p.shape[1])
                by_shape[shape].append((p, i, rows, len(units) == 1))
        for members in by_shape.values():
            W = torch.stack([p.detach().to(dt) if whole else p.detach().to(dt)[rows] for p, _, rows, whole in members])
            A = torch.stack([self.state[p]["cov"].matrix.to(device=p.device, dtype=dt) for p, *_ in members])
            D = torch.stack([
                self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt) if whole
                else self.state[p]["cov_backward"].matrix.to(device=p.device, dtype=dt)[rows, rows]
                for p, _, rows, whole in members])
            Wt = W.transpose(-1, -2)
            gram_out = W @ A @ Wt
            gram_in = Wt @ D @ W
            # D is never the identity for a trained layer, so no cheap-path check (each was a host sync)
            basis_out = self._basis_out(D, gram_out, eps, members)
            basis_in = self._basis_in(A, gram_in, eps, members)
            for j, (p, i, _, _) in enumerate(members):
                self._unit_state(self.state[p], i)["bases"] = (
                    type(basis_in)(basis_in.P[j], basis_in.lam[j], basis_in.orthogonal, None, 1.0),
                    type(basis_out)(basis_out.P[j], basis_out.lam[j], basis_out.orthogonal, None, 1.0),
                )
                self.state[p]["since_refactor"] = 0

    def _basis_in(self, A: torch.Tensor, gram_in: torch.Tensor, eps: float, members: list):
        """In-side basis of the pencil (A, W^T D W) for a batch of matrices of one shape."""
        return basis_congruence(A, gram_in, eps)

    def _basis_out(self, D: torch.Tensor, gram_out: torch.Tensor, eps: float, members: list):
        """Out-side basis of the pencil (D, W A W^T)."""
        return basis_congruence(D, gram_out, eps)

    # --- optional per-part timing (off by default; one attribute check when off) ---------------------

    profile: dict | None = None          # set to {} to accumulate GPU-synchronised seconds per part

    def _tick(self, part: str | None) -> None:
        """Close the running part and open `part`. Synchronises the GPU only when profiling is on."""
        if self.profile is None:
            return
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        last = getattr(self, "_tick_open", None)
        if last is not None:
            self.profile[last[0]] = self.profile.get(last[0], 0.0) + now - last[1]
        self._tick_open = (part, now) if part else None

    # --- the step -------------------------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # fp32 must mean fp32: with TF32 the Cayley solve stops being orthogonal and W's spectrum drifts.
        with self._step_precision():
            for group in self.param_groups:
                active = [p for p in group["params"] if p.grad is not None]
                for p in active:
                    if not self._ready(p):
                        raise RuntimeError("no statistics for a parameter: attach the forward and backward hooks")
                stale = [p for p in active if "bases" not in self.state[p]]
                if stale:
                    self._await_refresh()
                    self._tick("refresh"); self._refactor(stale, group); self._tick(None)
                self._await_refresh()
                self._rotate(active, group)
                self._tick(None)
                due = [p for p in active if self.state[p]["since_refactor"] >= group["t_fac"]]
                self._last_input = None
                if due:
                    self._refresh(due, group)
        return loss

    def _refresh(self, params: list[torch.Tensor], group: dict) -> None:
        """Rebuild the bases of `params` now (the reference path)."""
        self._tick("refresh"); self._refactor(params, group); self._tick(None)

    def _await_refresh(self) -> None:
        """Wait for a refresh running elsewhere; nothing to wait for on the reference path."""

    @staticmethod
    def _ema(state: dict, key: str, g: torch.Tensor, beta: float) -> torch.Tensor:
        if key not in state:
            state[key] = torch.zeros_like(g)
        state[key].mul_(beta).add_(g, alpha=1.0 - beta)
        return state[key]

    def _sigma(self, state: dict, group: dict, X_in: torch.Tensor, X_out: torch.Tensor) -> float:
        """`max(||X_in||_2, ||X_out||_2)` by warm power iteration."""
        fresh = "angle_v_in" not in state or state["since_refactor"] == 0
        iters = group["angle_warmup"] if fresh else group["angle_iters"]
        s_in, state["angle_v_in"] = spectral_norm(X_in, iters, state.get("angle_v_in"))
        s_out, state["angle_v_out"] = spectral_norm(X_out, iters, state.get("angle_v_out"))
        return float(torch.maximum(s_in, s_out))

    def _solve(self, P, Pt, G, den):
        """`X = F^-1 G` in the basis: elementwise division by (lam_i + lam_j)/2, made exactly skew."""
        return skew(P @ ((Pt @ G @ P) / den) @ Pt)

    def _correct(self, X, G, P, den, items, rows, group):
        """Hook after `X = F^-1 G` in the basis; the reference path uses the basis solution as it is."""
        return X

    def _step_precision(self):
        """The arithmetic of the whole step: true fp32 (TF32 off), so the retraction stays orthogonal."""
        return exact_fp32()

    def _retraction_precision(self):
        """The arithmetic of the Cayley step and of the rotation of W inside the step's own; the reference path
        changes nothing."""
        return nullcontext()

    def _largest_angle(self, group: dict) -> float:
        """The largest rotation angle of this step, which decides whether Newton-Schulz may stand in for the exact
        Cayley solve: theta itself, since c = theta / sigma turns every matrix by theta."""
        return group["lr"]

    def _scales(self, rows: list, group: dict) -> list:
        """`c = theta / sigma` per rotated unit, one power iteration per matrix (the reference path)."""
        cs = []
        for p, W, X_in, X_out, b, _, _ in rows:
            state = self._unit_state(self.state[p], b)
            state["since_refactor"] = self.state[p]["since_refactor"]
            cs.append(group["lr"] / max(self._sigma(state, group, X_in, X_out), 1e-30))
        return cs

    def _rotate(self, active: list[torch.Tensor], group: dict) -> None:
        self._tick("generators+momentum")
        dt, beta1 = group["compute_dtype"], group["beta1"]
        rows = []                                  # [p, W, X_in, X_out, block index, block rows, whole]
        by_size: dict[int, list] = {}              # side size -> [(row, slot, generator, basis)]
        for p in active:
            units = self._units(p)
            Wp, Gp = p.detach().to(dt), p.grad.detach().to(dt)
            for i, r in enumerate(units):
                whole = len(units) == 1
                state = self._unit_state(self.state[p], i)
                W, G = (Wp, Gp) if whole else (Wp[r], Gp[r])
                G_in, G_out = generators(W, G)
                if beta1 > 0.0:
                    G_in = self._ema(state, "m_in", G_in, beta1)
                    G_out = self._ema(state, "m_out", G_out, beta1)
                b_in, b_out = state["bases"]
                rows.append([p, W, None, None, i, r, whole])
                by_size.setdefault(G_in.shape[0], []).append((len(rows) - 1, 2, G_in, b_in))
                by_size.setdefault(G_out.shape[0], []).append((len(rows) - 1, 3, G_out, b_out))

        # X = F^-1 G: in the basis F is elementwise division by (lam_i + lam_j)/2
        self._tick("F^-1")
        for items in by_size.values():
            P = torch.stack([b.P for *_, b in items])
            den = torch.stack([b.denominator for *_, b in items])     # (lam_i + lam_j) / 2
            if group["damping"]:
                den = den + group["damping"] * den.amax(dim=(-2, -1), keepdim=True)
            G = torch.stack([g for _, _, g, _ in items])
            Pt = P.transpose(-1, -2)
            X = self._solve(P, Pt, G, den)
            X = self._correct(X, G, P, den, items, rows, group)
            for k, (i, slot, _, _) in enumerate(items):
                rows[i][slot] = X[k]

        # c = theta / sigma, shared by both sides of a matrix
        self._tick("sigma")
        cs = self._scales(rows, group)
        solves: dict[int, list] = {}
        for i, (p, W, X_in, X_out, b, _, _) in enumerate(rows):
            solves.setdefault(X_in.shape[0], []).append((i, 2, cs[i], X_in))
            solves.setdefault(X_out.shape[0], []).append((i, 3, cs[i], X_out))

        # Cayley: R = (I + c/2 X)^-1 (I - c/2 X), exactly orthogonal for skew X
        self._tick("cayley")
        angle = self._largest_angle(group)
        iters = ns_iterations(angle, group["ns_iters"]) if angle <= group["ns_guard"] else None
        with self._retraction_precision():
            for n, items in solves.items():
                X = torch.stack([x for *_, x in items])
                if isinstance(items[0][2], torch.Tensor):
                    half = (0.5 * torch.stack([c for _, _, c, _ in items])).to(X.dtype).reshape(-1, 1, 1)
                else:
                    half = torch.as_tensor([0.5 * c for _, _, c, _ in items], dtype=X.dtype, device=X.device).reshape(-1, 1, 1)
                eye = torch.eye(n, dtype=X.dtype, device=X.device).expand_as(X)
                if group["retraction"] == "trunc":
                    A = -(half + half) * X                       # -c X: Cay(cX) and exp(-cX) agree to second order
                    R = eye + A + 0.5 * (A @ A)
                elif group["retraction"] == "ns_delta":
                    # The same Newton-Schulz iteration carried in the correction z = Z - I: residual r = M Z - I =
                    # A + z + A z, and Z (I - r) gives z <- z - r - z r. Nothing is rounded next to 1, which in "ns"
                    # biases R^T R - I positive by ~1e-6 per side per step in fp32 (the trained spectrum grew 1.1%
                    # over 3000 steps). R = I + 2 z.
                    # Large angles by scaling and squaring: the step's largest angle is divided by 2^k down to 0.2, where
                    # `ns_iters` iterations are exact to fp32, and the rotation is squared k times in the same correction
                    # form, D <- 2 D + D^2 for R = I + D. Orthogonal at any angle, never slower than more iterations
                    # (scripts/probes/cayley_speed.py); the angle tends to the exponential's rather than Cayley's.
                    k = squarings(angle)
                    A = (half / 2 ** k) * X
                    z = -A
                    for _ in range(group["ns_iters"]):
                        r = A + z + A @ z
                        z = z - r - z @ r
                    D = 2.0 * z
                    for _ in range(k):
                        D = 2.0 * D + D @ D
                    R = eye + D
                elif group["retraction"] == "ns" and iters is not None:
                    # Cay = 2 (I + A)^-1 - I; Newton-Schulz Z <- Z (2I - M Z) from Z0 = I - A, residual A^2 squaring
                    A = half * X
                    M, Z = eye + A, eye - A
                    for _ in range(iters):
                        Z = Z @ (2.0 * eye - M @ Z)
                    R = 2.0 * Z - eye
                else:
                    R = torch.linalg.solve(eye + half * X, eye - half * X)
                for k, (i, slot, _, _) in enumerate(items):
                    rows[i][slot] = R[k]

        self._tick("apply W")
        new: dict[int, list] = {}
        # the rotation of W belongs to the retraction: in TF32 its products alone move the spectrum (3.7e-2 median after
        # 3000 steps, measured 2026-11-06)
        with self._retraction_precision():
            for p, W, R_in, R_out, b, r, whole in rows:
                if whole:
                    p.copy_((R_out @ W @ R_in).to(p.dtype))
                else:
                    new.setdefault(id(p), [p, []])[1].append(R_out @ W @ R_in)
            for p, blocks in new.values():
                p.copy_(torch.cat(blocks, dim=0).to(p.dtype))
        for p in active:
            state = self.state[p]
            state["step"] += 1
            state["since_refactor"] += 1

    def load_state_dict(self, state_dict) -> None:
        """torch moves plain tensors in the state; the accumulators and bases have to be moved by hand."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for key in ("cov", "cov_backward"):
                    if key in state:
                        state[key].to(p.device)
                for unit in [state, *state.get("blocks", {}).values()]:
                    if "bases" in unit:
                        unit["bases"] = tuple(
                            type(b)(b.P.to(p.device), b.lam.to(p.device), b.orthogonal, None, 1.0) for b in unit["bases"])
                    for key in ("m_in", "m_out", "angle_v_in", "angle_v_out"):
                        if isinstance(unit.get(key), torch.Tensor):
                            unit[key] = unit[key].to(p.device)


class NaturalPionFast(NaturalPion):
    """Natural Pion, **conservative** tier: every speed-up standard and explainable, loss within 0.010 of
    `NaturalPion` at 3000 and 15000 steps, and only in Natural Pion's own parts (never those shared with Pion).
    Basis every 25 steps, statistics in the 6 steps before each rebuild, EMA 0.5 on both factors, one `A` per
    distinct input tensor, the Cayley inverse by Newton-Schulz in the correction form (`ns_delta`; the exact
    solve above `ns_guard`). Everything else as `NaturalPion`.

    `lr` is the peak step scale of Pion's RMS rule (the RMS of `W X_in + X_out W` per step), not an angle: the paper
    uses 3e-4 to 6e-4 (Section 4.1). The 0.05 of `NaturalPion` is an angle in radians and is far too large here."""

    def __init__(self, params, lr: float = 3e-4, *, t_fac: int = 25, stats_window: int | None = 6,
                 beta_a: float = 0.5, beta_d: float = 0.5, share_inputs: bool = True,
                 retraction: str = "ns_delta", stats_dtype: str = "bf16", batch_sigma: bool = True,
                 share_root: bool = True, direction: str = "natural", length: str = "rms", sides: str = "joint", planes: str = "scaled",
                 attn_mult: float = 1.0, angle_mult: dict[int, float] | None = None, a_window: int | None = None, d_window: int | None = None, a_every: int = 1,
                 tf32: str = "upstream", fisher_dtype: str = "bf16", floor_schedule: str = "constant", eps: float = 1e-3,
                 **kwargs) -> None:
        super().__init__(params, lr, eps=eps, t_fac=t_fac, stats_window=stats_window, beta_a=beta_a, beta_d=beta_d,
                         share_inputs=share_inputs, retraction=retraction, stats_dtype=stats_dtype, **kwargs)
        # Ablation switches (Section 4.4); every default is the method itself.
        #   direction  "natural": X = F^-1 G | "raw": X = G
        #   length     "spectral": c = theta / sigma(X) | "before": theta / sigma(G) | "rms": the RMS of W X_in + X_out W
        #              is theta, as in Pion | "fisher": theta / sqrt(<X, F X>) | "quad": theta min(1, <G, X> / <X, F X>)
        #              | "fixed": c = theta.  F here is the Fisher at the current W and statistics.
        #   sides      "joint": one c for both sides, from the larger sigma | "each": each side by its own
        #   planes     "scaled": X as it is | "equal": every plane of X turned by theta (polar factor, one SVD per side)
        #   attn_mult  the angle of the square (attention) matrices times this
        #   angle_mult {id(param): m}: the angle of those matrices times m (a diagnostic: which matrices want more)
        #   a_window, d_window   steps before a rebuild in which A, D are fed (default: stats_window); a_every: A every
        #              n-th step only
        if direction not in ("natural", "raw"):
            raise ValueError(f"direction must be natural or raw, got {direction!r}")
        if length not in ("spectral", "before", "rms", "fisher", "quad", "fixed"):
            raise ValueError(f"unknown length rule {length!r}")
        if sides not in ("joint", "each") or planes not in ("scaled", "equal"):
            raise ValueError("sides must be joint or each, planes scaled or equal")
        if sides == "each" and length not in ("spectral", "before"):
            raise ValueError("sides='each' is defined for the spectral rules only")
        if planes == "equal" and (length != "spectral" or sides != "joint"):
            raise ValueError("planes='equal' sets the length itself: use it with length='spectral', sides='joint'")
        if attn_mult <= 0.0 or a_every < 1:
            raise ValueError("attn_mult must be positive and a_every at least 1")
        self.direction, self.length, self.sides, self.planes = direction, length, sides, planes
        self.attn_mult, self.a_window, self.d_window, self.a_every = attn_mult, a_window, d_window, a_every
        self.angle_mult = dict(angle_mult or {})
        # the spectral rule and Pion's RMS rule run batched without host synchronisation; the other rules and
        # switches take the per-matrix path of _ablated_scales
        self._ablated = (length not in ("spectral", "rms") or (sides, planes, attn_mult) != ("joint", "scaled", 1.0)
                         or bool(self.angle_mult) or (length == "rms" and not batch_sigma))
        self._angle_t = None                               # realised angle per unit, on the batched path (a tensor)
        self.angles: dict[int, float] | None = None    # realised angle per matrix, on the ablated length path
        self._gens: dict[tuple, torch.Tensor] = {}
        self.batch_sigma = batch_sigma
        self.share_root = share_root
        # tf32: "off" -- the whole step in true fp32 (the method); "upstream" -- fp32 products in TF32 everywhere except
        # the Cayley step and the rotation of W, so only the direction is affected (the rotation stays orthogonal to fp32 round-off);
        # "all" -- the Cayley step in TF32 too (measured, not assumed: scripts/probes/tf32_check.py)
        if tf32 not in ("off", "upstream", "all"):
            raise ValueError(f"tf32 must be off, upstream or all, got {tf32!r}")
        self.tf32 = tf32
        # fisher_dtype "bf16": the four products of F^-1 in bf16, the result made exactly skew in fp32, so the
        # Cayley step still rotates; only the direction is approximated
        if fisher_dtype not in ("fp32", "bf16"):
            raise ValueError(f"fisher_dtype must be fp32 or bf16, got {fisher_dtype!r}")
        self.fisher_dtype = fisher_dtype
        # floor_schedule "constant": the floor is eps (the method) | "lr": eps * lr(t) / lr(0), the floor follows the step
        # size down the schedule (the bases go stale in proportion to the step; late, a small step leaves them nearly fresh)
        if floor_schedule not in ("constant", "lr"):
            raise ValueError(f"floor_schedule must be constant or lr, got {floor_schedule!r}")
        self.floor_schedule = floor_schedule

    def _step_precision(self):
        return exact_fp32() if self.tf32 == "off" else tf32_matmul()

    def _solve(self, P, Pt, G, den):
        if self.fisher_dtype != "bf16":
            return super()._solve(P, Pt, G, den)
        Pb, Ptb = P.to(torch.bfloat16), Pt.to(torch.bfloat16)
        Y = ((Ptb @ G.to(torch.bfloat16) @ Pb).float() / den).to(torch.bfloat16)
        return skew((Pb @ Y @ Ptb).float())

    def _retraction_precision(self):
        return exact_fp32() if self.tf32 == "upstream" else nullcontext()

    def _collecting(self, param: torch.Tensor, kind: str = "a") -> bool:
        window = self.a_window if kind == "a" else self.d_window
        if window is None and (kind == "d" or self.a_every == 1):
            return super()._collecting(param, kind)
        group, state = self._group_of(param), self.state[param]
        if "bases" not in state:
            return True
        if window is None:
            window = group["stats_window"]
        inside = window is None or state["since_refactor"] >= group["t_fac"] - window
        return inside and (kind == "d" or state.get("step", 0) % self.a_every == 0)

    def _correct(self, X, G, P, den, items, rows, group):
        if self.direction == "raw":
            X = G
        if self.length in ("before", "quad"):            # the generators, for the rules that read them
            self._gens.update({(i, slot): G[k] for k, (i, slot, _, _) in enumerate(items)})
        return X

    def _floor(self, group: dict) -> float:
        if self.floor_schedule == "constant":
            return group["eps"]
        lr0 = group.setdefault("floor_lr0", group["lr"])   # the first rebuild is at step 0, at the peak (no warmup)
        return group["eps"] * group["lr"] / lr0

    def _basis_in(self, A: torch.Tensor, gram_in: torch.Tensor, eps: float, members: list):
        """`A^{-1/2}` once per distinct input moment (q, k, v and gate, up share one), then the rest per matrix."""
        if not self.share_root:
            return super()._basis_in(A, gram_in, eps, members)
        slot: dict[int, int] = {}
        order = []
        for j, (p, *_) in enumerate(members):
            key = id(self.state[p]["cov"])
            if key not in slot:
                slot[key] = len(order); order.append(j)
        roots = floored_inverse_root(A[order], eps)
        root = roots[[slot[id(self.state[p]["cov"])] for p, *_ in members]]
        return basis_congruence_from_root(root, gram_in, eps)

    def _scales(self, rows: list, group: dict) -> list:
        """The power iteration batched by (side size, iterations); `c` stays on the device, so the step makes no
        host synchronisation for it. Same estimate as the reference path up to the round-off of batched products."""
        if self._ablated:
            return self._ablated_scales(rows, group)
        if not self.batch_sigma:
            return super()._scales(rows, group)
        n = len(rows)
        sig = torch.zeros(2, n, dtype=rows[0][2].dtype, device=rows[0][2].device)
        groups: dict[tuple, list] = defaultdict(list)
        states = []
        for k, (p, W, X_in, X_out, b, _, _) in enumerate(rows):
            st = self._unit_state(self.state[p], b)
            st["since_refactor"] = self.state[p]["since_refactor"]
            states.append(st)
            fresh = "angle_v_in" not in st or st["since_refactor"] == 0
            it = group["angle_warmup"] if fresh else group["angle_iters"]
            groups[(X_in.shape[0], it)].append((0, k, X_in, "angle_v_in"))
            groups[(X_out.shape[0], it)].append((1, k, X_out, "angle_v_out"))
        for (_, it), items in groups.items():
            X = torch.stack([x for _, _, x, _ in items])
            vs = [states[k].get(key) for _, k, _, key in items]
            v = None if any(x is None for x in vs) else torch.stack(vs)
            s_, v = spectral_norm(X, it, v)
            for j, (side, k, _, key) in enumerate(items):
                states[k][key] = v[j]
                sig[side, k] = s_[j]
        if self.length == "rms":
            # Pion's rule: c such that the RMS of the induced change c (W X_in + X_out W) is theta, batched by shape
            rms = torch.empty(n, dtype=sig.dtype, device=sig.device)
            by_shape: dict[tuple, list] = defaultdict(list)
            for k, row in enumerate(rows):
                by_shape[tuple(row[1].shape)].append(k)
            for (m_, n_), ks in by_shape.items():
                W = torch.stack([rows[k][1] for k in ks])
                D = W @ torch.stack([rows[k][2] for k in ks]) + torch.stack([rows[k][3] for k in ks]) @ W
                rms[ks] = torch.linalg.matrix_norm(D) / (m_ * n_) ** 0.5
            c = group["lr"] / rms.clamp_min(1e-30)
            self._angle_t = c * sig.amax(dim=0)
            self._angle_ids = [id(row[0]) for row in rows]
        else:
            c = group["lr"] / sig.amax(dim=0).clamp_min(1e-30)
        return list(c.unbind(0))

    def realised_angles(self) -> dict[int, float] | None:
        """The largest rotation angle each matrix took in the last step (blocks: the largest of them), where a length
        rule other than the spectral one sets it; None under the spectral rule (theta for every matrix)."""
        if self._ablated:
            return self.angles
        if self._angle_t is None:
            return None
        out: dict[int, float] = {}
        for i, a in zip(self._angle_ids, self._angle_t.tolist()):
            out[i] = max(out.get(i, 0.0), a)
        return out

    def _largest_angle(self, group: dict) -> float:
        if self._ablated and self.angles:                  # the length rules set the angle themselves
            return max(self.angles.values())
        if not self._ablated and self._angle_t is not None:
            return float(self._angle_t.max())              # one host synchronisation: the Newton-Schulz count
        return super()._largest_angle(group)

    def _fisher_quadratic(self, p, W, r, whole, X_in, X_out):
        """<X, F X> summed over both sides, F at the current W with the statistics as last collected."""
        st = self.state[p]
        A = st["cov"].matrix.to(W.dtype)
        D = st["cov_backward"].matrix.to(W.dtype)
        if not whole:
            D = D[r, r]
        Wt = W.transpose(-1, -2)
        S, V = Wt @ D @ W, W @ A @ Wt
        F_in = 0.5 * (S @ X_in @ A + A @ X_in @ S)
        F_out = 0.5 * (V @ X_out @ D + D @ X_out @ V)
        return (X_in * F_in).sum() + (X_out * F_out).sum()

    def _ablated_scales(self, rows: list, group: dict) -> list:
        """The length rules of Section 4.4, one matrix at a time. Each side's X is scaled in place to the step it
        takes (c_in X_in, c_out X_out) and the returned c is 1. Records the realised angle per matrix."""
        gens, self._gens = self._gens, {}
        self.angles = {}
        cs = []
        for i, (p, W, X_in, X_out, b, r, whole) in enumerate(rows):
            st = self._unit_state(self.state[p], b)
            st["since_refactor"] = self.state[p]["since_refactor"]
            theta = group["lr"] * (self.attn_mult if p.shape[0] == p.shape[1] else 1.0) * self.angle_mult.get(id(p), 1.0)
            if self.planes == "equal":
                X_in, X_out = _polar_skew(X_in), _polar_skew(X_out)
                c_in = c_out = theta
                angle = theta
            else:
                fresh = "angle_v_in" not in st or st["since_refactor"] == 0
                it = group["angle_warmup"] if fresh else group["angle_iters"]
                # sigma sets c only under the angle rules, and the retraction only through its number of squarings or
                # iterations; the exact Cayley solve with any other rule needs neither, so it is skipped (the logged
                # angle is then unknown)
                if group["retraction"] == "cayley" and self.length not in ("spectral", "before"):
                    s_in = s_out = float("nan")
                else:
                    s_in, st["angle_v_in"] = spectral_norm(X_in, it, st.get("angle_v_in"))
                    s_out, st["angle_v_out"] = spectral_norm(X_out, it, st.get("angle_v_out"))
                    s_in, s_out = float(s_in), float(s_out)
                if self.length == "before":
                    g_in, g_out = gens[(i, 2)], gens[(i, 3)]
                    s_in_g, st["g_v_in"] = spectral_norm(g_in, it, st.get("g_v_in"))
                    s_out_g, st["g_v_out"] = spectral_norm(g_out, it, st.get("g_v_out"))
                    ref_in, ref_out = float(s_in_g), float(s_out_g)
                else:
                    ref_in, ref_out = s_in, s_out
                if self.length in ("spectral", "before"):
                    if self.sides == "each":
                        c_in, c_out = theta / max(ref_in, 1e-30), theta / max(ref_out, 1e-30)
                    else:
                        c_in = c_out = theta / max(ref_in, ref_out, 1e-30)
                elif self.length == "rms":
                    m, n = W.shape
                    rms = float(torch.linalg.matrix_norm(W @ X_in + X_out @ W)) / (m * n) ** 0.5
                    c_in = c_out = theta / max(rms, 1e-30)
                elif self.length == "fixed":
                    c_in = c_out = theta
                else:
                    curv = float(self._fisher_quadratic(p, W, r, whole, X_in, X_out))
                    if self.length == "fisher":
                        c_in = c_out = theta / max(curv, 1e-30) ** 0.5
                    else:
                        quad = float((gens[(i, 2)] * X_in).sum() + (gens[(i, 3)] * X_out).sum())
                        c_in = c_out = theta * min(1.0, quad / max(curv, 1e-30))
                angle = max(c_in * s_in, c_out * s_out)
            rows[i][2], rows[i][3] = c_in * X_in, c_out * X_out
            self.angles[id(p)] = angle if angle != angle else max(self.angles.get(id(p), 0.0), angle)   # NaN: unknown
            cs.append(1.0)
        return cs


def _polar_skew(X: torch.Tensor) -> torch.Tensor:
    """The polar factor of skew X, kept skew: X = U S V^T goes to U V^T on the planes above round-off (1e-6 of the
    largest), so every plane of X turns by the same angle. Computed as X (X^T X)^{-1/2} from the symmetric eigenproblem
    of X^T X = -X^2 (the same factor as from an SVD, which on these shapes runs tens of seconds per step on a GPU)."""
    lam, V = safe_eigh(-(X @ X))
    s = lam.clamp_min(0).sqrt()
    keep = s > 1e-6 * s.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    inv = torch.where(keep, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
    Y = X @ (V * inv.unsqueeze(-2)) @ V.transpose(-1, -2)
    return 0.5 * (Y - Y.transpose(-1, -2))


def default_of(cls: type, name: str):
    """The default a Natural Pion class gives keyword `name` (its own, or the nearest base class's)."""
    import inspect
    for k in cls.__mro__:
        p = inspect.signature(k.__init__).parameters.get(name) if "__init__" in vars(k) else None
        if p is not None and p.default is not inspect.Parameter.empty:
            return p.default
    raise KeyError(f"{cls.__name__} has no default for {name!r}")


BY_NAME = {"natural-pion": NaturalPion, "natural-pion-fast": NaturalPionFast}
