"""How to compute the Cayley step at large angles, on a GPU, on the batches of LLaMA-60M's rotated sides (512: 24 matrices
plus the 64 query head blocks' input side; 1376: 24; 64: the 64 query head blocks' output side). For skew X scaled so
that its largest plane turns by `angle`, three ways to get R:

  ns       Newton-Schulz in Z - I with as many iterations as the angle needs (natural_pion.ns_iterations); the exact
           solve above 1.8 rad (the method today)
  solve    the exact solve (I + A) R = (I - A)
  square   scaling and squaring: A / 2^k with angle / 2^k <= 0.2, two Newton-Schulz iterations, then k squarings carried
           in the correction R = I + D, D <- 2 D + D^2 (nothing next to 1)

Reported: milliseconds per batch (median of 20, synchronised), max ||R^T R - I||_2 over the batch (fp32), and the
largest plane's angle of R.

    python scripts/probes/cayley_speed.py      # writes runs-record/probes/cayley_speed.json
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from ngd_pion.linalg import exact_fp32, skew
from ngd_pion.natural_pion import ns_iterations

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
SIZES = {512: 24 + 64, 1376: 24, 64: 64}


def snorm(X):
    return torch.linalg.eigvalsh(-(X.double() @ X.double())).amax(dim=-1).clamp_min(0).sqrt()


def method(A, how, angle):
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    if how == "solve" or (how == "ns" and ns_iterations(angle, 2) is None):
        return torch.linalg.solve(eye + A, eye - A)
    if how == "ns":
        z = -A
        for _ in range(ns_iterations(angle, 2)):
            r = A + z + A @ z
            z = z - r - z @ r
        return eye + 2.0 * z
    k = max(0, math.ceil(math.log2(angle / 0.2))) if angle > 0.2 else 0
    As = A / 2 ** k
    z = -As
    for _ in range(2):
        r = As + z + As @ z
        z = z - r - z @ r
    D = 2.0 * z
    for _ in range(k):
        D = 2.0 * D + D @ D
    return eye + D


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "rows": []}
    g = torch.Generator(device="cpu").manual_seed(0)
    with exact_fp32():
        for n, b in SIZES.items():
            X = skew(torch.randn(b, n, n, generator=g)).to(DEV)
            X = X / snorm(X).float().reshape(-1, 1, 1)                      # largest plane: unit
            for angle in (0.05, 0.2, 0.4, 0.8, 1.6, 2.5):
                A = (0.5 * angle) * X                                          # Cay(cX) with c sigma = angle
                for how in ("ns", "solve", "square"):
                    method(A, how, angle); torch.cuda.synchronize()
                    ts = []
                    for _ in range(20):
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        R = method(A, how, angle)
                        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                    eye = torch.eye(n, device=DEV, dtype=torch.float64)
                    Rd = R.double()
                    orth = float(torch.linalg.matrix_norm(Rd.transpose(-1, -2) @ Rd - eye, 2).max())
                    ang = float(torch.angle(torch.linalg.eigvals(Rd)).abs().max())
                    row = {"n": n, "batch": b, "angle": angle, "how": how, "ms": 1e3 * statistics.median(ts),
                           "orth_err": orth, "R_angle": ang}
                    out["rows"].append(row)
                    print(f"n {n:5d} x{b:3d} angle {angle:4.2f} {how:6s} {row['ms']:8.2f} ms  ||R^T R - I|| {orth:.1e}"
                          f"  R's largest angle {ang:.3f}", flush=True)
    (ROOT / "runs-record" / "probes" / "cayley_speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
