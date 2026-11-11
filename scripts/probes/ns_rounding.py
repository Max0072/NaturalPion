"""Rounding bias of the Newton-Schulz retraction in fp32, and its correction form. CPU, a minute.

For a random skew X with ||X||_2 = 1 and A = theta/2 X, the retraction R ~ (I+A)^-1 (I-A) is computed
  cayley    by an exact solve;
  ns        by two Newton-Schulz iterations Z <- Z (2I - M Z), Z0 = I - A, R = 2Z - I (ngd_pion retraction "ns");
  ns_delta  by the same iterations carried in z = Z - I, r = A + z + A z, z <- z - r - z r, R = I + 2z ("ns_delta");
and applied 200 times to a random W (one side). Reported: the relative change of ||W||_F^2 (fp64), which a
rotation keeps exactly, and max |R_ns_delta - R_cayley|. Writes runs-record/probes/ns_rounding.json.

    python scripts/probes/ns_rounding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def ns(A, k=2):
    eye = torch.eye(A.shape[-1]); M, Z = eye + A, eye - A
    for _ in range(k):
        Z = Z @ (2 * eye - M @ Z)
    return 2 * Z - eye


def ns_delta(A, k=2):
    z = -A
    for _ in range(k):
        r = A + z + A @ z
        z = z - r - z @ r
    return torch.eye(A.shape[-1]) + 2 * z


def main():
    torch.manual_seed(0)
    out = {}
    for n in (256, 512, 1376):
        for th in (0.2, 0.05):
            G = torch.randn(n, n); X = G - G.T; X = X / torch.linalg.matrix_norm(X, 2); A = 0.5 * th * X
            eye = torch.eye(n)
            Rs = {"cayley": torch.linalg.solve(eye + A, eye - A), "ns": ns(A), "ns_delta": ns_delta(A)}
            W = torch.randn(n, n) / n ** 0.5
            row = {}
            for name, R in Rs.items():
                V = W.clone()
                for _ in range(200):
                    V = R @ V
                row[name] = (V.double().square().sum() / W.double().square().sum() - 1).item()
            row["delta_vs_cayley"] = (Rs["ns_delta"] - Rs["cayley"]).abs().max().item()
            out[f"n{n}_th{th}"] = row
            print(n, th, row, flush=True)
    dst = ROOT / "runs-record" / "probes" / "ns_rounding.json"
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
