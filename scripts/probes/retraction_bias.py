"""Is the fp32 rounding of the Cayley inverse biased, and does the Z - I form remove the bias? CPU, minutes.

For random skew X with ||X||_2 = 1 and A = theta/2 X, R is computed in fp32 by the exact solve
(I + A)^-1 (I - A), by standard Newton-Schulz (two iterations) and by Newton-Schulz carried in z = Z - I. For each
draw t = tr(R^T R - I) / n in fp64; a rotation has t = 0, and t > 0 means R inflates ||W||_F. Reported: mean of t,
its standard deviation, and the mean in units of its standard error (|z| > 3: a bias, not noise).

    python scripts/probes/retraction_bias.py     # writes runs-record/probes/retraction_bias.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def solve(A, eye):
    return torch.linalg.solve(eye + A, eye - A)


def ns(A, eye, k=2):
    M, Z = eye + A, eye - A
    for _ in range(k):
        Z = Z @ (2 * eye - M @ Z)
    return 2 * Z - eye


def ns_delta(A, eye, k=2):
    z = -A
    for _ in range(k):
        r = A + z + A @ z
        z = z - r - z @ r
    return eye + 2 * z


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = {}
    for n, draws in ((512, 400), (1376, 100)):
        eye = torch.eye(n)
        for th in (0.2, 0.05):
            ts = {"solve": [], "ns": [], "ns_delta": []}
            for _ in range(draws):
                G = torch.randn(n, n); X = G - G.T; X = X / torch.linalg.matrix_norm(X, 2); A = 0.5 * th * X
                for name, f in (("solve", solve), ("ns", ns), ("ns_delta", ns_delta)):
                    R = f(A, eye).double()
                    ts[name].append(float(((R.T @ R).diagonal() - 1).sum() / n))
            row = {}
            for name, v in ts.items():
                t = torch.tensor(v, dtype=torch.float64)
                row[name] = {"mean": float(t.mean()), "sd": float(t.std()), "z": float(t.mean() / (t.std() / len(v) ** 0.5))}
            out[f"n{n}_th{th}"] = row
            print(n, th, {k: f"mean {r['mean']:+.2e} sd {r['sd']:.1e} z {r['z']:+.1f}" for k, r in row.items()}, flush=True)
    (ROOT / "runs-record" / "probes" / "retraction_bias.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
