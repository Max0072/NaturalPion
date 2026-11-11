"""What one refactor and one Cayley step cost at Qwen3-1.7B's shapes (RLVR_PLAN.md 3.2, items 5-6).

Qwen3-1.7B: hidden 2048, FFN 6144, 28 layers, k/v 1024 x 2048. Per layer the rotated sides are
    6144: gate out, up out, down in                  (3)
    2048: q in/out, o in/out, k in, v in, gate in, up in, down out   (9)
    1024: k out, v out                               (2)
A refactor solves one generalized eigenproblem per side by the congruence path: three fp32 `eigh`
(B, C, the pencil) plus matrix products. A step solves one Cayley system per side.
Random SPD inputs with a spread spectrum; timing only, nothing is trained.

    python scripts/probes/rl_refactor_cost.py [--layers 28] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from ngd_pion.factorization import basis_congruence
from ngd_pion.linalg import exact_fp32

SIDES = {6144: 3, 2048: 9, 1024: 2}


def spd(b: int, n: int, dev) -> torch.Tensor:
    X = torch.randn(b, n, n // 4, device=dev)
    return X @ X.transpose(-1, -2) / n + 1e-3 * torch.eye(n, device=dev)


def timed(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=12, help="matrices per batched call")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = {"gpu": torch.cuda.get_device_name(0), "layers": args.layers, "per_size": {}}
    total_ref = total_cay = 0.0
    for n, per_layer in SIDES.items():
        count = per_layer * args.layers
        b = min(args.chunk, count)
        calls = -(-count // b)
        B, C = spd(b, n, dev), spd(b, n, dev)
        X = torch.randn(b, n, n, device=dev)
        X = 0.5 * (X - X.transpose(-1, -2))
        eye = torch.eye(n, device=dev).expand_as(X)

        def ref():
            with exact_fp32():
                basis_congruence(B, C, 1e-4)

        def cay():
            with exact_fp32():
                torch.linalg.solve(eye + 0.05 * X, eye - 0.05 * X)

        t_ref, t_cay = timed(ref, args.repeats) * calls, timed(cay, args.repeats) * calls
        total_ref += t_ref
        total_cay += t_cay
        out["per_size"][n] = {"sides": count, "batch": b, "refactor_s": round(t_ref, 3), "cayley_s": round(t_cay, 3)}
        print(json.dumps({n: out["per_size"][n]}), flush=True)
        del B, C, X
        torch.cuda.empty_cache()
    out["refactor_total_s"] = round(total_ref, 2)
    out["cayley_total_s"] = round(total_cay, 2)
    out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
