"""Floating-point operations per training step, analytically: the model's forward and backward pass, and each part of
the optimizer step for Natural Pion Fast and for Pion. Hardware- and implementation-independent; it is what the
wall-clock overhead would be if every part ran at the same throughput.

Counting conventions (one multiply-add = 2 FLOP):
  * model: 6 * tokens * (matrix parameters, head included) + 12 * layers * hidden * seq * tokens (attention scores)
  * a product of (a x b) by (b x c): 2abc
  * a symmetric eigendecomposition with eigenvectors of an n x n matrix: 9 n^3 (Golub & Van Loan's count for the
    tridiagonal QR route; libraries vary, so treat the refresh row as an estimate)
Per rotated unit W (m x n, out x in; a matrix, or one head's row block of the query projection), side sizes m (out)
and n (in):
  Pion:          generators W^T G, G W^T          2mn^2 + 2m^2n
                 truncated exponential, both sides 2n^3 + 2m^3            (one square of each generator)
                 RMS scale W O_in + O_out W        2mn^2 + 2m^2n
                 rotation O_out W, W O_in          2m^2n + 2mn^2
  Natural Pion Fast (as used from 2026-11-04):
                 generators                        2mn^2 + 2m^2n
                 F^-1, both sides                  8n^3 + 8m^3            (P^T G P, P Y P^T)
                 RMS scale W X_in + X_out W        2mn^2 + 2m^2n          (Pion's length rule)
                 Cayley by Newton-Schulz in Z - I  8n^3 + 8m^3            (two iterations, two products each; the
                                                                          squarings of large-angle steps not counted)
                 rotation                          2m^2n + 2mn^2
                 sigma                             negligible (matrix-vector products)
                 statistics, 6 of 25 steps         2 T_s n^2 per distinct input, 2 T_s m^2 per output, over the
                                                   first micro-batch only (T_s = micro-batch tokens, half the step)
                 refresh, 1 of 25 steps            per side 3 eigendecompositions + 5 products of the side size,
                                                   plus the grams W A W^T and W^T D W (2mn^2 + 2m^2n each)

    python scripts/analysis/flops.py        # prints, writes runs-record/probes/flops.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T_FAC, WINDOW = 25, 6
MODELS = {   # hidden, ffn, layers, vocab, seq, batch (sequences): ours (60M) and Pion's 1.3B
    "60M": dict(h=512, f=1376, L=8, heads=8, V=32100, seq=256, batch=512, micro=256),
    "1.3B": dict(h=2048, f=5460, L=24, heads=16, V=32100, seq=256, batch=512, micro=256),
}


def matrices(c):
    """(m, n, input group, units) per rotated matrix: q, k, v share an input, gate and up share one. The query
    projection rotates as `heads` row blocks (harness.train.query_blocks), each with its own pair of rotations."""
    h, f, k = c["h"], c["f"], c["heads"]
    out = []
    for l in range(c["L"]):
        out += [(h, h, (l, "attn"), k), (h, h, (l, "attn"), 1), (h, h, (l, "attn"), 1), (h, h, (l, "o"), 1),
                (f, h, (l, "ffn"), 1), (f, h, (l, "ffn"), 1), (h, f, (l, "down"), 1)]
    return out


def units(mats):
    """(m, n) of every rotated unit: a matrix, or one row block of it."""
    return [(m // k, n) for m, n, _, k in mats for _ in range(k)]


def count(c):
    T = c["seq"] * c["batch"]
    mats = matrices(c)
    us = units(mats)                                # the parts of the step that act per rotated unit
    params = sum(m * n for m, n, _, _ in mats) + c["V"] * c["h"]
    model = 6 * T * params + 12 * c["L"] * c["h"] * c["seq"] * T
    gen = sum(2 * m * n * n + 2 * m * m * n for m, n in us)
    rot = gen
    pion = {"generators": gen, "truncated exp": sum(2 * n ** 3 + 2 * m ** 3 for m, n in us),
            "RMS scale": gen, "rotation": rot}
    inputs = {}
    for m, n, g, _ in mats:
        inputs[g] = n
    Ts = c["seq"] * c["micro"]                      # the hooks record the first micro-batch of each step only
    # D is one gram per matrix over its full output (a query block reads its diagonal block)
    stats = (sum(2 * Ts * n * n for n in inputs.values()) + sum(2 * Ts * m * m for m, _, _, _ in mats)) * WINDOW / T_FAC
    side = lambda k: 3 * 9 * k ** 3 + 5 * 2 * k ** 3
    refresh = sum(side(n) + side(m) + 2 * (2 * m * n * n + 2 * m * m * n) for m, n in us) / T_FAC
    npf = {"generators": gen, "RMS scale": gen, "F^-1": sum(8 * n ** 3 + 8 * m ** 3 for m, n in us),
           "Cayley (Newton-Schulz)": sum(8 * n ** 3 + 8 * m ** 3 for m, n in us), "rotation": rot,
           "statistics": stats, "refresh": refresh}
    return model, pion, npf


def main():
    out = {}
    for name, c in MODELS.items():
        model, pion, npf = count(c)
        p, q = sum(pion.values()), sum(npf.values())
        out[name] = {"model": model, "pion": pion, "natural_pion_fast": npf, "pion_total": p, "np_total": q,
                     "pion_overhead": p / model, "np_overhead": q / model, "np_over_pion": (model + q) / (model + p)}
        print(f"{name}: model {model:.3e} FLOP/step; optimizer Pion {p/model:.2%}, Natural Pion Fast {q/model:.2%}; "
              f"step ratio {(model + q) / (model + p):.3f}")
        for k, v in npf.items():
            print(f"    NP {k:24s} {v / model:.2%}")
    (ROOT / "runs-record" / "probes" / "flops.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
