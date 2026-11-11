"""Seconds per optimizer step on the shapes of LLaMA-60M, nothing else: no model, synthetic statistics and gradients.

56 rotated matrices (8 layers x wq, wk, wv, wo 512x512; gate, up 1376x512; down 512x1376). Every step feeds the
statistics hooks with 131 072 tokens (q, k, v and gate, up share their input tensor, as in the model) and fresh
gradients, then calls `step()`. Timed over 100 steps (four refresh periods) after 50 warm-up steps, GPU-synchronised.

    python scripts/probes/optimizer_step_bench.py     # writes runs-record/probes/optimizer_step_bench.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from ngd_pion.linalg import exact_fp32
from ngd_pion.natural_pion import NaturalPion, NaturalPionFast


def qb(ps):
    """The query projection of every layer (the first of its seven matrices) in 8 head blocks."""
    return {id(p): 8 for p in ps[0::7]}
from ngd_pion.pion_baseline import Pion

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda")
N, H, F, L = 131072, 512, 1376, 8


def params():
    g = torch.Generator(device="cpu").manual_seed(0)
    out = []
    for _ in range(L):
        for shape in [(H, H)] * 4 + [(F, H), (F, H), (H, F)]:
            out.append(torch.nn.Parameter((torch.randn(*shape, generator=g) * 0.02).to(DEV)))
    return out


def run(name, make, steps=100, warm=50, compile_step=False, parts=False):
    torch.manual_seed(0)
    ps = make_ps = params()
    opt = make(ps)
    feeds = isinstance(opt, NaturalPion)
    xa, xo, xf = (torch.randn(N, H, device=DEV), torch.randn(N, H, device=DEV), torch.randn(N, F, device=DEV))
    dH, dF = torch.randn(N, H, device=DEV) / N, torch.randn(N, F, device=DEV) / N
    grads = [torch.randn_like(p) * 1e-3 for p in ps]
    step = torch.compile(opt.step) if compile_step else opt.step
    times, stats_t = [], 0.0
    for t in range(warm + steps):
        if parts and t == warm:
            opt.profile = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if feeds:
            for i, p in enumerate(ps):
                k = i % 7
                x = xa if k < 3 else xo if k == 3 else xa if k < 6 else xf
                opt.observe(p, x)
                opt.observe_backward(p, dF if k in (4, 5) else dH)
        if parts and t >= warm:
            torch.cuda.synchronize(); stats_t += time.perf_counter() - t0
        for p, gr in zip(ps, grads):
            p.grad = gr
        step()
        torch.cuda.synchronize()
        if t >= warm:
            times.append(time.perf_counter() - t0)
    s = sum(times) / len(times)
    print(f"{name:42s} {s:.4f} s/step", flush=True)
    if parts:
        prof = {"statistics": stats_t / steps, **{k: v / steps for k, v in (opt.profile or {}).items()}}
        prof["total (synchronised)"] = s
        for k, v in prof.items():
            print(f"    {k:28s} {v:.4f}", flush=True)
        del opt, ps
        torch.cuda.empty_cache()
        return prof
    del opt, ps
    torch.cuda.empty_cache()
    return s


def main():
    torch.backends.cuda.matmul.allow_tf32 = True          # as in training; the optimizers guard their own fp32
    out = {"gpu": torch.cuda.get_device_name(0)}
    # the two forms of Table 7 at the method's scale (NP_LR: the best point of the k-series 3000-step grid)
    lr = float(os.environ.get("NP_LR", "6e-4"))
    out["np_lr"] = lr
    cases = {
        # every rotating optimizer rotates the query projection per head (harness.train.query_blocks)
        "pion": lambda ps: Pion(ps, lr=1e-3, scaling="rms", rms=0.2, momentum="lie", retraction="trunc", beta1=0.9,
                                row_blocks=qb(ps)),
        # after Appendix C.4 (v2.sbatch row c-form46): batched by shape, exact Cayley, fp32 statistics
        "after44": lambda ps: NaturalPionFast(ps, lr=lr, eps=1e-3, retraction="cayley", stats_dtype="fp32",
                                              share_inputs=False, batch_sigma=False, length="rms", row_blocks=qb(ps)),
        # the method
        "final": lambda ps: NaturalPionFast(ps, lr=lr, eps=1e-3, length="rms", tf32="upstream", fisher_dtype="bf16",
                                            row_blocks=qb(ps)),
    }
    for name, make in cases.items():
        out[name] = run(name, make)
    for name in ("after44", "final"):
        out[name + "_parts"] = run(name + " (parts, synchronised)", cases[name], parts=True)
    (ROOT / "runs-record" / "probes" / "optimizer_step_bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    with exact_fp32():
        pass
    main()
