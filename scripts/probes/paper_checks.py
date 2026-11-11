"""The measurements the preprint quotes that are not training runs. Writes runs-record/probes/paper_checks.json.

    apptainer exec --nv --bind $DATA_p330 $SIF python scripts/probes/paper_checks.py [checkpoint.pt]

Needs a GPU (timings, TF32). The checkpoint is a Natural Pion Fast run (default: the 3000-step run behind
the ablations); its initial weights are rebuilt from the run's seed, as the harness builds them.

Every quantity is defined here as the paper states it:
* power_*: relative error of the warm-start-free (cold) power iteration of ngd_pion.linalg.spectral_norm on a
  random 512 x 512 skew matrix, after 1, 5, 20 iterations; warm: one iteration from the vector converged on a
  matrix that differs from this one by 1% (relative, Frobenius).
* svd_*: exact matrix_norm(X, 2) against 2 power iterations on the 112 skew sides of LLaMA-60M's 56 rotated
  matrices (88 of size 512, 24 of size 1376), fp32, GPU-synchronised, best of 3.
* ns_*: Newton-Schulz inverse (2 iterations) in place of the Cayley solve. One-step ||R^T R - I||_max in fp64
  on a 256 x 256 skew matrix at largest angle 0.2 / 0.4 / 0.8; fp32 one-step errors of both; 3000 two-sided
  steps on a random 256 x 256 W at a cosine angle from 0.2, max relative singular-value change; cost of the
  NS inverse over the solve on the real side sizes (batched, fp32).
* tf32_drift: 200 two-sided Cayley steps at angle 1e-2 on a random 512 x 1376 W, max relative singular-value
  change with TF32 allowed in the solve, and without.
* fp32_drift_*: for each of the 56 rotated matrices of the checkpoint, max_k |s_k / s_k(init) - 1|; median and
  max over matrices, and the mean signed change.
* kfac_vs_exact: along the Natural Pion in-side direction X = F^-1 G_in of three layers (square, tall, wide) of
  the checkpoint, the K-FAC curvature <X, F X> over the exact per-token curvature E_t[(u_t^T X x_t)^2]
  (empirical Fisher, 128 sequences).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.config import RunConfig, ModelConfig  # noqa: E402
from harness.data import TokenCorpus  # noqa: E402
from harness.model import Transformer  # noqa: E402
from ngd_pion.direction import fisher_apply, generators, natural_gradient  # noqa: E402
from ngd_pion.factorization import basis_congruence  # noqa: E402
from ngd_pion.linalg import cayley, cayley_newton_schulz, exact_fp32, skew, spectral_norm  # noqa: E402

DEV = torch.device("cuda")
OUT = {}


def rand_skew(n, dt=torch.float64, seed=0, dev="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return skew(torch.randn(n, n, generator=g, dtype=dt)).to(dev)


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()
    best = math.inf
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); best = min(best, time.perf_counter() - t)
    return best


def power_checks():
    X = rand_skew(512)
    true = float(torch.linalg.matrix_norm(X, 2))
    for it in (1, 5, 20):
        s, _ = spectral_norm(X, it)
        OUT[f"power_cold_{it}"] = abs(float(s) - true) / true
    Y = X + 0.01 * torch.linalg.matrix_norm(X) / torch.linalg.matrix_norm(rand_skew(512, seed=1)) * rand_skew(512, seed=1)
    _, v = spectral_norm(Y, 200)
    s, _ = spectral_norm(X, 1, v)
    OUT["power_warm_1"] = abs(float(s) - true) / true


def svd_timing():
    sides = [rand_skew(512, torch.float32, seed=i, dev=DEV) for i in range(88)] + \
            [rand_skew(1376, torch.float32, seed=100 + i, dev=DEV) for i in range(24)]
    vs = [spectral_norm(X, 50)[1] for X in sides]
    with exact_fp32():
        t_exact = timed(lambda: [torch.linalg.matrix_norm(X, 2) for X in sides])
        t_power = timed(lambda: [spectral_norm(X, 2, v) for X, v in zip(sides, vs)])
    OUT["svd_exact_s"], OUT["svd_power_s"] = t_exact, t_power
    OUT["power_speedup"] = t_exact / t_power


def ns_checks():
    X = rand_skew(256)
    X = X / torch.linalg.matrix_norm(X, 2)
    I = torch.eye(256, dtype=torch.float64)
    for ang in (0.2, 0.4, 0.8):
        R = cayley_newton_schulz(X, ang, 2)
        OUT[f"ns_err_{str(ang).replace('.', '')}"] = float((R.T @ R - I).abs().max())
    Xf, If = X.float().to(DEV), I.float().to(DEV)
    with exact_fp32():
        OUT["fp32_onestep_cayley"] = float((cayley(Xf, 0.2).T @ cayley(Xf, 0.2) - If).abs().max())
        Rn = cayley_newton_schulz(Xf, 0.2, 2)
        OUT["fp32_onestep_ns"] = float((Rn.T @ Rn - If).abs().max())
    # 3000 two-sided steps on a toy W
    g = torch.Generator(device="cpu").manual_seed(5)
    W0 = torch.randn(256, 256, generator=g).to(DEV)
    s0 = torch.linalg.svdvals(W0.double())
    def trunc(X, c):
        A = -c * X
        return torch.eye(X.shape[-1], device=X.device, dtype=X.dtype) + A + 0.5 * (A @ A)
    for name, ret in (("cayley", lambda X, c: cayley(X, c)), ("ns", lambda X, c: cayley_newton_schulz(X, c, 2)),
                      ("trunc", trunc)):
        W = W0.clone()
        gg = torch.Generator(device="cpu").manual_seed(6)
        with exact_fp32():
            for t in range(3000):
                th = 1e-5 + 0.5 * (0.2 - 1e-5) * (1 + math.cos(math.pi * t / 3000))
                Xi = skew(torch.randn(256, 256, generator=gg)).to(DEV)
                Xo = skew(torch.randn(256, 256, generator=gg)).to(DEV)
                c = th / max(float(torch.linalg.matrix_norm(Xi, 2)), float(torch.linalg.matrix_norm(Xo, 2)))
                W = ret(Xo, c) @ W @ ret(Xi, c)
        OUT[f"{name}_drift_toy"] = float((torch.linalg.svdvals(W.double()) / s0 - 1).abs().max())
    # cost of the NS inverse relative to the solve, real side sizes, batched
    for n, b in ((512, 88), (1376, 24)):
        Xb = torch.stack([rand_skew(n, torch.float32, seed=i, dev=DEV) for i in range(b)]) * (0.1 / n ** 0.5)
        with exact_fp32():
            OUT[f"ns_cost_{n}"] = timed(lambda: cayley_newton_schulz(Xb, 1.0, 2)) / timed(lambda: cayley(Xb, 1.0))
    OUT["ns_cost"] = (88 * OUT["ns_cost_512"] + 24 * OUT["ns_cost_1376"]) / 112


def tf32_check():
    g = torch.Generator(device="cpu").manual_seed(7)
    W0 = torch.randn(512, 1376, generator=g).to(DEV)
    s0 = torch.linalg.svdvals(W0.double())
    for name, exact in (("tf32", False), ("fp32", True)):
        W = W0.clone()
        gg = torch.Generator(device="cpu").manual_seed(8)
        torch.backends.cuda.matmul.allow_tf32 = not exact
        torch.set_float32_matmul_precision("highest" if exact else "high")
        for _ in range(200):
            Xo = skew(torch.randn(512, 512, generator=gg)).to(DEV)
            Xi = skew(torch.randn(1376, 1376, generator=gg)).to(DEV)
            c = 1e-2 / max(float(torch.linalg.matrix_norm(Xi.double(), 2)), float(torch.linalg.matrix_norm(Xo.double(), 2)))
            n_o, n_i = 512, 1376
            Ro = torch.linalg.solve(torch.eye(n_o, device=DEV) + 0.5 * c * Xo, torch.eye(n_o, device=DEV) - 0.5 * c * Xo)
            Ri = torch.linalg.solve(torch.eye(n_i, device=DEV) + 0.5 * c * Xi, torch.eye(n_i, device=DEV) - 0.5 * c * Xi)
            W = Ro @ W @ Ri
        OUT[f"{name}_drift_200"] = float((torch.linalg.svdvals(W.double()) / s0 - 1).abs().max())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    OUT["tf32_drift"] = OUT["tf32_drift_200"]


def checkpoint_checks(ckpt: Path):
    man = json.load(open(ckpt.parent / "manifest.json"))
    cfg_d = man["config"]
    mcfg = ModelConfig(**cfg_d["model"])
    torch.manual_seed(cfg_d["seed"])
    init = Transformer(mcfg)
    trained = Transformer(mcfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    trained.load_state_dict(sd["model"])
    OUT["checkpoint"] = str(ckpt.relative_to(ckpt.parents[3])) if len(ckpt.parents) > 3 else str(ckpt)
    OUT["checkpoint_step"] = sd.get("step")
    lin0, _ = init.parameter_split()
    lin1, _ = trained.parameter_split()
    worst, signed = [], []
    for a, b in zip(lin0, lin1):
        s0 = torch.linalg.svdvals(a.weight.detach().double())
        s1 = torch.linalg.svdvals(b.weight.detach().double())
        r = s1 / s0 - 1
        worst.append(float(r.abs().max()))
        signed.append(float(r.mean()))
    worst.sort()
    OUT["fp32_drift_3000_median"] = worst[len(worst) // 2]
    OUT["fp32_drift_3000_max"] = worst[-1]
    OUT["fp32_drift_3000_signed_mean"] = sum(signed) / len(signed)
    OUT["fp32_drift_3000"] = OUT["fp32_drift_3000_median"]

    # K-FAC against the exact per-token curvature along X = F^-1 G_in
    model = trained.to(DEV)
    linears, _ = model.parameter_split()
    chosen = [linears[1], linears[5], linears[6]]      # k (square; q rotates per head), up, down of layer 0
    acts, grads, hs = {}, {}, []
    for m in chosen:
        hs.append(m.register_forward_pre_hook(lambda mod, inp, _m=m: acts.__setitem__(id(_m), inp[0].detach())))
        def fwd(mod, inp, out, _m=m):
            out.register_hook(lambda g, k=id(_m): grads.__setitem__(k, g.detach()))
        hs.append(m.register_forward_hook(fwd))
    data = TokenCorpus(f"{os.environ['DATA_p330']}/c4/c4_train.bin", mcfg.seq_len, seed=cfg_d["seed"])
    x, y = data.batch(128, DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    for h in hs:
        h.remove()
    ratios = []
    for m in chosen:
        W, G = m.weight.detach().float(), m.weight.grad.detach().float()
        xs = acts[id(m)].reshape(-1, W.shape[1]).float()
        ds = grads[id(m)].reshape(-1, W.shape[0]).float() * xs.shape[0]
        us = ds @ W
        with exact_fp32():
            A = xs.T @ xs / xs.shape[0]
            S = us.T @ us / us.shape[0]
            X = natural_gradient(generators(W, G)[0], basis_congruence(A, S, 1e-4))
            kfac = float((X * fisher_apply(A, S, X)).sum())
            exact = float((((us @ X) * xs).sum(-1) ** 2).mean())
        ratios.append(kfac / exact)
    OUT["kfac_vs_exact_layers"] = ratios
    OUT["kfac_vs_exact"] = sorted(ratios)[1]


def main():
    # `paper_checks.py <checkpoint> <tag>` measures only that checkpoint's spectrum (and K-FAC) and writes
    # runs-record/probes/paper_checks_<tag>.json; without a tag, everything, into paper_checks.json.
    if len(sys.argv) > 2:
        checkpoint_checks(Path(sys.argv[1]))
        dst = ROOT / "runs-record" / "probes" / f"paper_checks_{sys.argv[2]}.json"
        json.dump(OUT, open(dst, "w"), indent=1)
        print(json.dumps(OUT, indent=1))
        return
    # default: the method, 3000 steps, seed 0: the best point of the k-series grid (scripts/analysis/kseries.py)
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    import kseries
    base = Path(os.environ["DATA_p330"], "runs", "npv2")
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((base / kseries.method(base, 3000)).glob("*-s0-*/checkpoint.pt"))[0]
    torch.backends.cuda.matmul.allow_tf32 = False
    power_checks(); svd_timing(); ns_checks(); tf32_check(); checkpoint_checks(ckpt)
    OUT["gpu"] = torch.cuda.get_device_name(0)
    dst = ROOT / "runs-record" / "probes" / "paper_checks.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(OUT, open(dst, "w"), indent=1)
    print(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
