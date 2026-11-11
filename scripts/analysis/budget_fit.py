"""How many fewer steps Natural Pion needs, from runs that each have their own full schedule.

Every point is a finished run whose cosine schedule ends at its budget B, so no point is read off the middle of a
schedule. For each method the final validation loss is fitted as L(B) = a + b B^-c (least squares over the seed
means; for fixed c the fit is linear in a, b, so c is found by a fine grid). The saving at a loss L is
B_Pion(L) / B_NP(L). It is reported at the losses Pion reaches at its own budgets, with B_Pion taken as the
measured budget and B_NP from the inverse of Natural Pion's fit (interpolation where L lies inside Natural Pion's
measured range).

Uncertainty: a bootstrap that resamples seeds where there are four, and adds N(0, sd^2) to single-seed points with
sd the median seed standard deviation; and a second model, piecewise-linear interpolation in (log B, L), to show
how much the functional form matters.

    python scripts/analysis/budget_fit.py        # writes runs-record/probes/budget_fit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
R = ROOT / "runs-record"


def val_of(pattern: str, seed: int) -> float:
    for d in sorted(R.glob(pattern)):
        m = json.load(open(d / "manifest.json"))
        if m["config"]["seed"] != seed:
            continue
        v = [json.loads(l) for l in open(d / "log.jsonl") if '"val_loss"' in l]
        if v and v[-1]["step"] + 1 >= m["config"]["train_steps"]:
            return v[-1]["val_loss"]
    raise SystemExit(f"{pattern} seed {seed}: no finished run")


L = "stepnormlong"
# (budget, [(pattern, seed), ...]) -- the configurations of the paper (Pion lr 1e-3; Natural Pion Fast, angle per horizon)
PION = [
    (3000, [("adamw2d/pion-lr0.001-s0-9e6e6d32f6", 0)] + [("seedshead/pion-lr0.001-*", s) for s in (1, 2, 3)]),
    (3840, [("stepnormfinal/pion3840/*", 0)] + [(f"{L}/pion3840-lr1e-3/*", s) for s in (1, 2, 3)]),
    (4500, [("stepnormfinal/pion4500/*", 0)]),
    (5300, [("stepnormclock/pion-lr0.001-s0-dbe147d5df", 0)]),
    (15000, [("ladder15k/pion-lr0.001-s0-f67fdad3e9", 0)]),
    (19200, [("stepnormfinal/pion19200/*", 0)] + [(f"{L}/pion19200-lr1e-3/*", s) for s in (1, 2, 3)]),
    (30000, [(f"{L}/pion30000-lr1e-3/*", 0)]),
    (45000, [(f"{L}/pion45000-lr1e-3/*", 0)]),
    (73242, [(f"{L}/pion73242-lr1e-3/*", 0)]),
]
# Pion's lr per budget: the better (seed 0) of 1e-3 and 2e-3 (scripts/analysis/kseries.py pion_best); at 1e-3 a budget
# keeps all its seeds above, at 2e-3 it is that one run
import kseries  # noqa: E402


def _pion_best(spec):
    out = []
    for B, runs in spec:
        b = kseries.pion_best(R, B)
        if b is None:
            out.append((B, [(f"pion-lrdepth/pion{B}-lr2e-3/*", 0)]))      # unfinished: val_of says so
        elif b[0] == 1e-3:
            out.append((B, runs))
        else:                                                              # every finished seed of lr 2e-3
            out.append((B, [(kseries.PION_2E3[B][0], s) for s in sorted(b[1])]))
    return out


PION = _pion_best(PION)
# Natural Pion Fast with the query projection per head (scripts/paper/v2.sbatch, rows 99-110). The earlier Q-whole points
# (stepnormcombo, stepnormfinal, stepnormlong/ours*) are in git history (budget_fit.json before 2026-11-05 evening).
V = "npv2"
# the method: Natural Pion Fast with Pion's RMS length rule; where the scale was chosen on a grid at that budget, the
# grid's best point (seed 0) stands for the budget
# (the final form with the floor 1e-3, tags g*; seed 0 at 3000 / 15000 are the matrix runs of that configuration)
# the k-series (decay depth 1% of the peak); each budget's method run from scripts/analysis/kseries.py
import kseries  # noqa: E402
_K = R / V


def _m(steps, seed=0):
    return f"{V}/{kseries.method(_K, steps, seed) or 'k-unfinished'}/*"


NP = [
    (3000, [(_m(3000, s), s) for s in range(4)]),
    (9000, [(_m(9000), 0)]),
    (15000, [(_m(15000, s), s) for s in range(4)]),
    (30000, [(_m(30000), 0)]),
    (45000, [(_m(45000), 0)]),
    (73242, [(_m(73242), 0)]),
]
# equal wall-clock time is read off the Natural Pion curve: at Pion's B, Natural Pion has B * sps_pion / sps_np steps
SPS_PION = ("timing/pion-a/*", "timing/pion-b/*")      # one job, one GPU (scripts/paper/timing.sbatch)
SPS_NP = ("timing/final/*",)                            # the method's form; "timing/fast/*" if TF32 is not adopted
EQUAL_TIME = (3840, 19200, 73242)


def sps_of(*patterns: str) -> float:
    """Mean over the patterns of (wall at step 599 - wall at step 100) / 499, the timing job's measure."""
    out = []
    for pattern in patterns:
        for d in sorted(R.glob(pattern)):
            w = {r["step"]: r["wall"] for r in (json.loads(l) for l in open(d / "log.jsonl") if '"wall"' in l)}
            if 100 in w and 599 in w:
                out.append((w[599] - w[100]) / 499)
                break
        else:
            raise SystemExit(f"{pattern}: no timing run")
    return sum(out) / len(out)


def load(spec):
    out = []
    for B, *rest in spec:
        if rest[0] == "grid":
            out.append((B, np.array([min(val_of(p, 0) for p in rest[1])])))
        else:
            out.append((B, np.array([val_of(p, s) for p, s in rest[0]])))
    return out


def fit(B, y):
    """L = a + b B^-c by grid over c and linear least squares in (a, b)."""
    best = None
    for c in np.linspace(0.05, 2.0, 1951):
        X = np.stack([np.ones_like(B), B ** -c], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, coef[0], coef[1], c)
    return best[1:]


def inv_pow(a, b, c, Lt):
    return (b / (Lt - a)) ** (1 / c) if Lt > a else np.inf


def inv_interp(B, y, Lt):
    """Piecewise-linear in (log B, L); y decreasing in B."""
    lb = np.log(B)
    for i in range(len(B) - 1):
        if y[i] >= Lt >= y[i + 1]:
            t = (y[i] - Lt) / (y[i] - y[i + 1])
            return float(np.exp(lb[i] + t * (lb[i + 1] - lb[i])))
    return np.nan


def main():
    dst = R / "probes" / "budget_fit.json"
    try:
        pion, npn = load(PION), load(NP)
        sps_p, sps_n = sps_of(*SPS_PION), sps_of(*SPS_NP)
    except SystemExit as e:                         # a point is not finished: say so, do not keep stale numbers
        dst.write_text(json.dumps({"complete": False, "missing": str(e)}, indent=1))
        print("incomplete:", e)
        return
    sd = float(np.median([v.std(ddof=1) for _, v in pion + npn if len(v) > 1]))
    Bp = np.array([B for B, _ in pion], float); yp = np.array([v.mean() for _, v in pion])
    Bn = np.array([B for B, _ in npn], float); yn = np.array([v.mean() for _, v in npn])
    fp, fn = fit(Bp, yp), fit(Bn, yn)
    out = {"seed_sd": sd, "pion": {"B": Bp.tolist(), "L": yp.tolist(), "fit_abc": list(map(float, fp))},
           "natural_pion": {"B": Bn.tolist(), "L": yn.tolist(), "fit_abc": list(map(float, fn))},
           "residual_max_pion": float(np.abs(fp[0] + fp[1] * Bp ** -fp[2] - yp).max()),
           "residual_max_np": float(np.abs(fn[0] + fn[1] * Bn ** -fn[2] - yn).max()), "targets": {}}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        yb = np.array([rng.choice(v, len(v)).mean() if len(v) > 1 else v[0] + rng.normal(0, sd) for _, v in npn])
        boots.append(fit(Bn, yb))
    # the same saving against Pion at the best lr of Table 9 (one seed; scripts/paper/pion_lr_depth.sbatch, depth 1%)
    out["best_lr"] = {}
    for B, pats in ((19200, [("stepnormfinal/pion19200/*", "1e-3"), ("pion-lrdepth/pion19200-lr2e-3/*", "2e-3"),
                             ("pion-lrdepth/pion19200-lr4e-3/*", "4e-3")]),):
        try:
            Ls = {lr: val_of(p, 0) for p, lr in pats}
        except SystemExit:
            continue
        lr = min(Ls, key=Ls.get)
        out["best_lr"][str(B)] = {"lr": float(lr), "L": Ls[lr], "ratio_powerlaw": float(B / inv_pow(*fn, Ls[lr])),
                                  "ratio_interp": float(B / inv_interp(Bn, yn, Ls[lr]))}
    for B, v in pion:
        Lt = float(v.mean())
        r_pow = B / inv_pow(*fn, Lt)
        r_int = B / inv_interp(Bn, yn, Lt)
        rb = np.array([B / inv_pow(*f, Lt) for f in boots])
        rb = rb[np.isfinite(rb)]
        inside = yn.min() <= Lt <= yn.max()
        out["targets"][str(B)] = {"L": Lt, "ratio_powerlaw": float(r_pow), "ratio_interp": float(r_int),
                                  "ratio_p05": float(np.percentile(rb, 5)), "ratio_p95": float(np.percentile(rb, 95)),
                                  "interpolated": bool(inside)}
        print(f"Pion {B:>6} steps  L={Lt:.4f}  saving {r_pow:.2f} (power law; 90% {np.percentile(rb,5):.2f}-"
              f"{np.percentile(rb,95):.2f})  {r_int:.2f} (interp)  {'inside' if inside else 'OUTSIDE'} NP range")
    # equal wall-clock time from the curve
    out["sps_pion"], out["sps_np"], out["equal_time"] = sps_p, sps_n, {}
    for B in EQUAL_TIME:
        Lp = float(dict(pion)[B].mean())
        Beq = B * sps_p / sps_n
        Lpow = float(fn[0] + fn[1] * Beq ** -fn[2])
        lb, i = np.log(Bn), np.searchsorted(Bn, Beq)
        Lint = float(np.interp(np.log(Beq), lb, yn)) if Bn[0] <= Beq <= Bn[-1] else float("nan")
        out["equal_time"][str(B)] = {"L_pion": Lp, "B_np": Beq, "L_np_powerlaw": Lpow, "L_np_interp": Lint,
                                     "gap_powerlaw": Lp - Lpow, "gap_interp": Lp - Lint}
        print(f"equal time with Pion {B}: Natural Pion {Beq:,.0f} steps, L {Lpow:.4f} (fit) / {Lint:.4f} (interp), "
              f"gap {Lp - Lpow:.3f} / {Lp - Lint:.3f}")
    out["complete"] = True
    print("fits a,b,c:", [round(float(x), 4) for x in fp], [round(float(x), 4) for x in fn],
          "max residual", round(out["residual_max_pion"], 4), round(out["residual_max_np"], 4), "seed sd", round(sd, 4))
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
