r"""Every measured number in the paper (docs/paper/latex/main.tex), computed from runs-record/ (or a runnable check).

    python scripts/analysis/paper_numbers.py            # writes docs/paper/numbers.json and docs/paper/numbers.md
    python scripts/analysis/render_tex.py               # writes docs/paper/latex/numbers.tex

The paper refers to numbers only as \pn{key}{format}. Each key below says where its value comes from. A key
whose source is not yet in the repository is listed as MISSING with what is needed, and renders as a
visible gap rather than a number.

Conventions:
* `val` is the final validation loss of a run (last evaluation of the last attempt in its log.jsonl).
* "best of a grid" is the lowest final val over the runs matching a pattern (one seed, different angles).
* seconds per step = final wall time / steps, so it includes the run's periodic evaluations. Runs compared
  for speed in the paper share the same evaluation cadence.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs-record"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kseries  # noqa: E402  (the k-series: which run is the method at each budget)
KBASE = R / "npv2"
NUM, NOTE = {}, {}


# ---------------------------------------------------------------------------------------------- reading
def _rows(d: Path) -> list[dict]:
    rows = []
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        rows.append(r)
    return rows


def runs(pattern: str, seed: int | None = None, **cfg_eq) -> list[Path]:
    out = []
    for m in sorted(R.glob(f"{pattern}/manifest.json")):
        cfg = json.load(open(m))["config"]
        if seed is not None and cfg["seed"] != seed:
            continue
        if any(cfg.get(k) != v for k, v in cfg_eq.items()):
            continue
        out.append(m.parent)
    return out


def one(pattern: str, seed: int | None = None, **cfg_eq) -> Path:
    hits = runs(pattern, seed, **cfg_eq)
    if len(hits) != 1:
        raise SystemExit(f"{pattern} (seed {seed}, {cfg_eq}): {len(hits)} runs, expected one: {hits}")
    return hits[0]


def info(d: Path) -> dict:
    cfg = json.load(open(d / "manifest.json"))["config"]
    rows = _rows(d)
    vals = [r for r in rows if "val_loss" in r]
    walls = [r["wall"] for r in rows if "wall" in r]
    done = bool(vals) and vals[-1]["step"] >= cfg["train_steps"] - 1
    return {"val": vals[-1]["val_loss"] if vals else None, "steps": cfg["train_steps"], "lr": cfg["lr"],
            "wall": walls[-1] if walls else None, "done": done, "vals": {r["step"]: r["val_loss"] for r in vals},
            "rows": rows, "dir": d}


def val_at(d: Path, step: int) -> float:
    """Validation loss at `step`, from whichever attempt in the log recorded it (latest wins)."""
    found = None
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step") == step and "val_loss" in r:
            found = r["val_loss"]
    if found is None:
        raise SystemExit(f"{d}: no val at step {step}")
    return found


def val(pattern, seed=None, **kw) -> float:
    i = info(one(pattern, seed, **kw))
    if not i["done"]:
        raise SystemExit(f"{pattern}: not finished")
    return i["val"]


def sps(pattern, seed=None, **kw) -> float:
    """Wall seconds per step: last logged wall over last logged step (+1), so runs cut short count right."""
    rows = [r for r in _rows(one(pattern, seed, **kw)) if "wall" in r]
    return rows[-1]["wall"] / (rows[-1]["step"] + 1)


def sps_median(d: Path) -> float:
    """Seconds per step between consecutive logged rows, over every attempt (robust to resumes): the mean without the
    fastest and slowest tenth (evaluation pauses). Walls are logged to 0.1 s every 23 steps, so a single interval
    resolves only ~0.004 s/step; the mean over ~130 intervals resolves much finer than a median would."""
    rates, prev = [], None
    for line in open(d / "log.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            prev = None
            continue
        if "wall" in r:
            if prev and r["step"] > prev["step"] and r["wall"] > prev["wall"]:
                rates.append((r["wall"] - prev["wall"]) / (r["step"] - prev["step"]))
            prev = r
    rates.sort()
    k = len(rates) // 10
    core = rates[k:len(rates) - k] or rates
    return sum(core) / len(core)


def best(pattern, at: int | None = None, **kw) -> tuple[float, float, bool]:
    """(best final val, its lr, interior?) over a finished grid; with `at`, val at that step instead
    (for runs stopped early with --max-steps)."""
    infos = [info(d) for d in runs(pattern, **kw)]
    if at is not None:
        for i in infos:
            i["val"], i["done"] = val_at(i["dir"], at), True
    infos = [i for i in infos if i["done"]]
    if not infos:
        raise SystemExit(f"{pattern}: no finished runs")
    infos.sort(key=lambda i: i["lr"])
    k = min(range(len(infos)), key=lambda j: infos[j]["val"])
    return infos[k]["val"], infos[k]["lr"], 0 < k < len(infos) - 1


def put(key: str, value, source: str):
    NUM[key] = value
    NOTE[key] = source


def missing(key: str, need: str):
    NUM[key] = None
    NOTE[key] = "MISSING: " + need

L = "stepnormlong"

# ------------------------------------------------------------------------------------------- Pion (the baseline)
pf = info(one(f"{L}/pion73242-lr1e-3/*"))
put("pion_full", pf["val"], f"{L}/pion73242-lr1e-3")
put("pion_full_wall", pf["wall"], f"{L}/pion73242-lr1e-3, wall")
put("pion_sps", pf["wall"] / pf["steps"], "pion_full wall / steps")

# Pion lr grid
for steps, lrs in ((3840, ("5e-4", "1e-3", "2e-3", "4e-3")), (19200, ("1e-3", "2e-3", "4e-3"))):
    for lr in lrs:
        # lr 1e-3 decays to 1% (1e-5); the other rows are repeated at the same depth (scripts/paper/pion_lr_depth.sbatch)
        pat = f"stepnormfinal/pion{steps}/*" if lr == "1e-3" else f"pion-lrdepth/pion{steps}-lr{lr}/*"
        _pv = val(pat, seed=0) if runs(pat, 0) else None
        put(f"pion_lr_{steps}_{lr}", _pv, pat) if _pv is not None else missing(f"pion_lr_{steps}_{lr}", pat)

for _steps, _lrs in ((3840, ("5e-4", "1e-3", "2e-3", "4e-3")), (19200, ("1e-3", "2e-3", "4e-3"))):
    _vals = {lr: NUM.get(f"pion_lr_{_steps}_{lr}") for lr in _lrs}
    if all(v is not None for v in _vals.values()):
        _b = min(_vals, key=_vals.get)
        put(f"pion_lr_best_{_steps}", float(_b), f"Table 11 at {_steps}: best of {_lrs}")
        put(f"pion_lr_gain_{_steps}", _vals["1e-3"] - _vals[_b], f"Table 11 at {_steps}: val(1e-3) - val(best)")
    else:
        missing(f"pion_lr_best_{_steps}", "Table 11 runs"); missing(f"pion_lr_gain_{_steps}", "Table 11 runs")
if NUM.get("pion_lr_3840_2e-3") is not None:
    put("pion_lr_3840_better", NUM["pion_lr_3840_1e-3"] - NUM["pion_lr_3840_2e-3"], "Pion 3840: lr 1e-3 - lr 2e-3")
else:
    missing("pion_lr_3840_better", "pion-lrdepth/pion3840-lr2e-3")

p3000 = [val("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [val("seedshead/pion-lr0.001-*", s) for s in (1, 2, 3)]
put("pion3000_s0", p3000[0], "adamw2d/pion-lr0.001-s0-9e6e6d32f6 (Pion, 3000 steps, AdamW 8e-3)")
put("pion3000_mean", sum(p3000) / 4, "Pion 3000, seeds 0-3 (adamw2d, seedshead)")
put("pion15k", val("ladder15k/pion-lr0.001-s0-f67fdad3e9"), "ladder15k/pion-lr0.001-s0-f67fdad3e9")
# Pion at its better lr per budget (1e-3 or 2e-3, seed 0; kseries.pion_best), for Table 2 and the gaps
for _B, _k in ((15000, "pion15k_best"), (73242, "pion_full_best")):
    _pb = kseries.pion_best(R, _B)
    if _pb is None:
        missing(_k, f"pion-lrdepth/pion{_B}-lr2e-3"); missing(_k + "_lr", f"pion-lrdepth/pion{_B}-lr2e-3")
    else:
        put(_k, _pb[1][0], f"seed 0 at lr {_pb[0]:g}, the better of 1e-3 / 2e-3 on their common seeds"); put(_k + "_lr", _pb[0], f"lr of {_k}")


def derive(key: str, fn, deps: tuple, note: str):
    vals = [NUM.get(d) for d in deps]
    if any(v is None for v in vals):
        missing(key, "needs " + ", ".join(d for d, v in zip(deps, vals) if v is None))
        return None
    put(key, fn(*vals), note)
    return NUM[key]


def peak(d: Path) -> float:
    return max(r.get("peak_gb") or 0.0 for r in _rows(d))
put("peak_pion", peak(one("adamw2d/pion-lr0.001-s0-9e6e6d32f6")), "max peak_gb in the Pion 3000-step log (micro-batch 256)")

_tr = [r for r in _rows(one(f"{L}/pion73242-lr1e-3/*")) if "train_loss" in r]
_tr = [r["train_loss"] for r in _tr if r["step"] > _tr[-1]["step"] - 2000]
put("pion_full_train", sum(_tr) / len(_tr), f"{L}/pion73242-lr1e-3: mean logged training loss over the last 2000 steps")
import statistics as _st0
put("pion_train_batch_sd", _st0.pstdev(_tr), f"{L}/pion73242-lr1e-3: sd of the logged per-batch training loss, last 2000 steps")
_mu = info(one("muon/muon73242/*"))                      # Muon in the Pion authors' configuration (runs-record/muon)
put("muon_full", _mu["val"], "muon/muon73242 (Natural-Muon project's run 397404; same data, schedule and AdamW)")
(put("muon_gap_full", NUM["pion_full_best"] - _mu["val"], "pion_full_best - muon_full")
 if NUM.get("pion_full_best") is not None else missing("muon_gap_full", "pion_full_best"))
# the rotated share of the parameters, counted on the model itself
import torch as _torch  # noqa: E402
from harness.model import ModelConfig as _MC, Transformer as _TF  # noqa: E402
with _torch.device("meta"):
    _m = _TF(_MC())
_rot = sum(p.numel() for n, p in _m.named_parameters() if p.ndim == 2 and n.startswith("blocks."))
_all = sum(p.numel() for p in _m.parameters())
put("rotated_share", _rot / _all, f"2-D weights inside the blocks / all parameters ({_rot:,} / {_all:,})")
put("pion_vs_paper", NUM["pion_full_train"] - 3.3575,
    "pion_full_train - 3.3575 (the Pion paper's final training loss, Fig. 5, cited)")

# analytic: T^T T = I + Omega^4/4 (Proposition 4), so one step moves a singular value by about (angle)^4 / 8
for th in (0.05, 0.2):
    put(f"trunc_err_{th}", th ** 4 / 8, f"theta^4/8 at theta = {th} (Proposition 4: sqrt(1 + theta^4/4) - 1)")
rc = json.load(open(R / "probes" / "rl_refactor_cost_b200.json"))
put("rl_refactor_b200", rc["refactor_total_s"], "runs-record/probes/rl_refactor_cost_b200.json (scripts/probes/rl_refactor_cost.py, job 397885)")
put("rl_refactor_b200_per_step", rc["refactor_total_s"] / 25, "rl_refactor_b200 / 25: one refresh amortised over T_fac = 25 steps")
PA = R / "probes" / "pion_angles.json"
if PA.exists():
    pa = json.load(open(PA))          # {step: [angle of every rotated block]}, from scripts/probes/pion_angles.py
    per_step = [max(v) for k, v in pa.items() if int(k) > 0]
    put("pion_angle_max", max(per_step), "runs-record/probes/pion_angles.json: largest block angle, first 1058 steps of the 3840-step run (the probe hit its time limit there; the learning rate is still within 15% of its peak)")
    put("pion_angle_steps", max(int(k) for k in pa), "last logged step of the Pion angle probe")
    meds, sprs = [], []
    for k, v in pa.items():
        if int(k) == 0:
            continue
        xs = sorted(v)
        meds.append(xs[len(xs) // 2])
        sprs.append(xs[int(0.9 * (len(xs) - 1))] / xs[int(0.1 * (len(xs) - 1))])
    meds.sort(); sprs.sort()
    put("trunc_err_pion", meds[len(meds) // 2] ** 4 / 8, "(pion_angle_block_median)^4 / 8")
    put("pion_angle_block_median", meds[len(meds) // 2], "pion_angles.json: median over steps of the median block angle")
    put("pion_angle_spread", sprs[len(sprs) // 2], "pion_angles.json: median over steps of p90/p10 of the block angles")
    per_step.sort()
    put("pion_angle_median", per_step[len(per_step) // 2], "same, median over logged steps of the largest block angle")
else:
    missing("pion_angle_max", "run scripts/probes/pion_angles.sbatch and archive its output as runs-record/probes/pion_angles.json")

# probes and checks that are not in runs-record
try:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/check_fisher.py")], capture_output=True, text=True, check=True).stdout
    vals = dict(line.split(None, 2)[:2] if False else (line.rsplit(" ", 1)[0].strip(), line.rsplit(" ", 1)[1]) for line in out.splitlines() if line)
    put("mc_err_in", float(out.split("in  rel err")[1].split()[0]), "scripts/analysis/check_fisher.py")
    put("mc_err_out", float(out.split("out rel err")[1].split()[0]), "scripts/analysis/check_fisher.py")
    put("pullback_err", float(out.split("pullback")[1].split()[0]), "scripts/analysis/check_fisher.py")
except Exception as e:  # numpy missing, etc.
    for k in ("mc_err_in", "mc_err_out", "pullback_err"):
        missing(k, f"scripts/analysis/check_fisher.py failed: {e}")

# per-step change of ||W||_F^2 in training, first 200 steps (scripts/probes/ns_diag.py)
for ret in ("cayley", "ns", "trunc", "ns_delta"):
    f = R / "probes" / f"ns_diag_{ret}.json"
    if f.exists():
        d = json.load(open(f))
        ms = sorted(x["mean_signed"] for x in d["steps"])
        put(f"diag_{ret}_step", ms[len(ms) // 2], f"{f.name}: median over steps of the mean over matrices")
        put(f"diag_{ret}_neg", sum(m < 0 for m in ms), f"{f.name}: steps where ||W||_F^2 fell")
        put(f"diag_{ret}_sum", sum(ms), f"{f.name}: summed over the 200 steps")
        if ret == "ns" and d["sigma_ratio"]:
            put("diag_sigma_ratio", sorted(r[1] for r in d["sigma_ratio"])[len(ms) // 2],
                f"{f.name}: median sigma_est / ||X||_2")
            put("diag_precision", d["summary"]["precision_seen"] == ['[false, false, "highest"]'],
                f"{f.name}: autocast off, tf32 off, highest fp32 precision at every step")
    else:
        for k in ("step", "neg", "sum"):
            missing(f"diag_{ret}_{k}", f"RETS={ret} sbatch scripts/probes/ns_diag.sbatch")

f = R / "probes" / "retraction_bias.json"
if f.exists():
    d = json.load(open(f))["n1376_th0.05"]
    for tag in ("solve", "ns", "ns_delta"):
        put(f"rb_{tag}_mean", d[tag]["mean"], f"{f.name}: 1376 x 1376, angle 0.05, mean tr(R^T R - I)/n")
        put(f"rb_{tag}_z", d[tag]["z"], f"{f.name}: same, mean over its standard error")
else:
    missing("rb_solve_mean", "python scripts/probes/retraction_bias.py")

# optimizer step alone, by part (scripts/probes/optimizer_step_bench.py, synchronised timers)
f = R / "probes" / "optimizer_step_bench.json"
_bench = json.load(open(f)) if f.exists() else {}
# the final-form benchmark at the method's own scale (older files measured the angle rule; one ran at lr 4.0)
if "final_parts" in _bench and _bench.get("np_lr") == float(kseries.scale(KBASE, 3000) or "nan"):
    d = json.load(open(f))
    put("bench_pion", d["pion"], f.name + ": Pion, whole optimizer step")
    for tag, key in (("old", "after44_parts"), ("new", "final_parts")):
        for part, v in d[key].items():
            put(f"bench_{tag}_" + part.replace("^-1", "inv").replace("+", "_").replace(" ", "_").replace("(", "").replace(")", ""),
                v, f"{f.name}: {key} {part}")
    for tag in ("old", "new"):
        put(f"bench_{tag}_shared", sum(NUM[f"bench_{tag}_{k}"] for k in ("generators_momentum", "apply_W")),
            "generators+momentum + apply W")
else:
    for k in ['bench_new_Finv', 'bench_new_cayley', 'bench_new_refresh', 'bench_new_shared', 'bench_new_sigma', 'bench_new_statistics', 'bench_new_total_synchronised', 'bench_old_Finv', 'bench_old_cayley', 'bench_old_refresh', 'bench_old_shared', 'bench_old_sigma', 'bench_old_statistics', 'bench_old_total_synchronised', 'bench_pion']:
        missing(k, "scripts/probes/optimizer_step_bench.sbatch (final form)")

# FLOP per step (scripts/analysis/flops.py)
f = R / "probes" / "flops.json"
if f.exists():
    d = json.load(open(f))
    for size in ("60M", "1.3B"):
        tag = size.replace(".", "_")
        put(f"flops_{tag}_pion", d[size]["pion_overhead"], f"{f.name}: Pion optimizer / model FLOP, {size}")
        put(f"flops_{tag}_np", d[size]["np_overhead"], f"{f.name}: Natural Pion Fast optimizer / model FLOP, {size}")
        put(f"flops_{tag}_ratio", d[size]["np_over_pion"], f"{f.name}: step FLOP, Natural Pion Fast / Pion, {size}")
        put(f"flops_{tag}_shared", (d[size]["natural_pion_fast"]["generators"] + d[size]["natural_pion_fast"]["rotation"])
            / d[size]["model"], f"{f.name}: generators + rotation / model FLOP, {size}")
        for part, v in d[size]["natural_pion_fast"].items():
            put(f"flops_{tag}_part_" + part.split(" ")[0].replace("^-1", "inv"), v / d[size]["model"],
                f"{f.name}: {part} / model FLOP, {size}")
else:
    missing("flops_60M_np", "python scripts/analysis/flops.py")

# ---------------------------------------------------------------- Section 4 (v2.sbatch rows 99-110)

# Section 4 with the query projection per head (scripts/paper/v2.sbatch, rows 99-110)
def _v(pattern, seed=None):
    try:
        return val(pattern, seed)
    except SystemExit:
        return None
# the method: the final form with the floor 1e-3 (v2.sbatch tags g*). Seed 0 at 3000 and 15000 are the matrix runs of
# exactly that configuration (f-reg-eps1e-3, m-tf25-e3-15k).
# Since 2026-11-09 every run is the k-series (decay depth 1% of the peak; scripts/analysis/kseries.py): a tag t here
# reads runs-record/npv2/k-t; the method at 3000 / 15000 steps, seed 0, is the best point of its scale grid.
import kseries
KBASE = R / "npv2"


def tagpat(tag: str, seed: int = 0) -> str:
    if seed == 0 and tag in ("g3k", "g15k"):
        t = kseries.method(KBASE, 3000 if tag == "g3k" else 15000, 0)
        return f"npv2/{t or kseries.tag(tag) + '-grid-unfinished'}/*"
    return f"npv2/{kseries.tag(tag)}/*"


MAIN = {3000: "g3k", 15000: "g15k"}
# the method's realised rotation angles (3000 steps, seed 0; angle_p50 / angle_p90: quantiles over the rotated matrices
# of the largest plane's angle, logged every 23 steps) and one truncated-exponential step's change of a singular value
# there, sqrt(1 + a^4/4) - 1 (Proposition 4)
_mt = kseries.method(KBASE, 3000)
_ar = [r for r in _rows(next((KBASE / _mt).glob("*-s0-*"))) if "angle_p90" in r] if _mt else []
if _ar:
    import statistics as _stat
    for _k, _ang in (("ang_p50_med", _stat.median(r["angle_p50"] for r in _ar)),
                   ("ang_p90_med", _stat.median(r["angle_p90"] for r in _ar)),
                   ("ang_p90_max", max(r["angle_p90"] for r in _ar))):
        put(_k, _ang, f"{_mt}: over the logged steps")
        put("trunc_at_" + _k, math.sqrt(1 + _ang ** 4 / 4) - 1, f"sqrt(1 + {_k}^4/4) - 1")
    # Figure 1: the method's median angle over Pion's, at the nearest step Pion's probe logged (both log every 23
    # steps), over the steps both cover
    if PA.exists():
        _pm = {int(k): _stat.median(v) for k, v in pa.items() if int(k) > 0}
        _rr = sorted(r["angle_p50"] / _pm[min(_pm, key=lambda s: abs(s - r["step"]))]
                     for r in _ar if 0 < r["step"] <= max(_pm))
        put("ang_ratio_min", _rr[0], f"{_mt} angle_p50 / Pion's median block angle, steps 1-{max(_pm)}: smallest")
        put("ang_ratio_med", _stat.median(_rr), "same, median over the steps")
        put("ang_ratio_max", _rr[-1], "same, largest")
    else:
        for _k in ("ang_ratio_min", "ang_ratio_med", "ang_ratio_max"):
            missing(_k, "pion_angles.json")
else:
    for _k in ("ang_p50_med", "ang_p90_med", "ang_p90_max"):
        missing(_k, "k-m3k grid"); missing("trunc_at_" + _k, "k-m3k grid")
for _st in (3000, 9000, 15000, 30000):                    # the method's peak scale per budget (its grid's best)
    _t = kseries.scale(KBASE, _st)
    (put(f"scale_{_st}", float(_t), f"best of the k-series grid at {_st}: {_t}") if _t
     else missing(f"scale_{_st}", f"k-series grid at {_st} steps"))
FULL = f"npv2/{kseries.method(KBASE, 73242) or 'k-g73242-grid-unfinished'}/*"
for st in (3000, 15000):
    vs = [_v(tagpat(MAIN[st], sd), sd) for sd in range(4)]
    for sd, v in enumerate(vs):
        (put(f"v2_{st}_s{sd}", v, f"{MAIN[st]} seed {sd}") if v is not None
         else missing(f"v2_{st}_s{sd}", f"v2.sbatch, {st} steps, seed {sd}"))
    if all(v is not None for v in vs):
        put(f"v2_{st}_mean", sum(vs) / 4, f"{MAIN[st]}, seeds 0-3")
        put(f"v2_{st}_mean13", sum(vs[1:]) / 3, f"{MAIN[st]}, seeds 1-3 (seed 0 chose the scale)")
    else:
        missing(f"v2_{st}_mean13", "all four seeds")
        missing(f"v2_{st}_mean", "all four seeds")
p3 = [val("adamw2d/pion-lr0.001-s0-9e6e6d32f6")] + [val("seedshead/pion-lr0.001-*", sd) for sd in (1, 2, 3)]
if NUM.get("v2_3000_mean") is not None:
    put("v2_gap3000", NUM["pion3000_mean"] - NUM["v2_3000_mean"], "pion3000_mean - v2_3000_mean")
    for sd in range(4):
        put(f"v2_gap3000_s{sd}", p3[sd] - NUM[f"v2_3000_s{sd}"], f"Pion 3000 seed {sd} - v2 seed {sd}")
    put("v2_gap3000_min", min(NUM[f"v2_gap3000_s{sd}"] for sd in range(4)), "smallest per-seed gap at 3000")
else:
    missing("v2_gap3000", "v2_3000_mean"); missing("v2_gap3000_min", "v2_3000_mean")
if NUM.get("v2_15000_s0") is not None:
    (put("v2_gap15k", NUM["pion15k_best"] - NUM["v2_15000_s0"], "pion15k_best - v2_15000_s0 (seed 0)")
     if NUM.get("pion15k_best") is not None else missing("v2_gap15k", "pion15k_best"))
else:
    missing("v2_gap15k", "v2_15000_s0")
vf = [d for d in runs(FULL) if info(d)["done"]]
if vf:
    i = info(vf[-1])
    put("v2_full", i["val"], FULL)
    (put("v2_gap_full", NUM["pion_full_best"] - i["val"], "pion_full_best - v2_full")
     if NUM.get("pion_full_best") is not None else missing("v2_gap_full", "pion_full_best"))
else:
    for k in ("v2_full", "v2_gap_full"):
        missing(k, "v2.sbatch row rms73242 (73 242 steps)")

# step times: one job on one GPU, every configuration in turn (scripts/paper/timing.sbatch)
TIMING = {"pion": ("pion-a", "pion-b"), "alg1": ("alg1",), "tf25": ("tf25",), "windows": ("windows",),
          "batched": ("batched",), "fast": ("fast",), "final": ("final",)}
METHOD_FORM = "final"                 # "final" (TF32 before the Cayley step, bf16 F^-1) once it passes the 0.010 rule


def timing_sps(tag: str) -> float | None:
    for d in sorted(R.glob(f"timing/{tag}/*")):
        w = {r["step"]: r["wall"] for r in _rows(d) if "wall" in r}
        if 100 in w and 599 in w:
            return (w[599] - w[100]) / 499
    return None


for k, tags in TIMING.items():
    xs = [timing_sps(t) for t in tags]
    if all(x is not None for x in xs):
        put(f"time_{k}", sum(xs) / len(xs), "runs-record/timing/" + "+".join(tags) + ": (wall at 599 - wall at 100) / 499")
    else:
        missing(f"time_{k}", "scripts/paper/timing.sbatch")
if NUM.get("time_pion") is not None and len(TIMING["pion"]) == 2:
    a, b = timing_sps("pion-a"), timing_sps("pion-b")
    put("time_pion_spread", abs(a - b) / NUM["time_pion"], "|pion-a - pion-b| / their mean: how steady the node was")
    if NUM["time_pion_spread"] > 0.02:                  # the node was not steady over the job: no step time is quoted
        for _k in [k for k in NUM if k.startswith("time_") and k != "time_pion_spread"]:
            missing(_k, f"scripts/paper/timing.sbatch again: Pion first and last differ by {NUM['time_pion_spread']:.1%}")
derive("v2_sps", lambda x: x, (f"time_{METHOD_FORM}",), f"= time_{METHOD_FORM} (the method)")
derive("pion3000_sps", lambda x: x, ("time_pion",), "= time_pion")
derive("v2_ratio", lambda a, b: a / b, ("v2_sps", "pion3000_sps"), "v2_sps / pion3000_sps")

# steps saving from runs with their own full schedules (scripts/analysis/budget_fit.py)
f = R / "probes" / "budget_fit.json"
if f.exists():
    d = json.load(open(f))
if f.exists() and d.get("complete"):
    inside = {B: t for B, t in d["targets"].items() if t["interpolated"]}
    for B, t in inside.items():
        put(f"budget_{B}_L", t["L"], f"{f.name}: Pion val at {B} steps")
        put(f"budget_{B}_pow", t["ratio_powerlaw"], f"{f.name}: B_Pion / B_NP(L), power law")
        put(f"budget_{B}_int", t["ratio_interp"], f"{f.name}: same, interpolation in log B")
    for B, t in d.get("best_lr", {}).items():
        put(f"budget_best_{B}_pow", t["ratio_powerlaw"], f"{f.name}: saving against Pion at its best Table 11 lr ({t['lr']:g})")
        put(f"budget_best_{B}_int", t["ratio_interp"], f"{f.name}: same, interpolation")
    allr = [t["ratio_powerlaw"] for t in inside.values()]
    put("budget_lo", min(allr), f"{f.name}: smallest saving of the power-law fit over the targets of Table 4")
    put("budget_hi", max(allr), f"{f.name}: largest")
    put("budget_resid", max(d["residual_max_pion"], d["residual_max_np"]), f"{f.name}: worst fit residual")
    put("budget_n_pion", len(d["pion"]["B"]), f"{f.name}: Pion budgets")
    put("budget_n_np", len(d["natural_pion"]["B"]), f"{f.name}: Natural Pion budgets")
    for B, e in d["equal_time"].items():
        put(f"eqt_{B}_steps", e["B_np"], f"{f.name}: Natural Pion steps in Pion's time at {B}")
        put(f"eqt_{B}_L", e["L_np_powerlaw"], f"{f.name}: Natural Pion loss there, from the fit")
        put(f"eqt_{B}_gap", e["gap_powerlaw"], f"{f.name}: Pion - Natural Pion at equal time, fit")
        put(f"eqt_{B}_gap_int", e["gap_interp"], f"{f.name}: same, interpolation")
        put(f"eqt_{B}_Lp", e["L_pion"], f"{f.name}: Pion at {B}")
else:
    for k in ["budget_lo", "budget_hi", "budget_resid", "budget_n_pion", "budget_n_np"] + \
             [f"budget_{B}_{x}" for B in (15000, 19200, 30000, 45000, 73242) for x in ("L", "pow", "int")] + \
             [f"eqt_{B}_{x}" for B in (3840, 19200, 73242) for x in ("steps", "L", "gap", "gap_int", "Lp")] + ["budget_best_19200_pow", "budget_best_19200_int"]:
        missing(k, "python scripts/analysis/budget_fit.py (needs every npv2 run)")

# rounding of the retractions on bare matrices (scripts/probes/ns_rounding.py)
f = R / "probes" / "ns_rounding.json"
if f.exists():
    d = json.load(open(f))
    for ret in ("cayley", "ns", "ns_delta"):
        for th in ("0.2", "0.05"):
            put(f"rnd_{ret}_{th}", d[f"n1376_th{th}"][ret], f"{f.name}: 1376 x 1376, 200 one-sided steps")
    put("rnd_delta_vs_cayley", max(r["delta_vs_cayley"] for r in d.values()), f"{f.name}: max over sizes")
else:
    missing("rnd_ns_0.2", "python scripts/probes/ns_rounding.py")

# ------------------------------------------------------------------- Sections 3-4, 5.4, Appendix D: scripts/paper/v2.sbatch
# Every run is a row of v2.sbatch, found by its tag under runs-record/npv2/<tag>/ (collect_npv2.sh copies them there).
V = "npv2"


def vrun(tag: str, seed: int = 0) -> Path | None:
    hits = runs(tagpat(tag, seed), seed)
    return max(hits, key=lambda d: (d / "log.jsonl").stat().st_mtime) if hits else None


def vinfo(tag: str, seed: int = 0) -> dict | None:
    d = vrun(tag, seed)
    if d is None:
        return None
    i = info(d)
    return i if i["done"] else None


def vval(key: str, tag: str, seed: int = 0):
    i = vinfo(tag, seed)
    if i is None:
        missing(key, f"v2.sbatch row {tag}, seed {seed}")
        return None
    put(key, i["val"], f"{tagpat(tag, seed)} seed {seed}")
    return i["val"]


def vsps(key: str, tag: str, seed: int = 0, need_done: bool = True):
    """Seconds per step of a v2 run (median between logged rows; robust to resumes)."""
    d = vrun(tag, seed)
    if d is None or (need_done and not info(d)["done"]):
        missing(key, f"v2.sbatch row {tag}, seed {seed}")
        return None
    # median seconds per step between logged rows: "wall" restarts at 0 when a run resumes from a checkpoint while the
    # step goes on, so wall / steps would undercount a resumed run; the median also leaves out evaluation pauses
    v = sps_median(d)
    put(key, v, f"{tagpat(tag, seed)} seed {seed}, trimmed-mean s/step between logged rows (RTX)")
    return v


def vbest(key: str, tags: list[str], seed: int = 0):
    """Best final val over an angle grid (one tag per angle). Puts key, key_lr, key_tag and key_interior; missing
    until every point of the grid is done."""
    infos = [(t, vinfo(t, seed)) for t in tags]
    if any(i is None for _, i in infos):
        todo = [t for t, i in infos if i is None]
        for k in (key, key + "_lr", key + "_interior"):
            missing(k, "v2.sbatch rows " + ", ".join(todo))
        return None
    infos.sort(key=lambda ti: ti[1]["lr"])
    # a diverged point (val nan or None) is the worst of the grid, never the best
    fin = lambda v: v if isinstance(v, (int, float)) and math.isfinite(v) else math.inf
    k = min(range(len(infos)), key=lambda j: fin(infos[j][1]["val"]))
    interior = 0 < k < len(infos) - 1
    t, i = infos[k]
    put(key, i["val"], f"best of {', '.join(kseries.tag(x) for x in tags)}: {kseries.tag(t)} (angle {i['lr']}{'' if interior else ', EDGE: extend the grid'})")
    put(key + "_lr", i["lr"], f"angle of {key}")
    put(key + "_interior", interior, f"is {key} inside its grid")
    NOTE[key + "_tag"], NUM[key + "_tag"] = "run of the best angle", t
    return i["val"]


def grid(prefix: str, angles) -> list[str]:
    return [f"{prefix}-{a}" for a in angles]


# the method's own runs, read by many rows below
REF = vval("m_3000", "g3k")                                # the method, 3000 steps, seed 0
vval("m_15000", "g15k")
vsps("m_sps", "g3k")
_m = vrun("g3k")
if _m is not None:
    _man = json.load(open(_m / "manifest.json"))
    put("tokens_per_step", _man["tokens_per_step"], "manifest tokens_per_step")
    put("stats_tokens", _man["config"]["micro_batch"] * _man["config"]["model"]["seq_len"],
        "micro_batch x seq_len: the hooks record the first micro-batch of a step (harness/train.py)")
    put("peak_np", peak(_m), "max peak_gb in the Natural Pion Fast 3000-step log (micro-batch 256)")
    _mc = _man["config"]["model"]
    _dmax = max(_mc["hidden"], _mc["ffn_hidden"])
    put("stats_tokens_per_dim", NUM["stats_tokens"] / _dmax, f"stats_tokens / largest rotated dimension ({_dmax})")
else:
    for k in ("tokens_per_step", "stats_tokens", "peak_np", "stats_tokens_per_dim"):
        missing(k, "v2.sbatch row s3000")
derive("peak_d", lambda a, b: a - b, ("peak_np", "peak_pion"), "peak_np - peak_pion")

# 3.1: the true Fisher (D from targets drawn from the model) against the empirical one, best of three angles each
vbest("fisher_true", grid("g-fisher-mc", ("3e-4", "6e-4", "1.2e-3", "2.4e-3")))
# the empirical Fisher is the method itself: its angle grid is the direction grid with momentum
vbest("fisher_empirical", grid(*kseries.GRIDS[3000]))          # the method's own grid

# Pion's step time in a training run (seed 1 of the 3000-step seeds, same measure as m_sps)
put("pion_run_sps", sps_median(one("seedshead/pion-lr0.001-*", 1)), "seedshead/pion-lr0.001 seed 1: trimmed-mean s/step between logged rows")
# 4.6: precision before the rotation (
# the method vs the same with every product before the rotation in fp32 (rows k-prec-fp32-*), seed 0)
vval("prec_final_3k", "g3k")                    # the method, seed 0 (the best point of its grid)
vval("prec_fp32_3k", "prec-fp32-3k")            # the same with every product before the rotation in fp32
vval("prec_final_15k", "g15k")
vval("prec_fp32_15k", "prec-fp32-15k")
derive("prec_d_3k", lambda a, b: a - b, ("prec_final_3k", "prec_fp32_3k"), "prec_final_3k - prec_fp32_3k")
derive("prec_d_15k", lambda a, b: a - b, ("prec_final_15k", "prec_fp32_15k"), "prec_final_15k - prec_fp32_15k")

# 3.4 / 4.8: retractions (3000 steps angle 0.2; 15000 steps angle 0.05)
vval("ret_cayley_3000", "g-ret-cayley")
vval("ret_ns_3000", "g-ret-ns")
vval("ret_trunc_3000", "g-ret-trunc")
vval("ret_trunc_15000", "g-ret-trunc-15k")
vsps("ret_cayley_sps", "g-ret-cayley")
vsps("ret_ns_sps", "g-ret-ns")
vsps("ret_trunc_sps", "g-ret-trunc")

# 4.1: the step by part (400 profiled steps; ngd_pion.natural_pion's own GPU-synchronised timers)
PARTS = {"refresh": "refresh", "sigma": "sigma", "cayley": "cayley", "Finv": "F^-1",
         "shared": ("generators+momentum", "apply W")}
for t, tag in (("tfac1", "c-prof-alg1-nosig"), ("tfac25", "c-prof-tf25-nosig")):
    d = vrun(tag)
    pf_ = (d / "profile.json") if d is not None else None
    if pf_ is None or not pf_.exists():
        for k in list(PARTS) + ["step", "other"]:
            missing(f"prof_{t}_{k}", f"v2.sbatch row {tag} (writes profile.json)")
        continue
    pr = json.load(open(pf_))
    n, sec = pr["steps"], pr["seconds"]
    for k, names in PARTS.items():
        names = names if isinstance(names, tuple) else (names,)
        put(f"prof_{t}_{k}", sum(sec.get(x, 0.0) for x in names) / n, f"{V}/{tag}/profile.json: {' + '.join(names)} / steps")
    vsps(f"prof_{t}_step", tag, need_done=False)
    derive(f"prof_{t}_other", lambda s, *p: s - sum(p), (f"prof_{t}_step",) + tuple(f"prof_{t}_{k}" for k in PARTS),
           f"prof_{t}_step - the optimizer's parts: forward and backward with the statistics hooks, AdamW, data")

# 4.2: refresh cadence (Algorithm 1 = natural-pion; every row's angle tuned on its own grid)
# the chain's stages on the RMS rule with the method's floor 1e-3 (tags c-*; Algorithm 1 = Fast with every speed-up off)
vbest("tf1_mom", grid("c-alg1", ("3e-4", "6e-4", "1.2e-3", "2.4e-3")))
vval("tf25_mom", "c-tf25")
NUM["tf25_mom_tag"], NOTE["tf25_mom_tag"] = "c-tf25", "run of tf25_mom"
vval("tf10_mom", "c-tf10")
vsps("tf1_mom_sps", "c-alg1-speed")               # Algorithm 1 at its grid's best scale; chain step times from the
# "-speed" rows: the code computed a sigma that exact Cayley with the RMS rule does not need (skipped since 2026-11-10)
if NUM.get("tf25_mom_tag"):
    vsps("tf25_mom_sps", NUM["tf25_mom_tag"] + "-speed")
else:
    missing("tf25_mom_sps", "tf25_mom grid")
vsps("tf10_mom_sps", "c-tf10-speed")
derive("tf10_mom_d", lambda a, b: a - b, ("tf10_mom", "tf1_mom"), "tf10_mom - tf1_mom")
derive("tf25_mom_d", lambda a, b: a - b, ("tf25_mom", "tf1_mom"), "tf25_mom - tf1_mom")
_a1 = [vinfo(NUM.get("tf1_mom_tag") or "c-alg1-6e-4", 0)] + [vinfo("c-alg1-seeds", s) for s in (1, 2, 3)]
if all(_a1):
    put("tf1_mom_mean", sum(i["val"] for i in _a1) / 4, "Algorithm 1 at its best scale, seeds 0-3 (c-alg1 grid, c-alg1-seeds)")
else:
    missing("tf1_mom_mean", "c-alg1 grid and c-alg1-seeds 1-3")
vval("tf1_mom_15k", "c-alg1-15k")
derive("exact_vs_fast_3000", lambda a, b: a - b, ("v2_3000_mean", "tf1_mom_mean"), "v2_3000_mean - tf1_mom_mean")

# 4.3: statistics windows on the T_fac = 25 form, angle 0.2 (reference: tf25-mom-0.2)
vval("win_ref", "c-tf25")
vsps("win_ref_sps", "c-tf25-speed")
for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095"):
    vval(f"win_{k}", f"c-win-{k}")
    vsps(f"win_{k}_sps", f"c-win-{k}-speed")
_w = [NUM.get(f"win_{k}") for k in ("dlast6", "alast6", "adlast6", "aevery2", "betad09", "betad095")]
if NUM.get("win_ref") is not None and all(v is not None for v in _w):
    put("win_maxdev", max(abs(v - NUM["win_ref"]) for v in _w), "largest |win_* - win_ref|")
else:
    missing("win_maxdev", "every win-* row")

# 4.5: batching by shape (Fast with exact Cayley, fp32 statistics, unbatched sigma, one A per matrix)
vval("batch_val", "c-form46")
vsps("batch_sps", "c-form46-speed")

# 4.7: statistics inputs, fp32 against bf16 (the method)
vval("stats_fp32_val", "g-stats-fp32")
vsps("stats_fp32_sps", "g-stats-fp32")
vval("stats_fp32_15k", "g-stats-fp32-15k")
derive("stats_bf16_d", lambda a, b: a - b, ("m_3000", "stats_fp32_val"), "m_3000 - stats_fp32_val")
derive("stats_bf16_saving", lambda a, b: a - b, ("stats_fp32_sps", "m_sps"), "stats_fp32_sps - m_sps")

# cost of the method against Pion and Algorithm 1
derive("speedup_vs_exact", lambda a, b: a / b, ("tf1_mom_sps", "m_sps"), "tf1_mom_sps / m_sps")

# 5.4: direction, X = F^-1 G against X = G (best angle of each grid; 0.2 with momentum is the method's own run)
RMSG = ("3e-4", "6e-4", "1.2e-3")
vbest("dir_fisher_nomom", grid("g-dir-nat-nomom", RMSG + ("2.4e-3", "4.8e-3")))
vbest("dir_fisher_mom", grid(*kseries.GRIDS[3000]))          # the method's own scale grid
vbest("dir_raw_nomom", grid("g-dir-raw-nomom", RMSG + ("7.5e-5", "1.5e-4")))
vbest("dir_raw_mom", grid("g-dir-raw-mom", RMSG + ("7.5e-5", "1.5e-4")))
# the length rules compared among themselves (Fast, momentum, floor 1e-4, fp32): the angle rule's grid
vbest("len_angle", grid("t13-angle", ("0.05", "0.1", "0.2", "0.4", "0.8")))
derive("gain_fisher_nomom", lambda a, b: a - b, ("dir_raw_nomom", "dir_fisher_nomom"), "dir_raw_nomom - dir_fisher_nomom")
derive("gain_fisher_mom", lambda a, b: a - b, ("dir_raw_mom", "dir_fisher_mom"), "dir_raw_mom - dir_fisher_mom")

# 5.4: length rules, Fast with momentum; angle spread = median over logged steps >= 1000 of p90/p10 of the realised angle
# Table 8 and Appendix C on the method's configuration (v2.sbatch rows k-t13-*); the RMS rule is the method's own grid
GRIDS = {"before": ("1e-6", "3e-6", "1e-5", "3e-5", "1e-4"), "fisher": ("3e-4", "6e-4", "1.2e-3", "2.4e-3", "4.8e-3"),
         "quad": ("2.5e-4", "5e-4", "1e-3", "2e-3", "4e-3"), "fixed": ("6.25e-5", "1.25e-4", "2.5e-4", "5e-4", "1e-3")}


def angle_spread(d: Path) -> float | None:
    rs = sorted(r["angle_p90"] / r["angle_p10"] for r in _rows(d)
                if r.get("step", 0) >= 1000 and r.get("angle_p10", 0) > 0 and "angle_p90" in r)
    return rs[len(rs) // 2] if rs else None


put("angle_spread_spectral", 1.0, "by construction: c = theta / sigma turns every matrix by theta")
vbest("len_rms", grid(*kseries.GRIDS[3000]))
t = NUM.get("len_rms_tag")
s_ = angle_spread(vrun(t)) if t else None
(put("angle_spread_rms", s_, f"{V}/{t}: median over steps >= 1000 of angle_p90 / angle_p10") if s_ is not None
 else missing("angle_spread_rms", "k-m3k grid"))
for rule, g in GRIDS.items():
    vbest(f"len_{rule}", grid(f"t13-{rule}", g))
    t = NUM.get(f"len_{rule}_tag")
    s_ = angle_spread(vrun(t)) if t else None
    (put(f"angle_spread_{rule}", s_, f"{V}/{t}: median over steps >= 1000 of angle_p90 / angle_p10") if s_ is not None
     else missing(f"angle_spread_{rule}", f"len-{rule} grid"))
# the RMS rule against the spectral one over seeds (RMS at its best angle: rows "len-rms-seeds", seeds 1-3)
for s in range(4):
    rms = NUM.get("len_rms") if s == 0 else (vinfo("g3k", s) or {}).get("val")          # the method, seed s
    spec = NUM.get("len_angle") if s == 0 else (vinfo("t13-angle-seeds", s) or {}).get("val")  # angle rule at its best
    if rms is not None and spec is not None:
        put(f"rms_gap_s{s}", rms - spec, f"RMS rule at its best angle - the method, seed {s}")
    else:
        missing(f"rms_gap_s{s}", "v2.sbatch rows t13-angle-seeds (after the t13-angle grid) and the method's seeds")

# 5.4: regularisation of the Fisher (reference: the method, floor 1e-4)
for e in ("1e-6", "1e-5", "1e-2"):
    vval(f"reg_eps_{e}", f"f-reg-eps{e}")
vval("reg_eps_1e-4", "f3k")
for dmp in ("1e-4", "1e-3", "1e-2", "1e-1"):
    vval(f"reg_damp_{dmp}", f"f-reg-damp{dmp}")
_r = [NUM.get(f"reg_eps_{e}") for e in ("1e-6", "1e-5", "1e-4", "1e-2")] + \
     [NUM.get(f"reg_damp_{d}") for d in ("1e-4", "1e-3", "1e-2", "1e-1")]
if REF is not None and all(v is not None for v in _r):
    put("reg_best_other_d", min(_r) - REF, "best of the eight alternatives - m_3000")
else:
    missing("reg_best_other_d", "every reg-* row and s3000")

# 5.4: other choices (reference: the method, angle 0.2, cosine)
# trained spectrum drift per retraction, in units of each matrix's largest singular value (retraction_drift.py,
# on the v2 checkpoints: ret-cayley, ret-ns, ret-trunc and the method's s3000)
f = R / "probes" / "retraction_drift.json"
d = json.load(open(f)) if f.exists() else {}
for tag in ("cayley", "ns", "trunc", "ns_delta"):
    if tag in d and d[tag].get("setting") == "rms":
        put(f"rd_{tag}_med", d[tag]["scaled_median"], f"{f.name}: median over the rotated matrices of max|s1-s0|/max s0")
        put(f"rd_{tag}_max", d[tag]["scaled_max"], f"{f.name}: worst matrix, same measure")
        put(f"rd_{tag}_signed", d[tag]["signed_mean"], f"{f.name}: mean signed relative change")
    else:
        for k in ("med", "max", "signed"):
            missing(f"rd_{tag}_{k}", "python scripts/probes/retraction_drift.py (run by collect_npv2.sh once the checkpoints exist)")


PC = R / "probes" / "paper_checks.json"
pc = json.load(open(PC)) if PC.exists() else {}
for k in ("trunc_drift_toy", "tf32_drift", "fp32_drift_200", "fp32_drift_3000", "fp32_drift_3000_max", "fp32_drift_3000_signed_mean", "power_speedup",
          "power_cold_1", "power_cold_5", "power_cold_20", "power_warm_1", "ns_cost", "ns_err_02", "ns_err_04",
          "ns_err_08", "fp32_onestep_cayley", "fp32_onestep_ns", "ns_drift_toy", "cayley_drift_toy", "kfac_vs_exact",
          "svd_exact_s", "svd_power_s"):
    if k in pc:
        put(k, pc[k], "runs-record/probes/paper_checks.json (scripts/probes/paper_checks.py)")
    else:
        missing(k, "run scripts/probes/paper_checks.sbatch (writes runs-record/probes/paper_checks.json)")

if "svd_exact_s" in pc and all(NUM.get(f"prof_tfac25_{k}") is not None for k in ("refresh", "cayley", "Finv", "shared")):
    rest = sum(NUM[f"prof_tfac25_{k}"] for k in ("refresh", "cayley", "Finv", "shared"))
    put("svd_share_of_step", pc["svd_exact_s"] / (pc["svd_exact_s"] + rest),
        "exact spectral norms / (that + the rest of the T_fac 25 optimizer step: refresh, Cayley, F^-1, generators, rotation)")
else:
    missing("svd_share_of_step", "paper_checks.json and the prof-tf25 profile")

# derived from several sections above
if None not in (NUM.get("pion_full_best"), NUM.get("v2_full"), NUM.get("muon_full")):
    put("gap_closed_frac", (NUM["pion_full_best"] - NUM["v2_full"]) / (NUM["pion_full_best"] - NUM["muon_full"]),
        "(pion_full_best - v2_full) / (pion_full_best - muon_full): the share of the Pion-Muon gap closed")
else:
    missing("gap_closed_frac", "pion_full_best, v2_full, muon_full")
# AdamW's lr at full length (scripts/paper/adamw_long.sbatch: the full-length runs repeated with only adamw_lr changed;
# Pion at its published lr 1e-3, Natural Pion at its full-length scale); 8e-3 is the paper's own pair
# Pion at its better rotation lr (1e-3 / 2e-3) at every AdamW lr, as in Table 3
_aw = {("pion", "8e-3"): NUM.get("pion_full_best"), ("np", "8e-3"): NUM.get("v2_full")}
def _awv(name, a):
    h = [d for d in runs(f"adamwlong/{name}-adamw{a}/*") if info(d)["done"]]
    return info(h[-1])["val"] if h else None
for _a in ("1e-3", "4e-3"):
    _p1, _p2 = _awv("pion", _a), _awv("pion-lr2e-3", _a)
    _aw[("pion", _a)] = min(_p1, _p2) if None not in (_p1, _p2) else None
    _aw[("np", _a)] = _awv("np", _a)
for (_m, _a), _v in _aw.items():
    (put(f"aw_{_m}_{_a}", _v, f"adamwlong/{_m}-adamw{_a} (8e-3: the paper's full-length run)") if _v is not None
     else missing(f"aw_{_m}_{_a}", f"adamw_long.sbatch {_m} {_a}"))
if all(v is not None for v in _aw.values()):
    for _m in ("pion", "np"):
        _best = min(("1e-3", "4e-3", "8e-3"), key=lambda a: _aw[(_m, a)])
        put(f"aw_{_m}_best_lr", float(_best), f"the best of three AdamW lrs at full length, {_m}")
        put(f"aw_{_m}_gain", _aw[(_m, "8e-3")] - _aw[(_m, _best)], f"{_m}: val at 8e-3 - val at its best AdamW lr")
    _gaps = [_aw[("pion", a)] - _aw[("np", a)] for a in ("1e-3", "4e-3", "8e-3")]
    put("aw_gap_min", min(_gaps), "smallest full-length gap over the three AdamW lrs")
    put("aw_gap_max", max(_gaps), "largest")
    _pb = min(_aw[("pion", a)] for a in ("1e-3", "4e-3", "8e-3"))
    _nb = min(_aw[("np", a)] for a in ("1e-3", "4e-3", "8e-3"))
    put("aw_pion_best_val", _pb, "Pion at full length at its best AdamW lr")
    put("aw_gap_best", _pb - _nb, "full-length gap, each method at its own best AdamW lr")
    if NUM.get("muon_full") is not None:
        put("aw_gap_closed_frac_best", (_pb - _nb) / (_pb - NUM["muon_full"]),
            "share of the Pion-Muon gap closed, each at its best AdamW lr (Muon at 8e-3, not re-tuned)")
    else:
        missing("aw_gap_closed_frac_best", "muon_full")
else:
    for _k in ("aw_pion_best_lr", "aw_pion_gain", "aw_np_best_lr", "aw_np_gain", "aw_gap_min", "aw_gap_max",
               "aw_pion_best_val", "aw_gap_best", "aw_gap_closed_frac_best"):
        missing(_k, "every Table 3 run")
for _a in ("1e-3", "4e-3", "8e-3"):
    if _aw[("pion", _a)] is not None and _aw[("np", _a)] is not None:
        put(f"aw_gap_{_a}", _aw[("pion", _a)] - _aw[("np", _a)], f"Pion - Natural Pion at AdamW {_a}, full length")
    else:
        missing(f"aw_gap_{_a}", f"both full-length runs at AdamW {_a}")
if NUM.get("ret_trunc_15000") is not None and NUM.get("v2_15000_s0") is not None:
    put("ret_trunc_15k_d", NUM["ret_trunc_15000"] - NUM["v2_15000_s0"], "ret_trunc_15000 - v2_15000_s0")
else:
    missing("ret_trunc_15k_d", "g-ret-trunc-15k and the method at 15000")
_rl = [NUM.get(k) for k in ("v2_3000_s0", "ret_cayley_3000", "ret_ns_3000", "ret_trunc_3000")]
(put("ret_loss_range", max(_rl) - min(_rl), "max - min of the four retractions' val at 3000 steps") if None not in _rl
 else missing("ret_loss_range", "retraction rows"))
_sd = [NUM.get(f"v2_3000_s{i}") for i in range(4)]
(put("v2_3000_range", max(_sd) - min(_sd), "max - min of the method's val over seeds 0-3 at 3000 steps") if None not in _sd
 else missing("v2_3000_range", "the method's seeds at 3000"))
if NUM.get("v2_full") is not None:
    put("muon_gap_np", NUM["v2_full"] - NUM["muon_full"], "v2_full - muon_full")
else:
    missing("muon_gap_np", "v2_full")


def main():
    out = ROOT / "docs/paper"
    json.dump({k: v for k, v in NUM.items()}, open(out / "numbers.json", "w"), indent=1, default=str)
    with open(out / "numbers.md", "w") as f:
        f.write("# Numbers of the preprint\n\nGenerated by `scripts/analysis/paper_numbers.py`; do not edit.\n\n")
        f.write("| key | value | source |\n|---|---|---|\n")
        for k in NUM:
            v = NUM[k]
            f.write(f"| `{k}` | {'**MISSING**' if v is None else (f'{v:.6g}' if isinstance(v, float) else v)} | {NOTE[k]} |\n")
    n_miss = sum(v is None for v in NUM.values())
    print(f"{len(NUM)} keys, {n_miss} missing")


if __name__ == "__main__":
    main()
