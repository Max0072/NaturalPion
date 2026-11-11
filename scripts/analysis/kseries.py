"""The k-series (scripts/paper/v2.sbatch rows 331-452): every run the preprint reads, at a cosine decay depth of 1% of
each optimizer's peak. One place for the tag prefix and for which run is "the method" at each budget, shared by
paper_numbers.py, budget_fit.py, figures.py and the probes.

    tag(t)                  the k-series name of a v2 tag t
    best(base, grid, pts)   the tag of the best finished seed-0 point of a grid (diverged points never win), or None
    method(base, steps, s)  the tag of the method's run at a budget and seed
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PREFIX = "k-"
GRIDS = {3000: ("m3k", ("3e-4", "6e-4", "1.2e-3", "2.4e-3")),
         9000: ("g9k", ("1.5e-4", "3e-4", "6e-4", "1.2e-3")),
         15000: ("m15k", ("1.5e-4", "3e-4", "6e-4", "1.2e-3")),
         30000: ("m30k", ("1.5e-4", "3e-4", "6e-4", "1.2e-3"))}
SEEDS = {3000: "g3k", 15000: "g15k"}          # seeds 1-3 at the best scale of the budget's grid
FULL_AT_3E4 = "d1pc-73242"                    # the full length at depth 1%, scale 3e-4 (run before the k-series)


def tag(t: str) -> str:
    return t if t.startswith(PREFIX) else PREFIX + t


def _final(d: Path):
    cfg = json.load(open(d / "manifest.json"))["config"]
    v = [json.loads(l.replace("NaN", "null")) for l in open(d / "log.jsonl") if '"val_loss"' in l]
    done = bool(v) and v[-1]["step"] >= cfg["train_steps"] - 1
    return cfg, (v[-1]["val_loss"] if done else None), done


def best(base: Path, grid: str, pts) -> str | None:
    """`base` is a directory holding <tag>/<run>/ (runs-record/npv2 or $DATA_p330/runs/npv2)."""
    out = None
    for a in pts:
        t = tag(f"{grid}-{a}")
        hits = []
        for d in sorted((base / t).glob("*")):
            if (d / "manifest.json").exists():
                cfg, v, done = _final(d)
                if cfg["seed"] == 0:
                    hits.append((done, v))
        if not hits or not any(done for done, _ in hits):
            return None                                # an unfinished grid has no best yet
        v = next(v for done, v in hits if done)
        if isinstance(v, (int, float)) and math.isfinite(v) and (out is None or v < out[0]):
            out = (v, t)
    return out[1] if out else None


def scale(base: Path, steps: int) -> str | None:
    """The method's peak scale at a grid budget, as written in its tag (e.g. "6e-4"); None until the grid is done.
    (Not t.rsplit("-", 1): "6e-4" itself contains a minus.)"""
    g, _ = GRIDS[steps]
    t = best(base, g, GRIDS[steps][1])
    return t[len(tag(g)) + 1:] if t else None


def method(base: Path, steps: int, seed: int = 0) -> str | None:
    if steps in SEEDS and seed > 0:
        return tag(SEEDS[steps])
    if steps in GRIDS:
        g, pts = GRIDS[steps]
        return best(base, g, pts)
    if steps == 45000:
        return tag("g45k")
    if steps == 73242:
        b = best(base, *GRIDS[30000])
        if b is None:
            return None
        return FULL_AT_3E4 if b.endswith("-3e-4") else tag("g73242")
    raise ValueError(f"no method run at {steps} steps")


# Pion's lr per budget: the better (seed 0) of the published 1e-3 and 2e-3, as Natural Pion's scale is chosen per
# budget. Paths are relative to runs-record/. 2e-3 runs: scripts/paper/pion_lr_depth.sbatch (depth 1%).
PION_1E3 = {3000: ["adamw2d/pion-lr0.001-s0-9e6e6d32f6", "seedshead/pion-lr0.001-*"],
            3840: ["stepnormfinal/pion3840/*", "stepnormlong/pion3840-lr1e-3/*"], 4500: ["stepnormfinal/pion4500/*"],
            5300: ["stepnormclock/pion-lr0.001-s0-dbe147d5df"], 15000: ["ladder15k/pion-lr0.001-s0-f67fdad3e9"],
            19200: ["stepnormfinal/pion19200/*", "stepnormlong/pion19200-lr1e-3/*"],
            30000: ["stepnormlong/pion30000-lr1e-3/*"], 45000: ["stepnormlong/pion45000-lr1e-3/*"],
            73242: ["stepnormlong/pion73242-lr1e-3/*"]}
PION_2E3 = {B: [f"pion-lrdepth/pion{B}-lr2e-3/*"] for B in (3840, 15000, 19200, 30000, 45000, 73242)}


def _seeds(record: Path, pats) -> dict:
    """{seed: final val} of the finished runs matching any of the patterns."""
    out = {}
    for pat in pats:
        for d in sorted(record.glob(pat)):
            if (d / "manifest.json").exists():
                cfg, v, done = _final(d)
                if done:
                    out[cfg["seed"]] = v
    return out


def pion_best(record: Path, steps: int):
    """(lr, seeds {seed: val}, mean) of Pion's better lr at a budget; None while a 2e-3 run of that budget is unfinished.
    The two lrs are compared on the seeds both have (the mean over them); budgets without a 2e-3 run keep 1e-3."""
    one = _seeds(record, PION_1E3[steps])
    if 0 not in one:
        return None
    if steps not in PION_2E3:
        return 1e-3, one, sum(one.values()) / len(one)
    two = _seeds(record, PION_2E3[steps])
    if 0 not in two:
        return None
    common = sorted(set(one) & set(two))
    m1 = sum(one[s] for s in common) / len(common)
    m2 = sum(two[s] for s in common) / len(common)
    lr, seeds = (1e-3, one) if m1 <= m2 else (2e-3, two)
    return lr, seeds, sum(seeds.values()) / len(seeds)
