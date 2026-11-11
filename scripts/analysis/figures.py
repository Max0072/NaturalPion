"""Figures for the paper (docs/paper/latex/main.tex), from runs-record/: val against steps and against wall-clock time, full length.

    python scripts/analysis/figures.py [--muon path/to/muon/log.jsonl] [--out docs/paper/figures/preprint]

Colours: the first categorical slots of the dataviz reference palette, in fixed order; every line is also
direct-labelled and has its own dash, so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kseries  # noqa: E402

R = "runs-record/stepnormlong"
SERIES = [  # label, log glob or path, colour, dash
    ("Pion", f"runs-record/{kseries.PION_2E3[73242][0] if (kseries.pion_best(Path('runs-record'), 73242) or (1e-3,))[0] == 2e-3 else 'stepnormlong/pion73242-lr1e-3/*'}/log.jsonl", "#2a78d6", "-"),
    ("Natural Pion", f"runs-record/npv2/{kseries.method(Path('runs-record/npv2'), 73242) or 'k-unfinished'}/*/log.jsonl", "#1baf7a", "-"),     # FULL: kseries
]
MUON = ("Muon", None, "#eda100", "-.")
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
FIGSIZE = (3.4, 2.4)        # one ICML column (3.25 in) at 1:1, so 7 pt here is 7 pt on the page
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8})


def curve(path: str):
    """(step, wall seconds, val) at every evaluation of the last attempt."""
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        rows.append(r)
    wall = {}
    for r in rows:
        if "wall" in r:
            wall[r["step"]] = r["wall"]
    steps = sorted(wall)
    out = []
    for r in rows:
        if "val_loss" in r:
            s = r["step"]
            near = min(steps, key=lambda k: abs(k - s)) if steps else None
            out.append((s + 1, wall.get(near), r["val_loss"]))
    return out


def plot(series, x_index: int, xlabel: str, path: Path, xscale: float = 1.0):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    ends = []
    for label, pts, colour, dash, running in series:
        xs = [p[x_index] / xscale for p in pts if p[x_index] is not None]
        ys = [p[2] for p in pts if p[x_index] is not None]
        ax.plot(xs, ys, color=colour, linestyle=dash, linewidth=1.4, label=label)
        ends.append([xs[-1], ys[-1], f"{ys[-1]:.4f}" + (" (running)" if running else "")])
    # end labels: push apart any two that would overlap (same x region, within 0.035 in loss, a line of 7 pt text at this size)
    ends.sort(key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        ty = y
        for px, py in placed:
            if abs(px - x) < 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0]) and abs(ty - py) < 0.035:
                ty = py + 0.035
        placed.append((x, ty))
        ax.annotate(text, (x, y), xytext=(x, ty), textcoords="data", va="center", fontsize=7, color=INK,
                    xycoords="data", annotation_clip=False)
        ax.texts[-1].set_x(x)
        ax.texts[-1].set_position((x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ty))
    ax.set_ylim(3.28, 3.9)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("validation loss", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


PION_C, NP_C = SERIES[0][2], SERIES[1][2]


def _axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def budget_figure(path: Path, fit_json: str = "runs-record/probes/budget_fit.json"):
    """Final val of every full-schedule run against its budget, with the power law L(B) = a + b B^-gamma of
    scripts/analysis/budget_fit.py; the arrow at Pion's full-length loss shows the steps saved."""
    import numpy as np
    d = json.load(open(fit_json))
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    for key, label, colour, marker in (("pion", "Pion", PION_C, "o"), ("natural_pion", "Natural Pion", NP_C, "s")):
        B, L = d[key]["B"], d[key]["L"]
        a, b, g = d[key]["fit_abc"]
        xs = np.geomspace(min(B) * 0.9, 80000, 200)
        ax.plot(xs, a + b * xs ** -g, color=colour, linewidth=1.4, label=f"{label}, fit")
        ax.plot(B, L, linestyle="none", marker=marker, markersize=4, color=colour, markeredgecolor="white",
                markeredgewidth=0.8, label=f"{label}, runs")
    # steps saved at Pion's full-length loss
    Lt = d["pion"]["L"][-1]
    a, b, g = d["natural_pion"]["fit_abc"]
    Bnp = (b / (Lt - a)) ** (1 / g)
    ax.annotate("", xy=(Bnp, Lt), xytext=(d["pion"]["B"][-1], Lt),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1))
    ax.text(0.03, 0.05, f"arrow: {d['pion']['B'][-1] / Bnp:.2f}x fewer steps\nat Pion's full-length loss",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([3000, 10000, 30000, 73242])
    ax.set_xticklabels(["3 000", "10 000", "30 000", "73 242"])
    ax.minorticks_off()
    _axes(ax, "budget (steps of the run's own schedule)", "final validation loss")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def angle_figure(path: Path, pion_json: str = "runs-record/probes/pion_angles.json",
                 np_log: str = "runs-record/npv2/k-m3k-6e-4/*-s0-*/log.jsonl"):
    """Largest rotation angle per rotated block and step: median and 10th-90th percentile band over the blocks.
    Pion from scripts/probes/pion_angles.py (lr 1e-3, 3 840-step schedule, first 1 058 steps: the probe hit its time
    limit there); Natural Pion from its own log (the method at 3 000 steps, seed 0), cut to the steps Pion covers."""
    import numpy as np
    pa = json.load(open(pion_json))
    ps = sorted(int(k) for k in pa)
    pq = np.array([np.percentile(pa[str(s)], [10, 50, 90]) for s in ps])
    hits = glob.glob(np_log)
    assert len(hits) == 1, (np_log, hits)
    rows = []
    for line in open(hits[0]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        if "angle_p90" in r and r.get("angle_p10", 0) > 0:
            rows.append(r)
    rows = [r for r in rows if r["step"] <= ps[-1]]
    ns = [r["step"] for r in rows]
    nq = np.array([[r["angle_p10"], r["angle_p50"], r["angle_p90"]] for r in rows])
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    for xs, q, label, colour, dash in ((ps, pq, "Pion", PION_C, "--"), (ns, nq, "Natural Pion", NP_C, "-")):
        ax.fill_between(xs, q[:, 0], q[:, 2], color=colour, alpha=0.18, linewidth=0)
        ax.plot(xs, q[:, 1], color=colour, linestyle=dash, linewidth=1.4, label=label)
        ax.text(xs[-1], q[-1, 1], f"  {label}", va="center", fontsize=7, color=INK)
    ax.axhline(0.2, color=MUTED, linewidth=1, linestyle=":")
    ax.text(ns[-1], 0.2, "0.2 ", ha="right", va="bottom", fontsize=6,
            color=MUTED)
    ax.set_yscale("log")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _axes(ax, "optimizer step", "largest angle per step (rad)")
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muon", default=None, help="Muon's log.jsonl (Natural-Muon run), optional")
    ap.add_argument("--out", default="docs/paper/figures/preprint")
    ap.add_argument("--only", choices=("budget", "angles"), action="append",
                    help="write only these figures (default: all); val_vs_steps needs the Muon log")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only:
        if "budget" in args.only:
            budget_figure(out / "budget_fit.png")
            print("wrote", out / "budget_fit.png")
        if "angles" in args.only:
            angle_figure(out / "angles.png")
            print("wrote", out / "angles.png")
        return
    budget_figure(out / "budget_fit.png")
    angle_figure(out / "angles.png")
    series = []
    for label, pattern, colour, dash in SERIES:
        hits = glob.glob(pattern)
        assert len(hits) == 1, (pattern, hits)
        series.append((label, curve(hits[0]), colour, dash, False))
    if args.muon:
        pts = curve(args.muon)
        series.append((MUON[0], pts, MUON[2], MUON[3], pts[-1][0] < 73242))
    plot(series, 0, "optimizer steps (thousands)", out / "val_vs_steps.png", xscale=1000)
    print("wrote", out / "val_vs_steps.png")


if __name__ == "__main__":
    main()
