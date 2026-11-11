"""How far trained singular values moved from their initial values, per retraction. CPU, a few minutes.

For each checkpoint the initial weights are rebuilt from the run's seed and model configuration, and for every
rotated matrix with singular values s0 (initial) and s1 (trained, both sorted):
  rel     = max_i |s1_i / s0_i - 1|           relative; dominated by the smallest s0 when it is near zero
  scaled  = max_i |s1_i - s0_i| / max_i s0_i   in units of the matrix's own scale
Reported over the rotated units (each matrix; the query projection per attention head, as it is rotated): median of
rel, median and max of scaled, mean signed change mean_i(s1/s0 - 1).
Each checkpoint is measured in its own process (rebuilding several models in one process gave wrong numbers).

    python scripts/probes/retraction_drift.py        # writes runs-record/probes/retraction_drift.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("DATA_p330", "")) / "runs"
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import kseries  # noqa: E402
# the method (RMS rule), 3000 steps, seed 0, k-series (scripts/paper/v2.sbatch rows; scripts/analysis/kseries.py)
CHECKPOINTS = {
    "cayley": "npv2/k-g-ret-cayley/*/checkpoint.pt",
    "ns_delta": f"npv2/{kseries.method(RUNS / 'npv2', 3000) or 'k-m3k-unfinished'}/*/checkpoint.pt",  # the method itself
    "ns": "npv2/k-g-ret-ns/*/checkpoint.pt",
    "trunc": "npv2/k-g-ret-trunc/*/checkpoint.pt",
}


def measure(ckpt: Path) -> dict:
    import torch
    from harness.model import ModelConfig, Transformer
    c = json.load(open(ckpt.parent / "manifest.json"))["config"]
    m = ModelConfig(**c["model"])
    torch.manual_seed(c["seed"])
    init, trained = Transformer(m), Transformer(m)
    trained.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    rel, scaled, signed = [], [], []
    queries = {id(q) for q in init.query_projections()}
    for a, b in zip(init.parameter_split()[0], trained.parameter_split()[0]):
        k = m.heads if id(a) in queries else 1          # the units that rotate (harness.train.query_blocks)
        for wa, wb in zip(a.weight.detach().double().chunk(k), b.weight.detach().double().chunk(k)):
            s0, s1 = torch.linalg.svdvals(wa), torch.linalg.svdvals(wb)
            rel.append(float((s1 / s0 - 1).abs().max()))
            scaled.append(float((s1 - s0).abs().max() / s0.max()))
            signed.append(float((s1 / s0 - 1).mean()))
    rel.sort(); scaled.sort()
    return {"checkpoint": str(ckpt.relative_to(RUNS)), "setting": "rms", "units": len(rel), "rel_median": rel[len(rel) // 2],
            "scaled_median": scaled[len(scaled) // 2], "scaled_max": scaled[-1],
            "signed_mean": sum(signed) / len(signed)}


def main():
    if len(sys.argv) == 2:                             # child: one checkpoint
        print(json.dumps(measure(Path(sys.argv[1]))))
        return
    out = {}
    for tag, pat in CHECKPOINTS.items():
        hits = sorted(RUNS.glob(pat))
        if not hits:
            print(f"{tag}: no checkpoint under {RUNS / pat}")
            continue
        res = subprocess.run([sys.executable, __file__, str(hits[0])], capture_output=True, text=True, check=True)
        out[tag] = json.loads(res.stdout.strip().splitlines()[-1])
        print(tag, out[tag], flush=True)
    (ROOT / "runs-record" / "probes" / "retraction_drift.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
