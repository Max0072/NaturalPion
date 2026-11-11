"""Why the Newton-Schulz retraction inflated W's spectrum in training (+1.1% mean, npns/s3000) when its own
truncation error, R = Cay - 2 (I+A)^-1 A^8, can only shrink it.

Runs the first `--max-steps` steps of the method (natural-pion-fast in its final form: Pion's RMS length rule, floor 1e-3, TF32 before the rotation, scale 6e-4, cosine over 3000)
with an instrumented optimizer and records, per step:
  * the relative change of every weight's ||W||_F^2 = sum sigma_i^2 across `step()` (fp64; exactly invariant
    under the two-sided rotation), max |.| and mean signed over matrices, so the drift is located in time
    (refresh steps or not); the full singular values (fp64) at the first and last step;
  * sigma_est / ||X||_2 exact for every matrix (power-iteration underestimate makes the true angle > theta);
  * the matmul precision state inside the rotation (autocast, allow_tf32, float32_matmul_precision).

    python scripts/probes/ns_diag.py <ns|cayley|trunc> <max_steps> <data dir> <out dir>
writes runs-record/probes/ns_diag_<ret>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import harness.train as T
from ngd_pion.natural_pion import NaturalPionFast

ROOT = Path(__file__).resolve().parents[2]
RET, STEPS = sys.argv[1], int(sys.argv[2])
REC = {"ret": RET, "steps": [], "sigma_ratio": [], "precision": []}


class Probe(NaturalPionFast):
    def _sigma(self, state, group, X_in, X_out):
        s = super()._sigma(state, group, X_in, X_out)
        # ||X||_2 for skew X is sqrt(lambda_max(-X^2)); svd on skew matrices (paired singular values) falls back
        # to a slow path in cusolver, eigvalsh does not.
        nrm = lambda X: torch.linalg.eigvalsh(-(X.double() @ X.double())).amax().clamp_min(0).sqrt().item()
        exact = max(nrm(X_in), nrm(X_out))
        self._ratios.append(s / max(exact, 1e-30))
        return s

    def _rotate(self, active, group):
        REC["precision"].append([torch.is_autocast_enabled("cuda"), torch.backends.cuda.matmul.allow_tf32,
                                 torch.get_float32_matmul_precision()])
        return super()._rotate(active, group)

    @torch.no_grad()
    def step(self, closure=None):
        self._ratios = []
        ps = [p for g in self.param_groups for p in g["params"]]
        refresh = any(self.state[p].get("since_refactor", 0) == 0 for p in ps if "bases" in self.state[p])
        if "sv0" not in REC:
            REC["sv0"] = [torch.linalg.svdvals(p.detach().double()).cpu() for p in ps]
        before = [p.detach().double().square().sum() for p in ps]
        out = super().step(closure)
        rel = [(p.detach().double().square().sum() - b) / b for p, b in zip(ps, before)]
        self._ps = ps
        REC["steps"].append({
            "t": len(REC["steps"]), "theta": self.param_groups[0]["lr"], "refresh_before": refresh,
            "max_abs": max(r.abs().max().item() for r in rel),
            "mean_signed": sum(r.mean().item() for r in rel) / len(rel),
        })
        if self._ratios:                     # the per-matrix sigma path only; Fast batches sigma
            REC["sigma_ratio"].append([min(self._ratios), sorted(self._ratios)[len(self._ratios) // 2]])
        return out


T.NATURAL_PION["natural-pion-fast"] = Probe
from harness import run  # noqa: E402

sys.argv = ["run", "--optimizer", "natural-pion-fast", "--np-retraction", RET, "--lr", "6e-4", "--np-kwargs", '{"length": "rms", "tf32": "upstream", "fisher_dtype": "bf16"}', "--ngd-eps", "1e-3", "--ngd-beta1", "0.9",
            "--train-steps", "3000", "--max-steps", str(STEPS), "--batch-sequences", "512", "--micro-batch", "256",
            "--precision", "bf16", "--seed", "0", "--device", "cuda", "--log-every", "23",
            "--eval-every", "100000", "--adamw-lr", "8e-3", "--data-path", sys.argv[3] + "/c4_train.bin",
            "--val-path", sys.argv[3] + "/c4_val.bin", "--out-dir", sys.argv[4]]
try:
    run.main()
finally:
    s = REC["steps"]
    opt = next((o for o in __import__("gc").get_objects() if isinstance(o, Probe)), None)
    if opt is not None and "sv0" in REC:
        d = [(torch.linalg.svdvals(p.detach().double()).cpu() - a) / a for p, a in zip(opt._ps, REC.pop("sv0"))]
        REC["sv_drift_median"] = sorted(x.abs().max().item() for x in d)[len(d) // 2]
        REC["sv_drift_max"] = max(x.abs().max().item() for x in d)
        REC["sv_signed_mean"] = sum(x.mean().item() for x in d) / len(d)
    REC.pop("sv0", None)
    REC["summary"] = {
        "cum_mean_signed": sum(x["mean_signed"] for x in s),
        "cum_mean_signed_refresh": sum(x["mean_signed"] for x in s if x["refresh_before"]),
        "worst_step_max_abs": max(x["max_abs"] for x in s) if s else None,
        "sigma_ratio_min": min(r[0] for r in REC["sigma_ratio"]) if REC["sigma_ratio"] else None,
        "sv_drift_median": REC.get("sv_drift_median"), "sv_drift_max": REC.get("sv_drift_max"),
        "sv_signed_mean": REC.get("sv_signed_mean"),
        "precision_seen": sorted({json.dumps(p) for p in REC["precision"]}),
    }
    dst = ROOT / "runs-record" / "probes" / f"ns_diag_{RET}.json"
    dst.write_text(json.dumps(REC, indent=1))
    print(json.dumps(REC["summary"], indent=1))
