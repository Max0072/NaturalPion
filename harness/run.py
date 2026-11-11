"""Command-line entry point: one process, one run.

    python -m harness.run --optimizer ngd --lr 1e-3 --seed 0
    python -m harness.run --anchor bilateral

Every flag maps to a field of `RunConfig`, whose hash names the output
directory, so a sweep is a job array over flag combinations and the results
are keyed by what produced them.

`--anchor` ignores the optimizer flags and runs the configuration their
published figures come from, then reports the comparison. Run it before
trusting any number this harness produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from dataclasses import fields, replace
from typing import get_args, get_type_hints

import torch

from .anchor import KNOWN_DIFFERENCES, TARGETS, anchor_config, check
from .config import RunConfig
from .model import ModelConfig
from .train import train


def _boolean(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = RunConfig()
    # `from __future__ import annotations` leaves `field.type` a string, so the
    # annotations have to be resolved or every flag silently parses as str
    hints = get_type_hints(RunConfig)
    for f in fields(RunConfig):
        if f.name == "model":
            continue
        annotated = hints[f.name]
        if type(None) in get_args(annotated):            # `X | None`: parse as X, None stays the default
            annotated = next(a for a in get_args(annotated) if a is not type(None))
        kind = _boolean if annotated is bool else annotated
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=getattr(base, f.name))
    # The nested model config gets flags of its own. Without them the
    # architecture and its initialisation are the only part of `RunConfig` that
    # cannot be reached from the command line, which is not a deliberate
    # restriction -- the loop above skips `model` because it is a dataclass
    # rather than a scalar, and nothing put its fields back.
    model_hints = get_type_hints(ModelConfig)
    for f in fields(ModelConfig):
        annotated = model_hints[f.name]
        kind = _boolean if annotated is bool else annotated
        ap.add_argument(
            f"--{f.name.replace('_', '-')}", type=kind, default=getattr(base.model, f.name)
        )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None, help="cut the run short, for smoke tests")
    ap.add_argument(
        "--force", action="store_true",
        help="take the run directory's lock even if another process holds it",
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="start over even if a checkpoint is present",
    )
    ap.add_argument(
        "--from-manifest", metavar="MANIFEST",
        help="repeat a run exactly as its manifest.json records it; only --out-dir, --data-path, --val-path, "
             "--device and --max-steps apply",
    )
    ap.add_argument(
        "--anchor", choices=sorted(TARGETS),
        help="run their published configuration and compare against their figure",
    )
    args = ap.parse_args()

    if args.anchor:
        # Only what the anchor is allowed to vary: where it reads and writes,
        # the seed, the accumulation chunk -- and precision, but *only* when
        # asked for explicitly, because their configuration is bf16 and a
        # default silently overriding that would make the run measure our
        # arithmetic rather than their number.
        overrides = dict(seed=args.seed, out_dir=args.out_dir,
                         data_path=args.data_path, val_path=args.val_path,
                         micro_batch=args.micro_batch)
        if any(a.startswith("--precision") for a in sys.argv[1:]):
            overrides["precision"] = args.precision
        cfg = anchor_config(args.anchor, **overrides)
        print(f"anchor: {args.anchor}, target {TARGETS[args.anchor]}, name {cfg.name}")
        out = train(cfg, device=args.device, max_steps=args.max_steps,
                    resume=not args.no_resume, force=args.force)
        result = check(out / "log.jsonl", args.anchor, expected_steps=cfg.train_steps)
        (out / "anchor.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if result["status"] == "incomplete":
            print(
                f"\nran {result['steps_run']} of {cfg.train_steps} steps -- too short to "
                "compare. The target is a converged number."
            )
        elif not result["matched"]:
            print("\nmissed. before calling it a bug, rule out:")
            for reason in KNOWN_DIFFERENCES:
                print(f"  - {reason}")
        return

    if args.from_manifest:
        recorded = RunConfig.from_manifest(json.loads(Path(args.from_manifest).read_text())["config"])
        cfg = replace(recorded, out_dir=args.out_dir, data_path=args.data_path, val_path=args.val_path)
        print(f"from {args.from_manifest}: {cfg.name}")
        out = train(cfg, device=args.device, max_steps=args.max_steps,
                    resume=not args.no_resume, force=args.force)
        print(f"\nwrote {out}")
        return

    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(RunConfig)
        if f.name != "model" and hasattr(args, f.name)
    }
    model_overrides = {
        f.name: getattr(args, f.name) for f in fields(ModelConfig) if hasattr(args, f.name)
    }
    cfg = replace(base, **overrides, model=replace(base.model, **model_overrides))
    print(json.dumps(cfg.manifest()["config"] | {"name": cfg.name}, indent=2, default=str))
    out = train(cfg, device=args.device, max_steps=args.max_steps,
                resume=not args.no_resume, force=args.force)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
