"""Command-line entry point: one process, one run.

    python -m harness.run --optimizer natural-pion-fast --lr 3e-4 --seed 0
    python -m harness.run --from-manifest runs-record/<...>/manifest.json

Every flag maps to a field of `RunConfig`, whose hash names the output
directory, so a sweep is a job array over flag combinations and the results
are keyed by what produced them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import fields, replace
from typing import get_args, get_type_hints

import torch

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
    args = ap.parse_args()

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
