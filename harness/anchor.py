"""The anchor run: reproduce a published number before trusting our own.

Everything this project measures happens inside a harness we wrote. A reviewer
is entitled to ask how we know the harness is equivalent to theirs -- perhaps
our Pion baseline looks weak because of a bug, a different data order or a
subtly different architecture, and no measurement taken inside the harness can
answer that.

The anchor answers it from outside: run *their* configuration in *our* harness
and check we land on *their* published number. Once that holds, differences
measured inside are attributable to what we changed.

Their paper reports exactly one pair of concrete figures for the 60M setting,
in section 2.4.3: final loss **3.3575** for the bilateral update and **3.3654**
for alternate, both with Lie+Lie momentum. Note this is not the configuration
their 60M shell script runs -- that script sets `alternate` and
`transported_ambient_ambient`. The anchor has to follow the number, not the
script, or a miss says nothing.

Their `opt_llama_60M_pion.sh` has since been read rather than inferred, and it
confirms the shape this harness copies: 8 layers, hidden 512, ffn 1376, 8
heads, kv 64, RMSNorm, seq 256, rotary base 10000, init 0.02, T5 tokenizer,
lr 1e-3 to 1e-5 cosine with no warmup, weight decay 0.1, clip 1.0, global batch
512, 9.6B tokens, `--pion-degree 2`, `--pion-scaling rms`, `--pion-rms 0.2`. It
also confirms the two defaults above.

What it adds is `--bf16`, which `anchor_config` now sets, and
`--use-same-init-for-output-layers`, which makes O and down initialise at the
same 0.02 as everything else rather than at Megatron's default
`0.02/sqrt(2*layers)` -- this harness initialises uniformly at `init_std`, so
that one matches without having been aimed at.

Their optimizer has since been read too, not only their script, and it moved
three things. `_scale_update_matrix_rms` takes an `update_side` and normalises
the side being applied, which `pion_baseline` now does; their default leaves
the second moment off, which `pion_second_moment` now defaults to; and their
`pion_qkv_split_granularity` defaults to per-head Q, which every rotating
optimizer here now does (`train.query_blocks`). Two entries left `KNOWN_DIFFERENCES`
outright: their Pion never applies weight decay to a rotated weight, and their
sample windows cross document boundaries exactly as ours do.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import RunConfig

__all__ = ["anchor_config", "TARGETS", "TOLERANCE", "KNOWN_DIFFERENCES", "check"]

#: Final training loss reported in section 2.4.3, by update side.
TARGETS = {"bilateral": 3.3575, "alternate": 3.3654}

#: What counts as a match. Their own bilateral-to-alternate gap is 0.0079, so a
#: reproduction has to be tighter than that to be saying anything; 0.02 leaves
#: room for data-order and harness differences without admitting a real defect.
TOLERANCE = 0.02

#: Reasons the number could miss even with correct code. A miss should be
#: diagnosed against this list before anything is called a bug.
#: **The corpus is the same one.** C4, the T5 tokenizer, 9.6B tokens. This list
#: used to carry "a C4 subset against their full stream", which sounds like a
#: difference and is not: their 9.6B is drawn from a corpus of roughly 156B, so
#: they consume about 6% of C4 and we consume about 6% of C4. Neither run sees
#: a "full stream". C4's shards are a deterministic partition of an already
#: shuffled crawl, with no ordering by source, date or quality, so the first 64
#: shards are exchangeable with any other 64 and a different slice would be a
#: different draw from one distribution rather than a better one. Tokenising the
#: remaining 960 shards would cost about 156 core-hours, 312 GB and several days
#: of depressed fairshare, and buy nothing measurable -- the spread it would
#: change is the same one two runs on different hardware already showed at
#: 0.002. Do not put it back.
KNOWN_DIFFERENCES = (
    "where the master weights sit: Megatron's optimizer wrapper keeps its own "
    "fp32 copies, while this harness holds fp32 parameters and autocasts the "
    "forward pass. Both clip one global norm over every parameter before the "
    "step, so this is largely the same arrangement described twice, and it is "
    "the least likely entry here to move a number",
    "layout: Megatron fuses QKV and fuses up with gate, where this harness "
    "keeps five separate matrices. Their optimizer slices the fused parameters "
    "apart again before rotating -- Q per head, K and V whole, up and gate "
    "apart -- so with Q per head (`train.query_blocks`) the geometry matches and only the "
    "storage layout differs",
)


def anchor_config(update_side: str = "bilateral", **overrides) -> RunConfig:
    """Their published-number configuration, not their shell script's.

    `bilateral` targets 3.3575, `alternate` targets 3.3654. Running both is
    worth the second run: reproducing the 0.0079 *gap* between them is a
    sharper check on the harness than reproducing either level, because the
    gap is insensitive to data and initialisation in a way the level is not.
    """
    if update_side not in TARGETS:
        raise ValueError(f"update_side must be one of {sorted(TARGETS)}, got {update_side!r}")
    settings = dict(
        optimizer="pion",
        lr=1e-3,
        lr_min=1e-5,
        warmup_steps=0,
        train_steps=73242,          # 9.6e9 / 512 / 256, as their script derives it
        batch_sequences=512,
        weight_decay=0.1,
        grad_clip=1.0,
        pion_scaling="rms",
        pion_rms=0.2,
        pion_momentum="lie",        # Lie+Lie: the variant the numbers come from
        pion_retraction="trunc",    # their degree-2 truncated exponential
        pion_alternate=update_side == "alternate",
        # Their script says --bf16. Running the anchor in fp32 would compare
        # our arithmetic against their number, which is a different experiment
        # from the one this is for.
        precision="bf16",
    )
    # Anything named by the caller wins, so an override is an override rather
    # than a collision.
    settings.update(overrides)
    return replace(RunConfig(), **settings)


def _last_attempt(log_path: str | Path) -> list[dict]:
    """Rows belonging to the most recent run in an appended log.

    The log is appended so a preempted run keeps its history, which means a
    re-run of the same configuration lands in the same file. Reading everything
    would mix attempts and report the wrong step count.
    """
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
    return rows[starts[-1] + 1 :] if starts else rows


def final_loss(log_path: str | Path, window: int = 10) -> float:
    """Training loss at the end of a run, averaged over the last logged points.

    A single logged value is one batch and is noisy; their figure is read off a
    curve. Averaging a short window is the closest honest equivalent.
    """
    rows = _last_attempt(log_path)
    losses = [r["train_loss"] for r in rows if "train_loss" in r]
    if not losses:
        raise ValueError(f"{log_path} contains no training loss")
    tail = losses[-window:]
    return sum(tail) / len(tail)


def last_step(log_path: str | Path) -> int:
    """Highest step of the most recent attempt, not of the whole file."""
    return max((r["step"] for r in _last_attempt(log_path) if "step" in r), default=-1)


def check(
    log_path: str | Path, update_side: str = "bilateral", expected_steps: int | None = None
) -> dict:
    """Compare a finished run against the published figure.

    Refuses to judge a run that did not finish. A truncated run always sits far
    above the target simply because it stopped early, and reporting that as a
    miss would put a meaningless verdict in the log for someone to trip over
    later. `expected_steps` is what makes the difference detectable.
    """
    target = TARGETS[update_side]
    got = final_loss(log_path)
    delta = got - target
    reached = last_step(log_path) + 1
    complete = expected_steps is None or reached >= expected_steps
    return {
        "update_side": update_side,
        "target": target,
        "measured": round(got, 4),
        "delta": round(delta, 4),
        "relative": round(delta / target, 5),
        "steps_run": reached,
        "steps_expected": expected_steps,
        "status": "complete" if complete else "incomplete",
        "matched": bool(complete and abs(delta) <= TOLERANCE),
        "tolerance": TOLERANCE,
    }
