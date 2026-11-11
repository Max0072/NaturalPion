"""The training loop.

One process, one GPU, one run. There is no distributed anything here on
purpose: a 60M model fits comfortably on a single card, so N GPUs are worth
far more as N concurrent experiments than as one faster experiment, and the
sweeps this project needs are wide rather than deep.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from ngd_pion.hooks import attach, attach_backward
from ngd_pion.pion_baseline import Pion
from ngd_pion.linalg import EIGH_FALLBACKS
from ngd_pion.natural_pion import BY_NAME, NaturalPion

from .config import RunConfig
from .data import TokenCorpus
from .local_cache import stage_locally
from .model import Transformer

__all__ = ["build_optimizers", "lr_at", "train"]

def query_blocks(model, cfg: RunConfig) -> dict[int, int]:
    """The query projection rotates as one block per attention head (Pion's `qkv_split_granularity="head"`; K and V
    are separate matrices here already). Part of the model's geometry, the same for every rotating optimizer."""
    return {id(m.weight): cfg.model.heads for m in model.query_projections()}


def lr_at(step: int, cfg: RunConfig, base: float | None = None, shape: str = "cosine") -> float:
    """Cosine decay to `lr_min`. Their script warms up over zero steps.

    `base` overrides the peak so a second optimizer can share the schedule's
    *shape* without sharing its scale. The floor is scaled by the same factor,
    which keeps `lr_min / lr` -- the depth of the decay -- identical between
    them; a fixed absolute floor would otherwise mean something quite different
    to a rate three hundred times smaller.
    """
    peak = cfg.lr if base is None else base
    if cfg.lr_min_ratio is not None:
        floor = cfg.lr_min_ratio * peak
    else:
        floor = cfg.lr_min * (peak / cfg.lr) if cfg.lr else cfg.lr_min
    if step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if shape == "cosine":
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
    if shape == "constant":
        return peak
    if shape == "linear":
        return floor + (peak - floor) * (1.0 - progress)
    if shape == "warmcos":
        if progress < 0.2:
            return peak * (step + 1) / (0.2 * cfg.train_steps)
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * (progress - 0.2) / 0.8))
    if shape == "wsd":
        start = 1.0 - cfg.wsd_decay
        return peak if progress < start else floor + (peak - floor) * (1.0 - progress) / cfg.wsd_decay
    raise ValueError(f"unknown schedule {shape!r}")


# The rotating optimizers: Natural Pion's tiers (ngd_pion/natural_pion.py) and the Pion baseline.
NATURAL_PION = BY_NAME


def build_optimizers(model: Transformer, cfg: RunConfig):
    """Split the parameters the way Pion does, and give each half an optimizer.

    Returns `(rotational_opt, adamw_opt, recorder)`. `rotational_opt` is
    `None` for the plain AdamW baseline, in which case AdamW owns everything,
    and `recorder` is `None` for every optimizer that needs no activations.
    """
    linears, rest = model.parameter_split()
    weights = [m.weight for m in linears]

    if cfg.optimizer == "adamw":
        return None, _adamw(model.parameters(), cfg), None

    if cfg.optimizer in NATURAL_PION:
        # The method (natural-pion-fast) or the angle-rule reference (natural-pion). Only the scale, the momentum,
        # the floor and the switches below come from the configuration; cadence, statistics window and factor EMAs are the class's own, so the legacy
        # `ngd_*` defaults cannot leak in. `--lr` is the peak of the step scale (the RMS rule's target).
        rot = NATURAL_PION[cfg.optimizer](weights, lr=cfg.lr, beta1=cfg.ngd_beta1, row_blocks=query_blocks(model, cfg),
                                          eps=cfg.ngd_eps, damping=cfg.np_damping,
                                          ns_iters=cfg.ngd_ns_iters, ns_guard=cfg.ngd_ns_guard,
                                          **{k: v for k, v in (("retraction", cfg.np_retraction),
                                                               ("stats_dtype", cfg.np_stats_dtype)) if v is not None},
                                          **(json.loads(cfg.np_kwargs) if cfg.np_kwargs else {}))
        recorder = attach(linears, rot)
        if cfg.np_profile:
            rot.profile = {}
    elif cfg.optimizer == "pion":
        # Published Pion, every setting from `RunConfig` -- the baseline, and the
        # only Pion arm.
        rot = Pion(
            weights, lr=cfg.lr,
            scaling=cfg.pion_scaling,
            rms=cfg.pion_rms,
            momentum=cfg.pion_momentum,
            retraction=cfg.pion_retraction,
            alternate=cfg.pion_alternate,
            beta1=cfg.pion_beta1,
            beta2=cfg.pion_beta2 if cfg.pion_second_moment else None,
            row_blocks=query_blocks(model, cfg),
        )
        recorder = None
    else:
        raise ValueError(
            f"unknown optimizer {cfg.optimizer!r}; expected one of "
            + ", ".join(
                repr(k) for k in (
                    *NATURAL_PION, "pion", "adamw"
                )
            )
        )

    adamw = _adamw(rest, cfg)
    return rot, adamw, recorder


def _adamw(params, cfg: RunConfig) -> torch.optim.AdamW:
    """AdamW for the parameters Pion does not own, in their configuration.

    Two groups, because Megatron's standard override exempts every 1-D
    parameter and every bias from weight decay and this harness did not. See
    `RunConfig.decay_norms_and_biases` for why that is worse under a
    spectrum-preserving optimizer than under an ordinary one.
    """
    betas = (cfg.adam_beta1, cfg.adam_beta2)
    if cfg.decay_norms_and_biases:
        groups = [{"params": list(params), "weight_decay": cfg.weight_decay}]
    else:
        params = list(params)
        flat = [p for p in params if p.dim() <= 1]
        rest = [p for p in params if p.dim() > 1]
        groups = [
            {"params": rest, "weight_decay": cfg.weight_decay},
            {"params": flat, "weight_decay": 0.0},
        ]
    return torch.optim.AdamW(groups, lr=cfg.adamw_lr or cfg.lr, betas=betas, eps=cfg.adam_eps)


def _autocast(cfg: RunConfig, device: str):
    """Autocast for the forward pass, or a no-op context.

    Only the forward pass and the loss. The optimizer step runs outside it,
    under `no_grad` and with its own exact-fp32 guard, so the retraction and
    the covariance keep the precision they need whatever this is set to.
    """
    if cfg.precision == "fp32":
        return nullcontext()
    if cfg.precision != "bf16":
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {cfg.precision!r}")
    return torch.autocast("cuda" if str(device).startswith("cuda") else "cpu",
                          dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model: Transformer, batches, cfg: RunConfig, device: str) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        with _autocast(cfg, device):
            total += float(model(x, y)[1])
    model.train()
    return total / max(1, len(batches))


class RunLock:
    """One writer per run directory, with a heartbeat so a resume is not blocked.

    A run's directory is named by the configuration hash, so the same
    configuration on two machines is the same directory -- and two trainers
    appending to one log and overwriting one checkpoint produce a run that
    looks finished and is not. Nothing downstream could detect it.

    The lock is refreshed at every checkpoint, roughly every four minutes at
    this scale, so a lock older than the grace period belonged to something
    that died. That is deliberately taken over rather than refused: the cluster
    caps jobs at 24 hours and resubmitting to resume is the intended workflow,
    which a lock left behind by a SIGKILL would otherwise break.
    """

    def __init__(self, path: Path, grace: float = 900.0) -> None:
        self.path = path
        self.grace = grace
        self.owner = (f"{platform.node()} pid {os.getpid()} "
                      f"slurm {os.environ.get('SLURM_JOB_ID', '-')}")

    def take(self, force: bool = False) -> None:
        if self.path.exists() and not force:
            age = time.time() - self.path.stat().st_mtime
            if age < self.grace:
                raise SystemExit(
                    f"{self.path} is held by {self.path.read_text().strip()} "
                    f"({age:.0f}s ago). Two trainers in one run directory "
                    f"interleave their logs and overwrite each other's "
                    f"checkpoints. Use a different --out-dir, or --force if "
                    f"that process is gone."
                )
            print(f"taking over a lock {age/60:.0f} min stale, left by "
                  f"{self.path.read_text().strip()}", flush=True)
        self.path.write_text(self.owner + "\n")

    def touch(self) -> None:
        try:
            self.path.write_text(self.owner + "\n")
        except OSError:
            pass

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def _save(path: Path, step: int, model, rot, adamw, data) -> None:
    """Write the checkpoint, atomically.

    A job killed while `torch.save` is writing -- the 24 h wall, a preemption,
    a node failure -- leaves a truncated file where the resume path expects a
    checkpoint, and the run then has to start over from nothing. Writing beside
    it and renaming avoids that: `os.replace` is atomic within a filesystem, so
    whatever checkpoint exists is a complete one.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "rot": rot.state_dict() if rot is not None else None,
            "adamw": adamw.state_dict(),
            "data_rng": data.rng_state,
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def _restore(path: Path, model, rot, adamw, data) -> int:
    """Reload a checkpoint, returning the step to continue from."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if rot is not None and state["rot"] is not None:
        rot.load_state_dict(state["rot"])
    adamw.load_state_dict(state["adamw"])
    data.rng_state = state["data_rng"]
    torch.set_rng_state(state["torch_rng"])
    return state["step"] + 1


def train(
    cfg: RunConfig, device: str = "cpu", max_steps: int | None = None, resume: bool = True,
    force: bool = False,
) -> Path:
    """Train one run, picking up from a checkpoint if one is already there.

    Resuming by default is what the cluster wants: every partition caps at 24
    hours, a full 9.6B run may not fit inside that, and a requeued job should
    continue rather than start over. The sampler's position is restored too, so
    a resumed run does not replay the batches it already saw.
    """
    torch.manual_seed(cfg.seed)
    # Applies to the model's own matmuls as well as the optimizer's, which
    # guards itself in any case. Off by default: see RunConfig.tf32.
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out / ".run.lock")
    lock.take(force)
    (out / "manifest.json").write_text(json.dumps(cfg.manifest(device), indent=2))
    # Appending rather than truncating keeps a preempted run's history, but it
    # also means a re-run with the same configuration writes into the same
    # file. A start marker separates them so readers can take the last attempt.
    log = (out / "log.jsonl").open("a")
    log.write(json.dumps({"event": "start", "steps": max_steps or cfg.train_steps,
                          "time": time.time()}) + "\n")
    log.flush()

    model = Transformer(cfg.model).to(device)
    rot, adamw, recorder = build_optimizers(model, cfg)
    # Natural Pion reads the output gradients as a statistic (D), so it gets a backward recorder beside the
    # activation one.
    backward = attach_backward(model.parameter_split()[0], rot) if isinstance(rot, NaturalPion) else None

    # Local disk, not $DATA_p330, if the file is large enough to be worth the
    # copy -- see harness/local_cache.py for why.
    local_data_path = stage_locally(cfg.data_path, cfg.local_cache_dir)
    local_val_path = stage_locally(cfg.val_path, cfg.local_cache_dir)

    train_data = TokenCorpus(local_data_path, cfg.model.seq_len, seed=cfg.seed)
    val_data = TokenCorpus(local_val_path, cfg.model.seq_len, seed=cfg.seed + 1)
    val_batches = val_data.fixed_batches(cfg.micro_batch, cfg.eval_batches, seed=1234, device=device)

    steps = max_steps or cfg.train_steps
    accum = max(1, cfg.batch_sequences // cfg.micro_batch)
    started = time.time()

    # A corpus smaller than the budget is not an error here and never will be:
    # windows are sampled with replacement and there is no epoch, so the run
    # would quietly show the same tokens twice. Log the ratio instead, where a
    # reader can see it.
    passes = train_data.epochs_for(steps * cfg.tokens_per_step)
    log.write(json.dumps({"event": "corpus", "tokens": len(train_data),
                          "passes": round(passes, 3)}) + "\n")
    log.flush()

    first = 0
    checkpoint = out / "checkpoint.pt"
    if resume and checkpoint.exists():
        first = _restore(checkpoint, model, rot, adamw, train_data)
        log.write(json.dumps({"event": "resume", "from_step": first}) + "\n")
        log.flush()
        if first >= steps:
            log.close()
            lock.release()
            if recorder is not None:
                recorder.remove()
            if backward is not None:
                backward.remove()
            return out

    window_time, window_step = time.time(), first
    for step in range(first, steps):
        lr = lr_at(step, cfg, shape=cfg.schedule)
        # Two schedules when `adamw_lr` is set, one when it is not. Sharing the
        # rate is their published design and stays the default; decoupling is
        # what makes a sweep over `lr` a sweep over the rotational optimizer
        # rather than over AdamW.
        adamw_lr = lr_at(step, cfg) if not cfg.adamw_lr else lr_at(step, cfg, cfg.adamw_lr)
        if rot is not None:
            for group in rot.param_groups:
                group["lr"] = lr
        for group in adamw.param_groups:
            group["lr"] = adamw_lr

        model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            # one micro-batch per step feeds the covariance; see ActivationRecorder
            if recorder is not None:
                recorder.enabled = micro == 0
            if backward is not None:
                backward.enabled = micro == 0
            x, y = train_data.batch(cfg.micro_batch, device)
            if (
                backward is not None
                and cfg.ngd_fisher_mc_every
                and micro == 0
                and step % cfg.ngd_fisher_mc_every == 0
            ):
                # One extra pass whose labels are drawn from the model, which
                # is what makes E[dd^T] the true Fisher rather than the
                # empirical one. The activation recorder stays off so `A` is
                # not fed twice, and the gradients this leaves behind are
                # discarded -- only `D` is wanted from it.
                if recorder is not None:
                    recorder.enabled = False
                backward.enabled = True
                with _autocast(cfg, device):
                    logits, _ = model(x)
                flat = logits.reshape(-1, logits.shape[-1])
                with torch.no_grad():
                    # In chunks, and never in fp32 across the whole batch:
                    # `flat.float()` alone is 131072 x 32100 x 4 = 16.8 GB here,
                    # and the softmax another, against a step that already
                    # peaks at 83 of 95 GB.
                    drawn = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
                    for i in range(0, flat.shape[0], 8192):
                        piece = flat[i : i + 8192].float().softmax(-1)
                        drawn[i : i + 8192] = torch.multinomial(piece, 1).squeeze(-1)
                        del piece
                mc = F.cross_entropy(flat, drawn)
                mc.backward()
                model.zero_grad(set_to_none=True)
                # the graph is gone with the backward; drop the activations too,
                # or the real forward starts on top of them
                del logits, flat, drawn, mc
                backward.enabled = False
                if recorder is not None:
                    recorder.enabled = True
            with _autocast(cfg, device):
                _, loss = model(x, y)
            # D comes from this backward at micro == 0 and so carries the 1/accum (see BackwardRecorder); the
            # true-Fisher pass above does not. Neither matters under the RMS rule.
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if rot is not None:
            rot.step()
        adamw.step()

        if step % cfg.log_every == 0 or step == steps - 1:
            now = time.time()
            elapsed = now - started
            # Throughput decides which partition a run belongs on: a 60M model
            # does not fill a B200, so the peak-FLOPS ratio against an RTX card
            # is not the ratio that matters. Two numbers, because the average
            # over a whole attempt carries its startup and warm-up with it and
            # reads low on a short run -- and step 1 is a 200-step run whose
            # answer allocates three thousand GPU-hours. `window` is the rate
            # since the previous logged row; take that one.
            done_here = step - first + 1
            span, moved = now - window_time, step - window_step
            row = {"step": step, "lr": lr, "train_loss": loss_sum,
                   "wall": round(elapsed, 1),
                   "tokens_per_sec": round(cfg.tokens_per_step * done_here / max(elapsed, 1e-9)),
                   "tokens_per_sec_window": round(
                       cfg.tokens_per_step * moved / max(span, 1e-9)) if moved else None}
            if str(device).startswith("cuda"):
                # Peak since the last log line, not since the run began: what
                # decides `micro_batch` is what a step needs, and the head
                # materialises vocab-by-tokens logits, so the answer is not
                # obvious from the parameter count.
                row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                torch.cuda.reset_peak_memory_stats()
            window_time, window_step = now, step
            # Zero is the expected reading. Anything else says the pencil given
            # to `eigh` is badly enough conditioned that the primary solver is
            # giving up, which is worth seeing in the log rather than inferring
            # from a crash that did not happen.
            if any(EIGH_FALLBACKS.values()):
                row["eigh_fallbacks"] = dict(EIGH_FALLBACKS)
            # realised rotation angle per matrix, where a length rule other than the spectral one sets it
            got = rot.realised_angles() if hasattr(rot, "realised_angles") else None
            angles = sorted(got.values()) if got else []
            if angles:
                q = lambda f: angles[min(len(angles) - 1, int(f * len(angles)))]
                row["angle_p10"], row["angle_p50"], row["angle_p90"] = q(0.1), q(0.5), q(0.9)
            log.write(json.dumps(row) + "\n")
            log.flush()

        if (step + 1) % cfg.eval_every == 0 or step == steps - 1:
            row = {"step": step, "val_loss": evaluate(model, val_batches, cfg, device)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            _save(checkpoint, step, model, rot, adamw, train_data)
            lock.touch()

    log.close()
    if rot is not None and getattr(rot, "profile", None):
        (out / "profile.json").write_text(json.dumps({"steps": steps - first, "seconds": rot.profile}, indent=1))
    lock.release()
    if recorder is not None:
        recorder.remove()
    if backward is not None:
        backward.remove()
    return out
