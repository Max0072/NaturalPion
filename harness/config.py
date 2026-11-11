"""Run configuration, and the identity of a run.

Every field that can change a result lives here, and the hash of this object
is the run's name. Two runs with the same hash saw the same configuration; two
runs with the same hash and the same seed should see the same batches. That is
what makes a sweep auditable after the fact rather than a folder of guesses.

Defaults follow `opt_llama_60M_pion.sh`. Their script derives the step count
rather than stating it -- `TOKEN=9.6`, `GLOBAL_BATCH=512`, and
`TRAIN_ITER = TOTAL_TOKENS / GLOBAL_BATCH / 256` -- which gives 73242 steps at
131072 tokens each. 9.6B tokens is 8x the Chinchilla-optimal budget for a 60M
model, and their comment line `# 1: 1.2; 2: 2.4; 4: 4.8; 8: 9.6;` is that
multiplier table.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field

from ngd_pion.natural_pion import BY_NAME, default_of

from .model import ModelConfig

__all__ = ["RunConfig", "git_commit", "machine", "untracked_modules"]

def git_commit() -> str:
    """The working tree's commit, or a marker when it is not a repository.

    `-dirty` means a **tracked** file differs from the commit -- the code that
    ran is not the code the hash names, which is the only thing this marker is
    for. Untracked files are deliberately excluded. They used to count, and the
    consequence was that a notes file created beside the repository stamped two
    anchor runs `-dirty` while the code they executed was clean; a marker that
    fires on things which cannot change a result stops meaning anything exactly
    when it needs to.

    The exception is an untracked *module*, which a run can import and which
    would therefore be invisible here. `untracked_modules` reports those
    separately rather than folding them into a flag that says nothing about
    which file it means.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip()[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"

def untracked_modules() -> list[str]:
    """Untracked `.py` files, which `git_commit` does not count as dirty.

    An untracked module is the one kind of untracked file a run can actually
    execute, so it belongs in the manifest even though it does not belong in
    `-dirty`. Normally empty; a non-empty list is worth reading before trusting
    the commit hash beside it.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        return sorted(f for f in out if f.endswith(".py"))
    except Exception:
        return []

def machine(device: str | None = None) -> dict:
    """Which card, which host, which torch. Recorded, never hashed."""
    import platform

    import torch

    info = {
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device) if device is not None else None,
    }
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["arch"] = f"sm_{cap[0]}{cap[1]}"
    return info

@dataclass(frozen=True)
class RunConfig:
    # what is being compared
    optimizer: str = "natural-pion-fast"   # natural-pion-fast (the method) | natural-pion (angle-rule reference) | pion | adamw
    lr: float = 1e-3
    # AdamW's learning rate. `0` means "follow `lr`", which is what their
    # published configuration does -- one rate for both optimizers, 1e-3 --
    # so the anchor and every configuration written before this field existed
    # are unaffected.
    #
    # Setting it is what any statement about the *rotational* optimizer's
    # learning rate requires. Tied together, raising `lr` also raises AdamW's
    # rate on the embedding, the output head, the norm gains and the biases,
    # and AdamW detonates there long before the rotation does anything at all.
    # The `eta` sweep recorded in docs/history/JOURNAL.md before 2026-08-26 varied both
    # at once and measured AdamW.
    adamw_lr: float = 0.0
    seed: int = 0

    # the model, as their script configures it
    model: ModelConfig = field(default_factory=ModelConfig)

    # schedule, as their script configures it
    batch_sequences: int = 512      # global batch; 512 * 256 = 131072 tokens/step
    train_steps: int = 73242        # TOTAL_TOKENS / GLOBAL_BATCH / 256 = 9.6e9/512/256
    lr_min: float = 1e-5
    # The schedule's floor as a fraction of the peak (each optimizer's own); overrides `lr_min` when set. Pion's
    # 1e-5 at lr 1e-3 is 0.01. An absolute floor means a different decay depth for every peak. Out of the hash when unset.
    lr_min_ratio: float | None = 0.01
    # 2: NaturalPionFast's defaults are the method and lr_min_ratio defaults to 0.01 (2026-11-11); manifests without
    # this field are version 1 (see from_manifest)
    defaults_version: int = 2
    warmup_steps: int = 0
    # Shape of the rotation optimizer's schedule (AdamW always keeps the cosine): cosine | constant | linear |
    # warmcos (linear up over the first 20%, then cosine) | wsd (flat, then linear to lr_min over the last
    # `wsd_decay` of the steps). Out of the hash at the defaults.
    schedule: str = "cosine"
    wsd_decay: float = 0.2
    # Natural Pion: time each part of the optimizer step (GPU-synchronised) into profile.json. Slows the run.
    np_profile: bool = False
    # Natural Pion Fast: the angle of some matrices times a factor, by name, e.g. '{"wv": 2, "wo": 2}' (a diagnostic of
    # which matrices want a larger step). Out of the hash when empty.
    np_angle_mult: str = ""
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Arithmetic for the *model*. On an Ampere-or-newer card torch runs fp32
    # matrix operations in TF32 -- ten bits of mantissa -- and on an RTX PRO
    # 6000 Blackwell that is worth 2.2x to 2.6x on exactly the shapes this
    # model multiplies, measured. The gradient it produces is then wrong by a
    # relative 1e-3, against a step that is about 47% sampling noise, and
    # Megatron runs their own experiments in bf16, which is coarser still. So
    # it is on.
    #
    # It is emphatically **not** on inside the optimizer. TF32 there moves the
    # singular values of a weight by a relative 1.0 over 200 two-sided steps
    # (2.6e-04 without), which would silently destroy the one property the
    # method is built on, and it corrupts the covariance the method inverts.
    # `ngd_pion` turns it off around its own linear algebra and does not
    # depend on this field being anything in particular.
    #
    # A scientific field regardless, not plumbing: it changes results, so it
    # is in the hash.
    tf32: bool = True

    # Autocast for the forward pass and the loss. `bf16` is what their own runs
    # use -- `--bf16` in opt_llama_60M_pion.sh -- so it is the setting that
    # makes the anchor a fair test rather than a faster one.
    #
    # It defaults to bf16 because everything that actually runs is bf16:
    # `anchor_config` sets it, and every throughput measurement on record
    # passed `--precision bf16` explicitly. The default said `fp32` for a while
    # after precision stopped being a known difference from their setup, and
    # the mismatch was invisible until jobs 246662/246663 -- a plain
    # `train.sbatch` invocation with no `--precision` -- died allocating a
    # 15.67 GiB logit tensor, which is 512x256x32100 in fp32 against 8.4 GiB in
    # bf16. Nothing had launched a full run through that path before, so no
    # measurement caught it.
    #
    # Weights and the optimizer stay fp32 regardless: autocast changes what the
    # operations compute in, not what the parameters are, which is the same
    # arrangement as Megatron's fp32 master weights. The covariance is fed from
    # activations that are then bf16, and that is measured to be harmless --
    # thousands of independent roundings cancel in the average, perturbing `A`
    # by 4.5e-05, well under the 1e-4 spectral floor. Storing `A` in bf16 would
    # not be, and `CovarianceAccumulator` refuses to.
    precision: str = "bf16"          # fp32 | bf16

    # Natural Pion: the relative floor on every inverted spectrum
    ngd_eps: float = 1e-3
    # Draw the labels for `E[dd^T]` from the model instead of from the data,
    # every this many steps; `0` keeps the empirical Fisher (the true-Fisher ablation of Section 4.4).
    #
    # This is the difference between a curvature matrix and a noise covariance.
    # `E[gg^T]` equals `E[grad^2 l]` only when the labels come from the model's
    # own predictive distribution; with the data's labels it is the *empirical*
    # Fisher, whose scale is set by how much per-sample gradients disagree
    # rather than by how curved the loss is. Measured here, the per-component
    # signal-to-noise `|E[g]| / sqrt(E[g^2])` peaks at 0.035, so `E[g^2]`
    # exceeds `E[g]^2` by nearly three orders and `F^-1 G` is inflated by the
    # same amount -- which is why `eta*` sits three orders below the 2 the
    # theory gives for the true Fisher.
    #
    # `D` is an EMA, so the extra pass need not run every step.
    ngd_fisher_mc_every: int = 0
    ngd_beta1: float = 0.9
    # How often the reduction ratio is measured, independently of logging. `0`
    # keeps the old behaviour of measuring it only on logged steps.
    #
    # This mattered and nobody noticed: `rho` is what the Levenberg-Marquardt
    # rule in `damped.py` adapts on, and it was tied to `log_every`. At
    # `log_every = 100` the rule fired once every hundred steps, and the run
    # that judged it a dead end was 150 steps long -- so the mechanism adapted
    # perhaps five times in its entire evaluation. kfac-jax uses an interval of
    # **5**. One extra forward pass every `rho_every` steps is the whole cost.
    rho_every: int = 0
    # Sequences per chunk in the reduction-ratio forward. `0` evaluates it in
    # one piece, which is what it always did. It runs after the optimizer step,
    # so the optimizer's persistent state is resident and the head's
    # `vocab x tokens` logits are 8.4 GB on top -- enough to OOM a 96 GiB card
    # at a measured 82.6 GB peak, which is what happened to `rho_every` on rtx
    # while it ran fine on b200. Lowering `rho_every` does not help: one
    # evaluation is already too large.
    rho_micro: int = 0
    # Measure the reduction ratio on a *fresh* batch as well as on the batch
    # whose gradient produced the step, and log it as `rho_held`.
    #
    # `rho` as measured is evaluated on the batch it was fit to, so it asks
    # whether the quadratic model described *that* batch. A step several times
    # too long still describes it, because it is descending that batch's own
    # quadratic -- and the step here is 96.4% sampling noise, so the batch's
    # quadratic and the expected one are not the same object. That is the
    # obvious candidate for why a controller targeting `rho ~ 1` settles two to
    # four times above the swept optimum: the quantity it steers by is blind by
    # construction to the error that makes the step too long.
    #
    # Costs two extra chunked forwards per measurement, on a second reader over
    # the training corpus offset by its seed. **No longer diagnostic-only**
    # (2026-09-22): when this is on, `rho_held` is what `adapt_damping` steers
    # on instead of the same-batch `rho` -- see `harness/train.py` where it is
    # read. Turning `rho_holdout` on therefore changes the trajectory of any
    # run with `ngd_trust_lr` also on, not only what gets logged.
    rho_holdout: bool = False
    ngd_ns_iters: int = 2
    # Natural Pion: Newton-Schulz takes as many iterations as the angle needs for fp32 exactness
    # (natural_pion.ns_iterations); the exact solve only above this angle, where it converges too slowly.
    ngd_ns_guard: float = 1.8

    # vanilla Pion
    pion_scaling: str = "rms"
    pion_rms: float = 0.2
    pion_momentum: str = "lie"      # the variant their published 60M numbers use
    pion_retraction: str = "trunc"
    # Bilateral, which is their better published number for this model -- 3.3575
    # against 3.3654 for alternate -- and the variant NGD-Pion matches, since it
    # rotates both sides every step. This default said `True` for a while with
    # nothing defending it, which would have put Pion in a variant the arm it is
    # compared with does not use. The baseline has to be Pion at its best.
    #
    # `anchor_config` sets this from `update_side` and does not read the
    # default, so the anchors were never affected either way.
    pion_alternate: bool = False
    pion_beta1: float = 0.9
    pion_beta2: float = 0.95
    # Their `--pion-use-second-momentum` is `store_true` with `default=None`,
    # and `_use_second_momentum` falls through to `False` when it is absent.
    # `opt_llama_60M_pion.sh` passes it only when `USE_SECOND_MOMENTUM` is set
    # in the environment, so their published runs normalise the Lie momentum by
    # its first moment alone.
    #
    # This harness divided by `sqrt(v) + 1e-8` regardless, because
    # `Pion.__init__` defaults `beta2=0.95` and `build_optimizers` never passed
    # a value. Worse, neither beta was a field here, so the choice was outside
    # the hash: two runs differing in it produced the same name and shared a
    # directory. Both are fields now, and the second moment is off by default
    # because that is what their number comes from.
    pion_second_moment: bool = False

    # AdamW, for everything Pion does not own -- the embedding, the output head
    # and the norm gains, which are 32.9M of this model's 58.2M parameters.
    # `build_optimizers` used to construct `torch.optim.AdamW` without betas and
    # take torch's `(0.9, 0.999)`, while their script sets `--adam-beta2 0.95`
    # (and `--pion-beta2 0.95` beside it, so 0.95 either way). Like the Pion
    # betas before them, these were not fields, so the choice sat outside the
    # configuration hash and no manifest recorded it.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Megatron's standard override gives `wd_mult = 0.0` to every parameter with
    # `len(shape) == 1` or a name ending in `.bias`, so their RMSNorm gains are
    # never decayed. This harness applied `weight_decay` to all of `rest`, and
    # at this schedule that multiplies a gain by 0.0248 over the run -- a
    # **40x shrink** of a parameter that starts at 1.0.
    #
    # It matters more here than it would elsewhere. A network normally absorbs
    # decayed norm gains by growing the linear weights, and under Pion that
    # route is closed: the spectrum of every rotated matrix is frozen for the
    # whole run. The only compensation left is the embedding and the head.
    decay_norms_and_biases: bool = False

    # Natural Pion: damping of the Fisher relative to its largest eigenvalue (0 = floor only). Out of the hash at 0.
    np_damping: float = 0.0
    # Natural Pion: how the Cayley inverse is computed -- "cayley" exact solve, "ns" Newton-Schulz (biased in fp32),
    # "ns_delta" Newton-Schulz in Z - I, "trunc" Pion's truncated exponential -- and "bf16" or "fp32" inputs for the
    # grams of A and D. None: the optimizer class's own default (NaturalPion: cayley, fp32; NaturalPionFast: ns_delta,
    # bf16). The hash records the value in effect, out of it at cayley / fp32, so earlier names keep their meaning.
    np_retraction: str | None = None
    np_stats_dtype: str | None = None
    # Natural Pion: extra optimizer keyword arguments as JSON (the ablation switches of NaturalPionFast). Out of the hash
    # when empty.
    np_kwargs: str = ""

    # plumbing, deliberately excluded from the hash
    data_path: str = "data/c4_train.bin"
    val_path: str = "data/c4_val.bin"
    # Copy data_path/val_path to local disk before memory-mapping them, if they
    # are at least 500 MB (see harness/local_cache.py: $DATA_p330 is `hard`-
    # mounted NFS whose local-cache daemon has been dead since 2026-09-08, and
    # a training loop's scattered per-step reads are what triggers the
    # multi-hour hangs this works around). "" disables it and reads data_path/
    # val_path directly, as before this existed.
    local_cache_dir: str = "/tmp/c4"
    eval_every: int = 500
    eval_batches: int = 20
    log_every: int = 50
    # Gradient accumulation chunk. Excluded from the hash: it changes speed and
    # memory, not the result.
    #
    # 512 removes accumulation entirely -- one micro-batch is the whole step --
    # and lets the covariance see all 131k tokens of a step rather than a slice.
    # Measured on an RTX PRO 6000 Blackwell (96 GB) in job 246613, on warm page
    # cache: pion 0.460 s/step and ngd-pion 0.722 at 512, against 0.869 and
    # 0.673 at 256. So 512 is the right default for Pion by a wide margin, and
    # for NGD-Pion the two are close enough that seeing the full step's tokens
    # decides it.
    #
    # It fits only with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which
    # every GPU sbatch script in scripts/sbatch now sets. Without that flag
    # ngd-pion OOMs at 512 -- not for lack of room (385 MB of optimizer state
    # against 14.6 GB of headroom) but because the unbatched `_apply` leaves the
    # allocator unable to serve 7-30 MB contiguous blocks. Run outside those
    # scripts and you must set it yourself.
    micro_batch: int = 512
    out_dir: str = "runs"

    _EXCLUDED = ("data_path", "val_path", "local_cache_dir", "eval_every", "eval_batches",
                 "log_every", "micro_batch", "out_dir", "np_profile")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_sequences * self.model.seq_len

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.train_steps

    @classmethod
    def from_manifest(cls, config: dict) -> "RunConfig":
        """The configuration a run's manifest.json records. Fields since removed (the per-head flags of 2026-11-05:
        Q is always rotated per head now; the unused ngd_*/shampoo_* fields of 2026-11-11) are dropped; any other
        unknown field is an error. A manifest without `defaults_version` was written under the old defaults, which
        are restored so the run is reproduced as it ran: no lr_min_ratio, and NaturalPionFast's spectral length rule,
        tf32 off and fp32 F^-1 wherever its np_kwargs did not set them."""
        removed = {"np_split_q_per_head", "pion_split_q_per_head"} | set(['ngd_alpha_max', 'ngd_angle', 'ngd_angle_max', 'ngd_beta', 'ngd_beta_backward', 'ngd_exact_beta', 'ngd_exact_every', 'ngd_exact_tokens', 'ngd_lam', 'ngd_lam_adapt', 'ngd_momentum', 'ngd_power', 'ngd_retraction', 'ngd_s_power', 'ngd_t_fac', 'ngd_trust', 'ngd_trust_lr', 'ngd_trust_lr_cap', 'ngd_trust_lr_hi', 'ngd_trust_lr_lo', 'ngd_use_s', 'shampoo_beta', 'shampoo_damping', 'shampoo_eps', 'shampoo_plane_every', 'shampoo_power'])
        d = {k: v for k, v in config.items() if k not in removed}
        if "defaults_version" not in d:
            d["defaults_version"] = 1
            d.setdefault("lr_min_ratio", None)
            if d.get("optimizer") == "natural-pion-fast":
                kw = json.loads(d.get("np_kwargs") or "{}")
                for k, v in (("length", "spectral"), ("tf32", "off"), ("fisher_dtype", "fp32")):
                    kw.setdefault(k, v)
                d["np_kwargs"] = json.dumps(kw, separators=(",", ":"))
        return cls(model=ModelConfig(**d.pop("model")), **d)

    def scientific_fields(self) -> dict:
        """Everything that can change the result, and nothing that cannot."""
        d = asdict(self)
        for key in self._EXCLUDED:
            d.pop(key, None)
        if not d.get("np_damping"):
            d.pop("np_damping", None)
        if d.get("schedule") == "cosine":
            d.pop("schedule", None)
        if d.get("schedule") != "wsd":
            d.pop("wsd_decay", None)
        if self.optimizer in BY_NAME:                    # None means the class's default: hash what is in effect
            for field, kw in (("np_retraction", "retraction"), ("np_stats_dtype", "stats_dtype")):
                if d.get(field) is None:
                    d[field] = default_of(BY_NAME[self.optimizer], kw)
        if d.get("np_retraction") in ("cayley", None):
            d.pop("np_retraction", None)
        if d.get("np_stats_dtype") in ("fp32", None):
            d.pop("np_stats_dtype", None)
        if not d.get("np_angle_mult"):
            d.pop("np_angle_mult", None)
        if d.get("lr_min_ratio") is None:
            d.pop("lr_min_ratio", None)
        if not d.get("np_kwargs"):
            d.pop("np_kwargs", None)
        return d

    @property
    def hash(self) -> str:
        blob = json.dumps(self.scientific_fields(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    @property
    def name(self) -> str:
        return f"{self.optimizer}-lr{self.lr:g}-s{self.seed}-{self.hash}"

    def manifest(self, device: str | None = None) -> dict:
        """What gets written beside the results so a run can be reconstructed.

        The machine is part of that. Two cards do not agree bitwise -- cuBLAS
        picks different algorithms per architecture, floating-point addition is
        not associative, and some backward kernels accumulate atomically in no
        fixed order -- so a comparison whose arms ran on different hardware has
        a variable in it that nothing in the configuration records. Writing the
        device down does not prevent that; it makes it answerable afterwards.
        """
        return {
            "name": self.name,
            "hash": self.hash,
            "git_commit": git_commit(),
            "untracked_modules": untracked_modules(),
            "tokens_per_step": self.tokens_per_step,
            "total_tokens": self.total_tokens,
            "machine": machine(device),
            "config": asdict(self),
        }
