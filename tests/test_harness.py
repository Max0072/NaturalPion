"""The training harness: model shapes, data, schedule, and one end-to-end run."""

import json
import math
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from harness.config import RunConfig
from harness.data import TokenCorpus
from harness.model import ModelConfig, Transformer
from harness.train import (
    _measure_loss,
    _recorders_off,
    build_optimizers,
    lr_at,
    train,
)
from ngd_pion.natural_pion import NaturalPion, NaturalPionFast
from ngd_pion.pion_baseline import Pion

SMALL = ModelConfig(vocab_size=256, hidden=64, layers=2, heads=2, ffn_hidden=176, seq_len=32)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    for name in ("train", "val"):
        rng.integers(0, 256, size=200_000, dtype=np.uint16).tofile(d / f"{name}.bin")
    return d


def test_published_configuration_gives_the_published_size():
    """Their script says 60M; the configuration in model.py has to land there."""
    model = Transformer(ModelConfig())
    total = sum(p.numel() for p in model.parameters())
    assert 55e6 < total < 62e6, f"{total/1e6:.1f}M is not the 60M their script describes"


def test_parameter_split_follows_their_rule():
    """2-D weights that are neither the embedding nor the output head."""
    model = Transformer(ModelConfig())
    linears, rest = model.parameter_split()
    owned = {id(m.weight) for m in linears}
    assert id(model.embed.weight) not in owned
    assert id(model.head.weight) not in owned
    assert all(p.dim() == 2 for p in (m.weight for m in linears))
    assert len(linears) == 7 * ModelConfig().layers
    assert sum(p.numel() for p in rest) + sum(m.weight.numel() for m in linears) == sum(
        p.numel() for p in model.parameters()
    )


def test_block_matrix_shapes_are_the_three_the_spec_predicts():
    """Four square, two tall, one wide -- which decides each layer's basis path."""
    model = Transformer(ModelConfig())
    linears, _ = model.parameter_split()
    shapes = Counter(tuple(m.weight.shape) for m in linears[:7])
    assert shapes == Counter({(512, 512): 4, (1376, 512): 2, (512, 1376): 1})


def test_forward_starts_near_uniform_entropy():
    cfg = ModelConfig()
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, loss = model(x, x)
    assert abs(float(loss) - np.log(cfg.vocab_size)) < 0.5


def test_lr_schedule_is_cosine_without_warmup():
    cfg = RunConfig(lr=1e-3, lr_min=1e-5, train_steps=1000, warmup_steps=0)
    assert lr_at(0, cfg) == pytest.approx(1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-5, rel=1e-3)
    mid = lr_at(500, cfg)
    assert lr_at(750, cfg) < mid < lr_at(250, cfg)


def test_config_hash_covers_science_and_ignores_plumbing():
    cfg = RunConfig()
    assert cfg.hash == RunConfig().hash
    for changed in (dict(lr=2e-3), dict(seed=1), dict(optimizer="pion"), dict(ngd_eps=1e-4)):
        assert replace(cfg, **changed).hash != cfg.hash
    for ignored in (dict(out_dir="x"), dict(eval_every=7), dict(micro_batch=8)):
        assert replace(cfg, **ignored).hash == cfg.hash


def test_token_budget_matches_their_script():
    cfg = RunConfig()
    assert cfg.tokens_per_step == 512 * 256
    assert cfg.total_tokens == 73242 * 131072  # 9.6B, i.e. 8x Chinchilla for 60M
    assert abs(cfg.total_tokens - 9.6e9) / 9.6e9 < 1e-4


def test_corpus_targets_are_inputs_shifted_by_one(corpus):
    data = TokenCorpus(corpus / "train.bin", seq_len=16, seed=0)
    x, y = data.batch(4)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_corpus_is_deterministic_given_a_seed(corpus):
    a = TokenCorpus(corpus / "train.bin", 16, seed=3).batch(4)[0]
    b = TokenCorpus(corpus / "train.bin", 16, seed=3).batch(4)[0]
    assert torch.equal(a, b)


def test_held_out_batches_do_not_move(corpus):
    data = TokenCorpus(corpus / "val.bin", 16, seed=0)
    first = data.fixed_batches(2, 3, seed=99)
    second = data.fixed_batches(2, 3, seed=99)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))


def test_missing_corpus_says_what_to_run():
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenCorpus("nowhere.bin", 16)


@pytest.mark.parametrize(
    "optimizer,expected",
    [("natural-pion-fast", NaturalPionFast), ("natural-pion", NaturalPion),
     ("pion", Pion), ("adamw", None)],
)
def test_optimizer_wiring(optimizer, expected):
    model = Transformer(SMALL)
    rot, adamw, recorder = build_optimizers(model, RunConfig(optimizer=optimizer, model=SMALL))
    assert rot is None if expected is None else isinstance(rot, expected)
    assert isinstance(adamw, torch.optim.AdamW)
    assert (recorder is not None) == optimizer.startswith("natural-pion")
    if recorder is not None:
        recorder.remove()


def test_pion_is_wired_as_published():
    """The baseline is Pion as published: RMS scaling, Lie momentum, truncated exp."""
    model = Transformer(SMALL)
    rot, _, _ = build_optimizers(model, RunConfig(optimizer="pion", model=SMALL))
    group = rot.param_groups[0]
    assert group["scaling"] == "rms"
    assert group["momentum"] == "lie"
    assert group["retraction"] == "trunc"


def test_pion_ablated_is_gone():
    """ADR 0008 and 0011: there is no ablated Pion, and asking for one fails loudly."""
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizers(Transformer(SMALL), RunConfig(optimizer="pion_ablated", model=SMALL))


@pytest.mark.parametrize("optimizer", ["natural-pion-fast", "natural-pion", "pion", "adamw"])
def test_run_end_to_end_and_reduce_loss(optimizer, corpus, tmp_path):
    cfg = RunConfig(
        optimizer=optimizer, model=SMALL, batch_sequences=8, micro_batch=4,
        train_steps=20, eval_every=10, eval_batches=2, log_every=5,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=20)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if "train_loss" in r]
    assert train_rows[-1]["train_loss"] < train_rows[0]["train_loss"]
    assert any("val_loss" in r for r in rows)
    assert (out / "manifest.json").exists()
    assert (out / "checkpoint.pt").exists()


def test_anchor_follows_the_published_number_not_the_shell_script():
    """Their 60M script runs a different configuration than their reported figures."""
    from harness.anchor import TARGETS, anchor_config

    cfg = anchor_config("bilateral")
    assert cfg.optimizer == "pion"
    assert cfg.pion_momentum == "lie", "the published figures use Lie+Lie"
    assert cfg.pion_alternate is False, "3.3575 is the bilateral number"
    assert cfg.pion_scaling == "rms" and cfg.pion_rms == 0.2
    assert cfg.pion_retraction == "trunc", "their degree-2 truncated exponential"
    assert cfg.lr == 1e-3 and cfg.lr_min == 1e-5 and cfg.warmup_steps == 0
    assert cfg.total_tokens == 73242 * 131072
    assert TARGETS["bilateral"] == 3.3575

    other = anchor_config("alternate")
    assert other.pion_alternate is True
    assert TARGETS["alternate"] == 3.3654
    assert cfg.hash != other.hash


def test_anchor_tolerance_is_tighter_than_the_gap_it_must_resolve():
    """A tolerance looser than their own bilateral-alternate gap would say nothing."""
    from harness.anchor import TARGETS, TOLERANCE

    gap = TARGETS["alternate"] - TARGETS["bilateral"]
    assert 0 < TOLERANCE < 3 * gap


def test_anchor_check_reads_a_log(tmp_path):
    from harness.anchor import check

    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"step": i, "train_loss": 3.36}) for i in range(20)))
    result = check(log, "bilateral")
    assert result["matched"] and result["status"] == "complete"
    assert result["measured"] == pytest.approx(3.36)

    log.write_text(json.dumps({"step": 0, "train_loss": 4.0}))
    assert not check(log, "bilateral")["matched"]


def test_anchor_refuses_to_judge_a_truncated_run(tmp_path):
    """A short run always sits above a converged target; that is not a miss."""
    from harness.anchor import check

    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"step": i, "train_loss": 8.0}) for i in range(6)))
    result = check(log, "bilateral", expected_steps=73242)
    assert result["status"] == "incomplete"
    assert result["matched"] is False
    assert result["steps_run"] == 6 and result["steps_expected"] == 73242

    on_target = tmp_path / "full.jsonl"
    on_target.write_text("\n".join(json.dumps({"step": i, "train_loss": 3.357}) for i in range(10)))
    assert check(on_target, "bilateral", expected_steps=10)["status"] == "complete"


def test_anchor_rejects_an_unknown_update_side():
    from harness.anchor import anchor_config

    with pytest.raises(ValueError, match="update_side"):
        anchor_config("sideways")


def test_cli_flags_parse_to_their_declared_types(monkeypatch):
    """`from __future__ import annotations` turns field.type into a string.

    Resolving the annotations is not cosmetic: without it every flag parses as
    str, and the failure surfaces deep inside numpy rather than at the parse.
    """
    import sys as _sys
    from typing import get_type_hints

    from harness import run as run_module

    captured = {}
    monkeypatch.setattr(run_module, "train", lambda cfg, **kw: captured.setdefault("cfg", cfg) or Path("."))
    monkeypatch.setattr(
        _sys, "argv",
        ["run.py", "--optimizer", "pion", "--lr", "3e-4", "--micro-batch", "8",
         "--train-steps", "5", "--pion-alternate", "false"],
    )
    run_module.main()
    cfg = captured["cfg"]
    hints = get_type_hints(RunConfig)
    for name in ("lr", "micro_batch", "train_steps", "pion_alternate", "seed", "grad_clip"):
        assert isinstance(getattr(cfg, name), hints[name]), f"{name} parsed as the wrong type"
    assert cfg.lr == 3e-4 and cfg.micro_batch == 8 and cfg.pion_alternate is False


def test_anchor_reads_only_the_most_recent_attempt(tmp_path):
    """The log is appended, so a re-run of the same config shares the file.

    Without a start marker the check reports the previous attempt's step count,
    which would let a short re-run be judged against a long one's rows.
    """
    from harness.anchor import check, final_loss, last_step

    log = tmp_path / "log.jsonl"
    rows = [{"event": "start", "steps": 100}]
    rows += [{"step": i, "train_loss": 9.0} for i in range(50)]
    rows += [{"event": "start", "steps": 100}]
    rows += [{"step": i, "train_loss": 3.357} for i in range(3)]
    log.write_text("\n".join(json.dumps(r) for r in rows))

    assert last_step(log) == 2, "must not see the earlier attempt's steps"
    assert final_loss(log) == pytest.approx(3.357)
    assert check(log, "bilateral", expected_steps=100)["steps_run"] == 3


def test_training_writes_a_start_marker(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=3, eval_every=100, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=3)
    train(cfg, max_steps=2)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert sum(r.get("event") == "start" for r in rows) == 2


def test_a_run_resumes_from_its_checkpoint(corpus, tmp_path):
    """Every partition caps at 24h and a full run may not fit, so a requeued
    job has to continue rather than start over."""
    cfg = RunConfig(
        optimizer="natural-pion-fast", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=8, eval_every=2, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=4)
    first = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 3
    assert first["rot"] is not None, "the rotational optimizer's state must be kept"

    train(cfg, max_steps=8)
    second = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert second["step"] == 7

    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert any(r.get("event") == "resume" and r["from_step"] == 4 for r in rows)
    steps = [r["step"] for r in rows if "train_loss" in r]
    assert max(steps) == 7 and 0 in steps


def test_resume_is_a_no_op_when_already_finished(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=3, eval_every=1, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=3)
    before = (out / "log.jsonl").read_text().count("train_loss")
    train(cfg, max_steps=3)
    assert (out / "log.jsonl").read_text().count("train_loss") == before


def test_no_resume_starts_over(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=4, eval_every=2, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=2)
    train(cfg, max_steps=2, resume=False)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert not any(r.get("event") == "resume" for r in rows)


def test_bf16_autocast_trains_and_leaves_the_weights_in_fp32(corpus, tmp_path):
    """Their runs are --bf16, so ours must be able to be.

    Autocast covers the forward pass and the loss; the parameters stay fp32,
    which is the same arrangement as Megatron's fp32 master weights, and the
    optimizer keeps its own exact arithmetic underneath. This runs on CPU, so
    it pins the wiring rather than the numerics.
    """
    cfg = RunConfig(
        optimizer="natural-pion-fast", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=4, eval_every=2, eval_batches=1, log_every=1,
        precision="bf16",
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=4)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    losses = [r["train_loss"] for r in rows if "train_loss" in r]
    assert losses and all(math.isfinite(v) for v in losses)
    assert any("val_loss" in r for r in rows)

    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    dtypes = {v.dtype for v in state["model"].values() if torch.is_floating_point(v)}
    assert dtypes == {torch.float32}, f"parameters must stay fp32, got {dtypes}"


def test_precision_is_part_of_a_run_s_identity():
    """It changes results, so two precisions are two runs, not one."""
    assert RunConfig(precision="fp32").hash != RunConfig(precision="bf16").hash


def test_an_unknown_precision_is_refused(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=1, eval_every=99, log_every=1, precision="fp8",
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="precision"):
        train(cfg, max_steps=1)


def test_a_second_trainer_will_not_share_a_run_directory(corpus, tmp_path):
    """Same configuration, same directory -- and two writers ruin both.

    A run's directory is its configuration hash, so the same run on two
    machines lands in one place, appends to one log and overwrites one
    checkpoint. The result looks finished. Nothing downstream could tell.
    """
    from harness.train import RunLock

    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=2, eval_every=99, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    held = RunLock(out / ".run.lock")
    held.take()

    with pytest.raises(SystemExit, match="held by"):
        train(cfg, max_steps=2)

    # --force is the way past it, for when the holder is known to be gone.
    train(cfg, max_steps=2, force=True)


def test_a_stale_lock_does_not_block_a_resume(corpus, tmp_path):
    """The cluster caps jobs at 24 h and resubmitting is how a run continues.

    A lock left behind by a SIGKILL must not stand in the way of that, so one
    older than the grace period is taken over rather than obeyed.
    """
    from harness.train import RunLock

    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=2, eval_every=99, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    RunLock(lock).take()
    os.utime(lock, (time.time() - 3600, time.time() - 3600))

    train(cfg, max_steps=2)          # takes it over, loudly, rather than refusing
    assert not lock.exists(), "a finished run releases its lock"


def _corpus(tmp_path, tokens=10_000, seq_len=8, seed=0):
    from harness.data import TokenCorpus
    path = tmp_path / "toy.bin"
    np.arange(tokens, dtype=np.uint16).tofile(path)
    return TokenCorpus(path, seq_len=seq_len, seed=seed)


def test_an_epoch_visits_every_window_exactly_once(tmp_path):
    """Their sample index partitions the stream; a shuffle index orders it.

    This used to draw window starts uniformly *with replacement*, which is a
    different distribution over training data rather than a different order of
    the same data -- at 0.959 passes some windows arrive several times and
    others never. Reading `gpt_dataset.py` settled it: their samples are
    contiguous non-overlapping slices, permuted once per epoch.
    """
    c = _corpus(tmp_path)
    starts = Counter()
    for _ in range(c.windows):
        starts[int(c.batch(1)[0][0, 0])] += 1
    assert len(starts) == c.windows
    assert set(starts.values()) == {1}


def test_the_order_is_shuffled_and_not_sequential(tmp_path):
    """A partition alone is not enough -- the permutation has to be there."""
    c = _corpus(tmp_path)
    first = [int(c.batch(1)[0][0, 0]) for _ in range(20)]
    assert first != sorted(first)


def test_the_next_epoch_reshuffles(tmp_path):
    """Two epochs must not replay the same order, or the second pass is the first."""
    c = _corpus(tmp_path)
    one = [int(c.batch(1)[0][0, 0]) for _ in range(c.windows)]
    two = [int(c.batch(1)[0][0, 0]) for _ in range(c.windows)]
    assert sorted(one) == sorted(two)
    assert one != two


def test_resume_continues_the_permutation(tmp_path):
    """The sampler's position is an epoch and an offset, and it round-trips.

    The permutation is rebuilt from `(seed, epoch)` rather than checkpointed:
    39 million windows would be 313 MB of `int64` in every checkpoint.
    """
    c = _corpus(tmp_path)
    for _ in range(7):
        c.batch(3)
    saved = c.rng_state
    expected = c.batch(3)[0]

    d = _corpus(tmp_path)
    d.rng_state = saved
    assert torch.equal(d.batch(3)[0], expected)


def test_a_pre_partition_checkpoint_is_refused_rather_than_misread(tmp_path):
    """A bit-generator state belongs to the sampler this class no longer is.

    Accepting it would resume a run into a different data distribution from the
    one it started in, and nothing downstream could detect that.
    """
    c = _corpus(tmp_path)
    with pytest.raises(ValueError, match="bit-generator"):
        c.rng_state = np.random.default_rng(0).bit_generator.state


def _anchor_optims():
    from harness.anchor import anchor_config
    from harness.model import Transformer
    from harness.train import build_optimizers
    cfg = anchor_config("bilateral")
    model = Transformer(cfg.model)
    return cfg, model, build_optimizers(model, cfg)[1]


def test_adam_betas_are_theirs_not_torchs_default():
    """Their script sets `--adam-beta2 0.95`; torch's AdamW default is 0.999.

    `build_optimizers` constructed AdamW without betas, so every run here used
    0.999 on the embedding, the output head and the norm gains -- 32.9M of this
    model's 58.2M parameters. Like the Pion betas before them, these were not
    configuration fields, so nothing recorded the choice.
    """
    _, _, adamw = _anchor_optims()
    assert adamw.param_groups[0]["betas"] == (0.9, 0.95)
    assert adamw.param_groups[0]["eps"] == 1e-8


def test_norm_gains_and_biases_are_not_decayed():
    """Megatron gives `wd_mult = 0.0` to every 1-D parameter and every bias.

    This harness decayed them at 0.1, which over 73242 steps of the anchor's
    cosine multiplies a gain by 0.0248 -- a 40x shrink of a parameter that
    starts at 1.0. It matters more under Pion than it would elsewhere, because
    the usual compensation of growing the linear weights is unavailable when
    their spectra are frozen for the whole run.
    """
    cfg, _, adamw = _anchor_optims()
    by_wd = {g["weight_decay"]: g["params"] for g in adamw.param_groups}
    assert set(by_wd) == {0.0, cfg.weight_decay}
    assert all(p.dim() <= 1 for p in by_wd[0.0])
    assert all(p.dim() > 1 for p in by_wd[cfg.weight_decay])
    assert sum(p.numel() for p in by_wd[0.0]) == 17 * cfg.model.hidden


def test_a_norm_gain_holds_still_while_a_matrix_decays():
    """The split has to bite in the step, not only in the group bookkeeping.

    Zero gradients so Adam's own term is `0 / (0 + eps) = 0` and only the
    decoupled decay `p *= 1 - lr*wd` can move anything. `lr` has to be nonzero
    for that reason -- decoupled decay is proportional to it, so an `lr` of 0
    decays nothing and the test would pass on a broken split.
    """
    from dataclasses import replace
    from harness.config import RunConfig
    from harness.train import _adamw
    cfg = replace(RunConfig(), lr=0.1, weight_decay=0.5)
    gain = nn.Parameter(torch.ones(4))
    mat = nn.Parameter(torch.ones(4, 4))
    opt = _adamw([gain, mat], cfg)
    for p in (gain, mat):
        p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(gain.detach(), torch.ones(4)), "a 1-D parameter must not decay"
    assert torch.allclose(mat.detach(), torch.full((4, 4), 1 - cfg.lr * cfg.weight_decay)), \
        "a matrix must decay by exactly 1 - lr*wd"


def test_the_old_behaviour_is_still_reachable():
    """`decay_norms_and_biases` exists so the four completed anchors remain
    reproducible; it is not the default because it is not what their runs do."""
    from dataclasses import replace
    from harness.anchor import anchor_config
    from harness.model import Transformer
    from harness.train import build_optimizers
    cfg = replace(anchor_config("bilateral"), decay_norms_and_biases=True)
    _, adamw, _ = build_optimizers(Transformer(cfg.model), cfg)
    assert len(adamw.param_groups) == 1
    assert adamw.param_groups[0]["weight_decay"] == cfg.weight_decay


def test_normal_init_is_unchanged_by_the_new_flag():
    """Every run on disk used it, so the default must be bit-identical."""
    torch.manual_seed(0)
    a = Transformer(SMALL)
    torch.manual_seed(0)
    b = Transformer(replace(SMALL, init="normal"))
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


def test_orthogonal_init_gives_a_flat_spectrum_at_the_gain():
    """The point of it: Pion freezes the spectrum, so this one is permanent."""
    model = Transformer(replace(SMALL, init="orthogonal", init_gain=1.0))
    rotational, _ = model.parameter_split()
    for m in rotational:
        s = torch.linalg.svdvals(m.weight.detach())
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), (s.min(), s.max())


def test_orthogonal_gain_scales_the_whole_spectrum():
    model = Transformer(replace(SMALL, init="orthogonal", init_gain=0.5))
    m = model.parameter_split()[0][0]
    s = torch.linalg.svdvals(m.weight.detach())
    assert torch.allclose(s, 0.5 * torch.ones_like(s), atol=1e-5)


def test_xavier_init_has_the_glorot_bound():
    model = Transformer(replace(SMALL, init="xavier"))
    m = model.parameter_split()[0][0]
    w = m.weight.detach()
    fan_out, fan_in = w.shape
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    assert w.abs().max() <= bound * (1 + 1e-6)
    assert w.abs().max() > 0.8 * bound


def test_the_embedding_and_head_keep_the_normal_init():
    """The spectral condition is derived for matmul layers; an embedding is a
    lookup table, and both belong to AdamW which can move their spectra anyway."""
    torch.manual_seed(0)
    a = Transformer(SMALL)
    torch.manual_seed(0)
    b = Transformer(replace(SMALL, init="orthogonal"))
    assert torch.equal(a.embed.weight, b.embed.weight)
    assert torch.equal(a.head.weight, b.head.weight)


def test_two_ways_to_choose_a_spectrum_are_refused_together():
    with pytest.raises(ValueError):
        Transformer(replace(SMALL, init="orthogonal", init_pl_alpha=3.0))
    with pytest.raises(ValueError):
        Transformer(replace(SMALL, init="hilbert"))


def test_residual_flag_changes_the_block_and_defaults_on():
    """Off, a block is a plain composition; on, it is the usual pre-norm sum."""
    x = torch.randn(2, 8, SMALL.hidden)
    for flag in (True, False):
        torch.manual_seed(0)
        m = Transformer(replace(SMALL, residual=flag))
        blk = m.blocks[0]
        out = blk(x, m.cos[:8], m.sin[:8])
        composed = blk.ffn(blk.ffn_norm(blk.attn(blk.attn_norm(x), m.cos[:8], m.sin[:8])))
        if flag:
            assert not torch.allclose(out, composed, atol=1e-6)
        else:
            assert torch.allclose(out, composed, atol=1e-6)


def test_residual_default_is_bit_identical_to_before():
    torch.manual_seed(0)
    a = Transformer(SMALL)
    torch.manual_seed(0)
    b = Transformer(replace(SMALL, residual=True))
    x = torch.randint(0, SMALL.vocab_size, (2, 8))
    with torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])


def test_orthogonal_weights_survive_a_cayley_run():
    """The premise of the residual-free experiment: the spectrum is frozen, so
    an orthogonal initialisation is orthogonal for the whole run, not only at
    step zero."""
    torch.manual_seed(0)
    m = Transformer(replace(SMALL, init="orthogonal", residual=False))
    w = m.parameter_split()[0][0].weight
    before = torch.linalg.svdvals(w.detach().double())
    p = torch.nn.Parameter(w.detach().clone())
    opt = NaturalPion([p], lr=0.05, t_fac=3)
    x = torch.randn(64, p.shape[1])
    d = torch.randn(64, p.shape[0]) * 0.3
    for _ in range(6):
        opt.observe(p, x)
        opt.observe_backward(p, d)
        p.grad = torch.randn_like(p) * 0.1
        opt.step()
    after = torch.linalg.svdvals(p.detach().double())
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-7)


def test_diagnostic_forward_leaves_the_covariance_alone():
    """A measurement forward must not feed `A`, and for four days it did.

    `ActivationRecorder` hangs off `register_forward_pre_hook`, and a pre-hook
    fires under `torch.no_grad()` like any other, so the forward that evaluates
    `rho` was folding its activations into the covariance as though it were a
    training batch. `CovarianceAccumulator.observe` blends with a fixed
    `1 - beta` **per call**, not per sample, so a `rho_micro = 128` chunk of a
    512-sequence batch carried a full training batch's EMA weight from a
    quarter of the tokens -- four such calls every five steps at
    `rho_every = 5`, which is 44% of `A`'s weight taken at post-step weights.

    Nine runs were affected and one conclusion did not survive it: the trust
    region's 0.20 deficit was measured with the controller arms at 44% and the
    fixed-rate control at 1%, so the arms differed in two things at once.

    Both halves are asserted, because the guard is only meaningful if the thing
    it guards against would otherwise happen.
    """
    cfg = RunConfig(model=SMALL, optimizer="natural-pion-fast", lr=1e-2, micro_batch=4)
    model = Transformer(SMALL)
    rot, _, recorder = build_optimizers(model, cfg)
    assert recorder is not None, "Natural Pion records activations; this test needs one"

    x = torch.randint(0, SMALL.vocab_size, (4, SMALL.seq_len))
    y = torch.randint(0, SMALL.vocab_size, (4, SMALL.seq_len))
    _, loss = model(x, y)
    loss.backward()

    def covariances():
        out = {}
        for group in rot.param_groups:
            for p in group["params"]:
                acc = rot.state.get(p, {}).get("cov")
                if acc is not None and acc.ready:
                    out[id(p)] = acc.matrix.clone()
        return out

    before = covariances()
    assert before, "no covariance was accumulated, so the test cannot see the bug"

    with _recorders_off(recorder):
        _measure_loss(model, x, y, 0, cfg, "cpu")
    guarded = covariances()
    for key, was in before.items():
        assert torch.equal(was, guarded[key]), "a guarded diagnostic forward moved `A`"

    _measure_loss(model, x, y, 0, cfg, "cpu")
    unguarded = covariances()
    assert any(
        not torch.equal(before[key], unguarded[key]) for key in before
    ), "an unguarded forward left `A` untouched, so the guard is pinning nothing"


def test_rotation_schedules():
    from harness.config import RunConfig
    from harness.train import lr_at
    cfg = RunConfig(lr=0.2, train_steps=1000, schedule="wsd", wsd_decay=0.2, lr_min_ratio=None)   # absolute lr_min
    for shape, start, mid, end in (("cosine", 0.2, 0.1, 1e-5), ("constant", 0.2, 0.2, 0.2),
                                   ("linear", 0.2, 0.1, 1e-5), ("wsd", 0.2, 0.2, 1e-5)):
        got = [lr_at(t, cfg, shape=shape) for t in (0, 500, 1000)]
        assert abs(got[0] - start) < 1e-3 and abs(got[1] - mid) < 1e-3 and abs(got[2] - end) < 1e-3, (shape, got)
    assert lr_at(0, cfg, shape="warmcos") < 0.01 and abs(lr_at(200, cfg, shape="warmcos") - 0.2) < 1e-3
    assert "schedule" not in RunConfig().scientific_fields() and "wsd_decay" in cfg.scientific_fields()


def test_angle_multiplier_reaches_the_named_matrices():
    from harness.config import RunConfig
    from harness.train import build_optimizers
    from harness.model import Transformer
    model = Transformer(SMALL)
    rot, _, rec = build_optimizers(model, RunConfig(optimizer="natural-pion-fast", model=SMALL, np_angle_mult='{"wv": 2}'))
    vs = {id(m.weight) for n, m in model.named_modules() if n.endswith(".wv")}
    assert set(rot.angle_mult) == vs and all(f == 2 for f in rot.angle_mult.values()) and len(vs) == SMALL.layers
    rec.remove()
