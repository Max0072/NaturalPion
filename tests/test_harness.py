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


def _pion_optims():
    cfg = RunConfig(optimizer="pion")
    model = Transformer(cfg.model)
    return cfg, model, build_optimizers(model, cfg)[1]


def test_adam_betas_are_theirs_not_torchs_default():
    """Their script sets `--adam-beta2 0.95`; torch's AdamW default is 0.999.

    `build_optimizers` constructed AdamW without betas, so every run here used
    0.999 on the embedding, the output head and the norm gains -- 32.9M of this
    model's 58.2M parameters. Like the Pion betas before them, these were not
    configuration fields, so nothing recorded the choice.
    """
    _, _, adamw = _pion_optims()
    assert adamw.param_groups[0]["betas"] == (0.9, 0.95)
    assert adamw.param_groups[0]["eps"] == 1e-8


def test_norm_gains_and_biases_are_not_decayed():
    """Megatron gives `wd_mult = 0.0` to every 1-D parameter and every bias.

    This harness decayed them at 0.1, which over 73242 steps of the
    cosine multiplies a gain by 0.0248 -- a 40x shrink of a parameter that
    starts at 1.0. It matters more under Pion than it would elsewhere, because
    the usual compensation of growing the linear weights is unavailable when
    their spectra are frozen for the whole run.
    """
    cfg, _, adamw = _pion_optims()
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
    """`decay_norms_and_biases` decays every AdamW parameter, as this harness once did; it is not the default because
    it is not what their runs do."""
    cfg = replace(RunConfig(optimizer="pion"), decay_norms_and_biases=True)
    _, adamw, _ = build_optimizers(Transformer(cfg.model), cfg)
    assert len(adamw.param_groups) == 1
    assert adamw.param_groups[0]["weight_decay"] == cfg.weight_decay












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


def test_every_recorded_manifest_still_loads():
    """Fields removed from `RunConfig` must not make a recorded run unreadable: `--from-manifest` has to rebuild
    the configuration of every run in runs-record/."""
    root = Path(__file__).resolve().parents[1] / "runs-record"
    manifests = sorted(root.rglob("manifest.json"))
    assert manifests, "no run records found"
    for m in manifests:
        config = json.loads(m.read_text())["config"]
        if config["optimizer"] == "muon":       # Muon ran on a copy of this harness with fields of its own
            continue
        RunConfig.from_manifest(config)
