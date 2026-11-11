#!/usr/bin/env python
"""Turn C4 into the flat token file the harness reads.

C4 is a cleaned Common Crawl dump -- roughly 156B tokens of English web text,
and the corpus Pion's 60M ablations use, which is the only reason it is the one
here: the comparison is only meaningful on the same data.

Output is one little-endian uint16 array per split. The T5 vocabulary is 32100
entries, so uint16 is exact and halves the file against uint32.

The budget is **9.6B tokens** -- 73242 steps at 131072 -- which is what the
paper reports and what `RunConfig` runs. An earlier reading of their shell
script gave 37500 steps, i.e. 4.9B; that came from a defect in their released
code, not from a second configuration, and it should not reappear here. The
default target is 10B, which gives a single pass with a margin.

Size the corpus to the budget deliberately: `TokenCorpus` samples random
windows with replacement and has no notion of an epoch, so a corpus smaller
than the budget does not fail or warn -- it silently shows every token more
than once, which is exactly the difference the anchor run is meant to detect.

**Download first, then tokenise.** Streaming the dataset over HTTP reached
129k tokens/s in one process and only 210k across eight, with the CPUs idle:
the cost is not bandwidth and not tokenising, it is the round trip on every
chunk the reader asks for. The same file on local disk tokenises at **785k
tokens/s on one core**, and the network itself delivers 23 MB/s in one stream
and 40 MB/s in four. So each shard fetches its files and works from disk, where
the job is CPU-bound and scales with cores.

C4 ships as 1024 files, and `--shards N` gives each process its own slice of
that list. One core per shard, no more shards than cores. On eight cores 10B
tokens is roughly forty minutes -- seven of downloading, half an hour of
tokenising -- against the thirteen to twenty hours the streaming path needed.

It resumes. Each shard records the file it is in, how far into that file it
read, and how many tokens it has written; re-running the same command picks up
there. Downloads are cached, so a resumed shard re-reads nothing it already
has.

Requires `datasets` and `transformers`, which the container installs and this
repository does not depend on -- nothing else here needs them.

    python scripts/prepare_data.py --out data --target-tokens 1e10 --shards 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _mark(fh, progress: Path, file_index: int, records_in_file: int, tokens: int) -> None:
    """Make what is written durable, then record how far it goes.

    The order is the whole point. The marker must never claim more than the
    file holds, or a resume would trust bytes that were never written; the
    reverse costs nothing, because resuming truncates the file back to the
    marker. The marker itself is replaced atomically for the same reason.

    The position is a file and an offset into it rather than a running count of
    documents, so resuming skips whole files instead of re-reading them.
    """
    fh.flush()
    os.fsync(fh.fileno())
    tmp = progress.with_name(progress.name + ".tmp")
    tmp.write_text(json.dumps(
        {"file_index": file_index, "records_in_file": records_in_file, "tokens": tokens}))
    os.replace(tmp, progress)


def fetch(dataset: str, filename: str, cache: Path, attempts: int = 7) -> Path:
    """Bring one dataset file to local disk, or return the copy already there.

    `hf_hub_download` resumes partial transfers and does nothing when the file
    is present, so a shard that restarts pays for no byte twice.

    Retries, because the Hub rate-limits. Thirty-two shards asking for download
    tokens at once earned a 429 and killed twenty of them outright; a 429 is a
    request to wait, not a failure, so it is waited out with a doubling delay
    and a little jitter so the retries do not re-synchronise. An HF_TOKEN
    raises the limits and makes this rare rather than routine.
    """
    from huggingface_hub import hf_hub_download

    delay = 5.0
    for attempt in range(attempts):
        try:
            return Path(hf_hub_download(dataset, filename, repo_type="dataset",
                                        local_dir=str(cache)))
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            transient = ("429" in text or "Too Many Requests" in text
                         or "Network error" in text or "timed out" in text
                         or isinstance(exc, (ConnectionError, TimeoutError, OSError)))
            if attempt == attempts - 1 or not transient:
                raise
            wait = delay * (1.0 + 0.5 * random.random())
            print(f"    [{filename}] {text.splitlines()[0][:80]} -- retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")


@contextmanager
def exclusive(out: Path, force: bool = False):
    """Refuse to run a second build into the same directory.

    Two builds sharing an output directory append to the same part files and
    interleave them, and nothing downstream would notice: the markers would
    still agree with the sizes, and the corpus would simply be wrong. It
    happened here -- a submission that looked cancelled had already reached the
    queue -- and was caught only because both jobs were still downloading.

    The lock carries the job id so a stale one says who left it.
    """
    lock = out / ".build.lock"
    owner = os.environ.get("SLURM_JOB_ID", f"pid {os.getpid()}")
    if force:
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(
            f"{lock} exists: another build is writing to {out} "
            f"({lock.read_text().strip()}). Two builds in one directory "
            f"interleave their parts silently. If that build is dead, pass "
            f"--force or remove the file."
        ) from None
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{owner}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def repo_files(dataset: str, subset: str, split: str) -> list[str]:
    """The dataset's own file list, which is what shards are cut along.

    Asked of the Hub rather than assembled from a guessed pattern: the layout
    is the dataset's business and has changed before.
    """
    from huggingface_hub import HfApi

    prefix = f"{subset}/c4-{split}"
    files = sorted(
        f for f in HfApi().list_repo_files(dataset, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".json.gz")
    )
    if not files:
        raise RuntimeError(f"no files matching {prefix}*.json.gz in {dataset}")
    return files


def write_split(path: Path, files: list[str], budget: float, args, label: str) -> int:
    """Fetch `files` one at a time and tokenise them from local disk.

    Local is the point. Read over HTTP the same work runs five times slower with
    the CPU idle, because every chunk the reader asks for costs a round trip.

    Returns the number of tokens the file holds when it stops.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    progress = path.with_name(path.name + ".progress.json")
    first_file, skip_records, written = 0, 0, 0

    if args.no_resume:
        path.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
    elif path.exists() and progress.exists():
        state = json.loads(progress.read_text())
        first_file = int(state["file_index"])
        skip_records = int(state["records_in_file"])
        written = int(state["tokens"])
        # uint16 on disk: two bytes a token. Anything past the marker was
        # written but never confirmed, so it goes.
        with path.open("r+b") as fh:
            fh.truncate(written * 2)
        print(f"[{label}] resuming in file {first_file} at {written/1e6:.1f}M tokens",
              flush=True)

    if written >= budget:
        print(f"[{label}] already holds {written/1e9:.3f}B tokens", flush=True)
        return written

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert tok.vocab_size < 65536, "uint16 storage assumes a vocabulary under 65536"

    done = False
    with path.open("ab") as fh:
        for index in range(first_file, len(files)):
            local = fetch(args.dataset, files[index], args.cache_dir)
            stream = load_dataset("json", data_files=str(local), split="train",
                                  streaming=True)
            consumed = skip_records
            if consumed:
                stream = stream.skip(consumed)
            skip_records = 0

            buffer, texts, since_sync = [], [], 0
            for record in stream:
                texts.append(record["text"])
                if len(texts) < args.batch:
                    continue
                for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                    buffer.extend(ids)
                    buffer.append(eos)
                np.asarray(buffer, dtype=np.uint16).tofile(fh)
                written += len(buffer)
                consumed += len(texts)
                buffer, texts = [], []
                since_sync += 1
                if since_sync >= args.sync_every:
                    _mark(fh, progress, index, consumed, written)
                    since_sync = 0
                    print(f"[{label}] {written/1e6:.1f}M tokens", flush=True)
                if written >= budget:
                    done = True
                    break

            # Stopping inside a file records where inside it; finishing one
            # records the start of the next, so a resume skips it whole.
            _mark(fh, progress, index if done else index + 1,
                  consumed if done else 0, written)
            if done:
                break
        else:
            print(f"[{label}] ran out of files at {written/1e9:.3f}B tokens", flush=True)

    print(f"[{label}] {path}: {written/1e9:.3f}B tokens", flush=True)
    return written


def concatenate(parts: list[Path], target: Path) -> int:
    """Join the shards into the single file the harness opens.

    Written beside the target and renamed, so an interrupted join leaves no
    half corpus wearing the real name. Each part is dropped as it is consumed,
    which keeps the peak at one part above the final size rather than double.
    """
    tmp = target.with_name(target.name + ".joining")
    total = 0
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                while chunk := fh.read(64 << 20):
                    out.write(chunk)
                    total += len(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, target)
    for part in parts:
        part.unlink(missing_ok=True)
        part.with_name(part.name + ".progress.json").unlink(missing_ok=True)
    return total // 2


def run_shards(args, files: list[str]) -> None:
    """One process per shard, each over its own slice of the file list."""
    out = args.out
    parts = [out / f"c4_train.part{k}.bin" for k in range(args.shards)]
    env = dict(os.environ)
    # One thread per process. N shards times M threads on N cores is thrash,
    # and the tokeniser is only a third of the work in any case.
    env.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    # The xet transfer path asks the Hub for a per-file token, and thirty-two
    # shards doing that at once is what earned a 429. Plain HTTP downloads at
    # the same speed -- 23.8 MB/s against 24.1 measured -- without the extra
    # API call.
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    children = []
    for k in range(args.shards):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--out", str(out), "--target-tokens", repr(args.target_tokens),
               "--val-tokens", repr(args.val_tokens), "--tokenizer", args.tokenizer,
               "--dataset", args.dataset, "--subset", args.subset,
               "--batch", str(args.batch), "--sync-every", str(args.sync_every),
               "--cache-dir", str(args.cache_dir),
               "--shards", str(args.shards), "--shard-index", str(k)]
        if args.no_resume:
            cmd.append("--no-resume")
        children.append(subprocess.Popen(cmd, env=env))

    failed = [k for k, c in enumerate(children) if c.wait() != 0]
    if failed:
        raise RuntimeError(
            f"shards {failed} failed; the rest kept their progress markers, so "
            "re-running the same command resumes instead of starting over"
        )

    tokens = concatenate(parts, out / "c4_train.bin")
    print(f"{out / 'c4_train.bin'}: {tokens/1e9:.3f}B tokens from {args.shards} shards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--target-tokens", type=float, default=1e10)
    ap.add_argument("--val-tokens", type=float, default=1e7)
    ap.add_argument("--tokenizer", default="t5-base")
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--subset", default="en")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=32,
                    help="batches between durable checkpoints; a crash costs at most this much")
    ap.add_argument("--force", action="store_true",
                    help="take the directory lock even if one is already there")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild from scratch, discarding any partial corpus")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="where dataset files are downloaded; defaults to <out>/downloads")
    ap.add_argument("--shards", type=int, default=1,
                    help="parallel worker processes over disjoint C4 files; one core each")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="internal: which slice this process owns")
    args = ap.parse_args()
    if args.shards < 1:
        ap.error("--shards must be at least 1")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = args.out / "downloads"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    listing = args.out / ".files.json"

    if args.shard_index is not None:
        # A child. It reads the file list the parent already fetched: asking
        # for it again is what cost us the corpus once. Listing this repository
        # is a recursive walk of eleven hundred files, and thirty-two children
        # doing it at once exhausted a quota of a thousand API requests per
        # five minutes, which arrived as a 429 and killed twenty shards.
        files = json.loads(listing.read_text())["train"]
        k, n = args.shard_index, args.shards
        share = args.target_tokens / n
        write_split(args.out / f"c4_train.part{k}.bin", files[k::n], share, args,
                    label=f"train {k}/{n}")
        return

    with exclusive(args.out, args.force):
        train_files = repo_files(args.dataset, args.subset, "train")
        val_files = repo_files(args.dataset, args.subset, "validation")
        listing.write_text(json.dumps({"train": train_files, "val": val_files}))
        _build(args, train_files, val_files)


def _build(args, train_files, val_files) -> None:
    write_split(args.out / "c4_val.bin", val_files, args.val_tokens, args, label="val")

    # A finished corpus is not rebuilt. The shards' own markers cover a build
    # interrupted partway, but they are removed once the parts are joined, so
    # without this a re-run of the same command after a successful build --
    # a requeued job, a repeated line in a script -- would silently start the
    # whole corpus again.
    final = args.out / "c4_train.bin"
    if not args.no_resume and final.exists() and final.stat().st_size // 2 >= args.target_tokens:
        print(f"{final}: already holds {final.stat().st_size / 2e9:.3f}B tokens")
    elif args.shards == 1:
        write_split(final, train_files, args.target_tokens, args, label="train")
    else:
        run_shards(args, train_files)


if __name__ == "__main__":
    main()
