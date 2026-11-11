"""Copy a large, static, heavily-reread file to local disk before memory-mapping it.

Why (2026-09-22, docs/CLUSTER.md "rtx6002 wedges too"): `$DATA_p330` is a
`hard`-mounted NFS export from one server, and that server's local-cache
daemon (`cachefilesd`, the backend for the mount's `fsc` option) has been
dead since 2026-09-08 -- `SIGSEGV`, core-dumped, a root-owned service this
project cannot restart; reported to cluster admins. `hard` means a slow or
overloaded server hangs the client's read rather than erroring it, and the
thousands of scattered per-step reads a training loop makes against the
corpus (`TokenCorpus` samples random windows) are exactly the access pattern
that triggers it -- two multi-hour wedges in one evening, `0%` GPU
utilisation, page cache staying a fraction of the corpus size despite
hundreds of GB of free RAM.

`scripts/stage_corpus.sh` is the same idea as a standalone shell script, for
use outside Python (a training run's own read pattern is not the only thing
that can hit this). This module is the version wired into the harness itself,
so a run does not depend on whoever wrote its sbatch script remembering to
call the shell one -- `harness/train.py` runs it on every `TokenCorpus` path
by default. `RunConfig.local_cache_dir = ""` turns it off.
"""

from __future__ import annotations

import fcntl
import shutil
from pathlib import Path

__all__ = ["stage_locally"]

# Below this size, staging is pure overhead (c4_val.bin is 16 MB) -- only pay
# the one-time copy cost for a file worth reading from scattered thousands of
# times afterwards. c4_train.bin is ~15 GB; the one-time copy took 32 s from
# the login node and up to ~45 min from a loaded compute node (docs/CLUSTER.md)
# -- worth it either way against a training loop that reads the file thousands
# of times, but budget minutes, not seconds, when this runs for the first time
# on a given node.
_MIN_SIZE_TO_STAGE = 500_000_000  # 500 MB


def stage_locally(path: str, cache_dir: str) -> str:
    """Copy `path` into `cache_dir` if it is large and `cache_dir` is usable.

    Returns the local path on success, or `path` unchanged if `cache_dir` is
    empty, the source is missing or too small to be worth it, or the local
    disk turns out to be unwritable -- callers always get a path back, never
    an exception from this function; a genuinely missing file is left for the
    real `open()`/`np.memmap()` call to report.

    Idempotent and safe under concurrent callers on one machine (a SLURM array
    sharing a node): an `flock` on a lock file in `cache_dir` serialises the
    copy, so an eight-task array copies the corpus once, not eight times, and
    a task that finds an exact byte-size match returns immediately.
    """
    if not cache_dir:
        return path
    src = Path(path)
    try:
        size = src.stat().st_size
    except OSError:
        return path
    if size < _MIN_SIZE_TO_STAGE:
        return path

    dst_dir = Path(cache_dir)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        with open(dst_dir / ".stage.lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                dst = dst_dir / src.name
                if dst.exists() and dst.stat().st_size == size:
                    return str(dst)
                # Resume a `.partial` left by a run that hit its time limit
                # mid-copy rather than restart -- found necessary 2026-09-22,
                # when the NFS server backing $DATA_p330 was slow enough that
                # a 20 GB copy did not finish inside a job's own time limit.
                # `shutil.copyfile` has no resume; append from the partial
                # file's own size instead, which is exact only if nothing
                # else is writing the same `.partial` concurrently -- true
                # here because the surrounding flock is still held.
                tmp = dst.with_suffix(dst.suffix + ".partial")
                resume_at = tmp.stat().st_size if tmp.exists() else 0
                if resume_at > size:
                    resume_at = 0  # a stale partial from a different source
                with open(src, "rb") as fsrc, open(tmp, "r+b" if resume_at else "wb") as fdst:
                    fsrc.seek(resume_at)
                    fdst.seek(resume_at)
                    shutil.copyfileobj(fsrc, fdst)
                if tmp.stat().st_size != size:
                    return path  # short read (source shrank, disk full); do not trust it
                tmp.rename(dst)
                return str(dst)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
    except OSError:
        # cache_dir unwritable, out of local disk space, or some other
        # problem with the destination -- fall back to the original path
        # rather than fail a run over an optimisation.
        return path
