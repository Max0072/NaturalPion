"""Token stream for pretraining.

The corpus is one flat array of token ids on disk, read through a memmap, and
batches are random windows into it. Nothing here knows about C4 specifically;
`scripts/prepare_data.py` is what turns a corpus into this format.

Megatron uses its own indexed binary format with a document index, a sample
index and a shuffle index. Reading `gpt_dataset.py` rather than assuming:
a sample there is a contiguous slice of the *concatenated* document stream and
is assembled from several documents when it straddles them, and their 60M
script passes none of `--reset-position-ids`, `--reset-attention-mask` or
`--eod-mask-loss`, all of which default off. So their windows cross document
boundaries exactly as these do, and the difference this file used to claim --
"Megatron respects document boundaries" -- was not real.

What *is* different is the sampling discipline, and this file now matches it.
Their samples partition the stream into non-overlapping windows which a
shuffle index then permutes, so each token is seen once per epoch. This used to
draw window starts uniformly **with replacement**, which sees some windows
several times and others never, and at 0.959 passes over the corpus that is a
different distribution over training data rather than a different order of the
same data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["TokenCorpus"]


class TokenCorpus:
    """A memmapped array of token ids, sampled as random windows.

    Args:
        path: `.bin` file of little-endian uint16 token ids.
        seq_len: window length. Batches carry `seq_len + 1` tokens internally
            so inputs and targets are one shifted copy of the same window.
        seed: fixes the window sequence. Two runs with the same seed see the
            same batches in the same order, which is the point.
    """

    def __init__(self, path: str | Path, seq_len: int, seed: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run scripts/prepare_data.py to build it"
            )
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < seq_len + 2:
            raise ValueError(f"corpus has {len(self.tokens)} tokens, need at least {seq_len + 2}")
        self.seq_len = seq_len
        self.seed = seed
        # Non-overlapping windows, as their sample index builds them. Window
        # `i` is `tokens[i*seq_len : i*seq_len + seq_len + 1]`, so consecutive
        # windows share the one token that is an input here and a target there.
        self.windows = (len(self.tokens) - 1) // seq_len
        self._epoch = 0
        self._cursor = 0
        self._order: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def epochs_for(self, tokens: int) -> float:
        """How many passes over the corpus a token budget implies.

        Above 1 the run sees some tokens more than once. Nothing here prevents
        that -- windows are sampled with replacement and there is no notion of
        an epoch -- so an undersized corpus produces no error and no warning.
        `train` logs this ratio at the start of a run so the fact is visible
        somewhere rather than nowhere.
        """
        return tokens / len(self.tokens)

    def _reshuffle(self) -> None:
        """The permutation for the current epoch, regenerated rather than stored.

        A 10B-token corpus at `seq_len` 256 is about 39 million windows, and an
        `int64` permutation of those is 313 MB. Keeping it out of the
        checkpoint and rebuilding it from `(seed, epoch)` costs a few seconds
        on resume and keeps the checkpoint the size of the model.
        """
        self._order = np.random.default_rng([self.seed, self._epoch]).permutation(self.windows)

    def _take(self, size: int) -> np.ndarray:
        """The next `size` window indices, crossing into the next epoch if needed."""
        if self._order is None:
            self._reshuffle()
        picks = []
        while size:
            if self._cursor >= self.windows:
                self._epoch += 1
                self._reshuffle()
                self._cursor = 0
            take = min(size, self.windows - self._cursor)
            picks.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            size -= take
        return picks[0] if len(picks) == 1 else np.concatenate(picks)

    def batch(self, size: int, device: str | torch.device = "cpu"):
        """One batch of `(inputs, targets)`, each `(size, seq_len)`."""
        starts = self._take(size).astype(np.int64) * self.seq_len
        window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        chunk = torch.from_numpy(window)
        inputs = chunk[:, :-1].to(device, non_blocking=True)
        targets = chunk[:, 1:].to(device, non_blocking=True)
        return inputs, targets

    @property
    def rng_state(self) -> dict:
        """The sampler's position, so a resumed run does not replay batches.

        Position is now an epoch and an offset into that epoch's permutation
        rather than a bit-generator state, because the permutation is what
        decides the order and it is rebuilt from the epoch number.
        """
        return {"seed": self.seed, "epoch": self._epoch, "cursor": self._cursor}

    @rng_state.setter
    def rng_state(self, state: dict) -> None:
        if not isinstance(state, dict) or "cursor" not in state:
            raise ValueError(
                "this checkpoint stores a bit-generator state, which belongs to "
                "the with-replacement sampler this class no longer uses. Resuming "
                "would silently change the data distribution mid-run; start the "
                "run over with --no-resume instead."
            )
        self.seed = state["seed"]
        self._epoch = state["epoch"]
        self._cursor = state["cursor"]
        self._order = None

    def fixed_batches(self, size: int, count: int, seed: int, device="cpu") -> list:
        """A held-out set that does not move between evaluations."""
        rng = np.random.default_rng(seed)
        hi = len(self.tokens) - self.seq_len - 1
        out = []
        for _ in range(count):
            starts = rng.integers(0, hi, size=size)
            window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
            chunk = torch.from_numpy(window)
            out.append((chunk[:, :-1].to(device), chunk[:, 1:].to(device)))
        return out
