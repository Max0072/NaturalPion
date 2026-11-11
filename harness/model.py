"""LLaMA-style transformer in the configuration Pion's 60M ablations use.

Values are taken from `opt_llama_60M_pion.sh` in their repository, not from
the paper, which omits several of them: hidden 512, 8 layers, 8 heads,
ffn 1376, kv_channels 64, RMSNorm, sequence length 256, `init_method_std`
0.02.

Plain `nn.Module` and plain `nn.Linear` on purpose. Their code runs on
Megatron-LM, which replaces linear layers with tensor-parallel ones, and none
of that machinery buys anything for a 60M model on a single GPU -- it only
makes the optimizer harder to attach. Megatron is worth reaching for exactly
once, to anchor these numbers against theirs.

Initialisation is Gaussian rather than semi-orthogonal, which matters more
here than it looks: Pion freezes the singular values of every 2-D weight for
the whole run, so this choice fixes the spectrum permanently. Gaussian keeps
the spectrum non-trivial and expressive at the cost of `W^T W != I`, which
sends the optimizer down its general path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModelConfig", "Transformer", "matrix_parameters"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32100          # T5-base sentencepiece plus its extra ids
    hidden: int = 512
    layers: int = 8
    heads: int = 8
    ffn_hidden: int = 1376
    seq_len: int = 256
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    # Power-law exponent for a heavy-tailed initialisation, or 0 for the usual
    # iid normal. Heavy-tailed self-regularisation (Martin & Mahoney) reports
    # the spectral density of a *trained* weight going as `lam^-alpha` with
    # `alpha` roughly in [2, 6], while an iid normal gives Marchenko-Pastur --
    # compact support, hard edge, no tail. Those differ in shape, not scale, so
    # no rescaling of the usual initialisation reaches one from the other.
    #
    # It matters more here than it would elsewhere. Pion and NGD-Pion move only
    # the singular *vectors*: the spectrum a weight starts with is the spectrum
    # it ends with. If the trained spectrum is heavy-tailed and the initial one
    # is not, a spectrum-preserving optimizer cannot reach it at all -- and
    # conversely, starting from the right shape is the one intervention that
    # would let it.
    init_pl_alpha: float = 0.0
    # The initialisation of the rotational weights, as a name rather than as a
    # corner of `init_pl_alpha`.
    #
    # `"normal"` is `normal(0, init_std)` and is what every run on disk used.
    # `"orthogonal"` is a plain semi-orthogonal weight with every singular value
    # equal to `init_gain`. `"xavier"` is Glorot uniform.
    #
    # **This matters more here than in an ordinary network.** Pion and NGD-Pion
    # rotate, so the singular values of a rotational weight are fixed at
    # initialisation and never move again. The choice is not a starting point,
    # it is a permanent property of the model.
    #
    # `"orthogonal"` is deliberately *not* the same as `init_pl_alpha = inf`.
    # That one is also flat but scaled so the spectral norm is
    # `sqrt(fan_out/fan_in)`, the feature-learning condition; this one sets
    # every singular value to `init_gain`, which is 1 by default. Since the
    # spectrum is frozen, the difference between them is a real experimental
    # variable and not a convention, so they are separate settings.
    # Residual connections. `False` makes every block a plain composition, so
    # the signal must pass through each weight rather than around it.
    #
    # It is here because this optimizer makes a question askable that is
    # usually not. Dynamical isometry -- orthogonal weights preserving the
    # signal norm through depth -- is the classical prerequisite for training
    # without residuals, and it normally decays as the weights train. Pion and
    # NGD-Pion **freeze the singular values**, so an orthogonal initialisation
    # stays orthogonal for the whole run, exactly, as long as the retraction is
    # Cayley. So: does preserving the spectrum substitute for a residual?
    #
    # Two things the flag cannot deliver, and both belong beside any result:
    # the *weights* stay orthogonal but the block Jacobian does not, since
    # softmax, RoPE and SwiGLU are in the way; and residuals earn their keep at
    # depth, while this model has 8 layers, so a negative result here is weak
    # evidence and a positive one is strong.
    residual: bool = True
    init: str = "normal"          # normal | orthogonal | xavier
    init_gain: float = 1.0        # singular value for `orthogonal`

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads:
            raise ValueError(f"hidden {self.hidden} is not divisible by heads {self.heads}")
        return self.hidden // self.heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _rope_tables(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv)
    return angles.cos()[None, None], angles.sin()[None, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with RoPE. Q, K, V stay separate.

    Megatron fuses them into one `3d x d` matrix; keeping them apart means each
    weight is a genuine `d x d` square, which is the shape the optimizer's
    cheap path is defined for and one fewer thing to reconcile later.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.wk = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.wv = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.wo = nn.Linear(cfg.hidden, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        shape = (B, T, self.heads, self.head_dim)
        q = self.wq(x).view(shape).transpose(1, 2)
        k = self.wk(x).view(shape).transpose(1, 2)
        v = self.wv(x).view(shape).transpose(1, 2)
        q, k = _apply_rope(q, cos[..., :T, :], sin[..., :T, :]), _apply_rope(k, cos[..., :T, :], sin[..., :T, :])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.residual = cfg.residual
        self.attn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if not self.residual:
            x = self.attn(self.attn_norm(x), cos, sin)
            return self.ffn(self.ffn_norm(x))
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        cos, sin = _rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return
        # The embedding and the head keep the usual initialisation whatever
        # `init_pl_alpha` says. The spectral condition below is derived for
        # matmul layers in the wide limit; an embedding is a lookup table, and
        # `sqrt(32100/512) = 7.9` would stretch its rows to twice the usual
        # scale for no stated reason. Both belong to AdamW in any case, which
        # can move their spectra, so nothing is frozen there.
        rotational = module is not self.embed and module is not self.head
        if not rotational:
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            return
        if self.cfg.init != "normal" and self.cfg.init_pl_alpha > 0.0:
            raise ValueError(
                f"init={self.cfg.init!r} and init_pl_alpha={self.cfg.init_pl_alpha} "
                "both set; they are two ways to choose the same spectrum"
            )
        if self.cfg.init == "orthogonal":
            # `orthogonal_` gives a semi-orthogonal matrix whatever the shape,
            # so every singular value is `init_gain` for both tall and wide.
            nn.init.orthogonal_(module.weight, gain=self.cfg.init_gain)
        elif self.cfg.init == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif self.cfg.init == "normal":
            if self.cfg.init_pl_alpha > 0.0:
                _power_law_(module.weight, self.cfg.init_pl_alpha, self.cfg.init_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        else:
            raise ValueError(
                f"init must be normal/orthogonal/xavier, got {self.cfg.init!r}"
            )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self.head(self.final_norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def parameter_split(self) -> tuple[list[nn.Linear], list[nn.Parameter]]:
        """Which weights Pion owns, and which fall to AdamW.

        Their rule verbatim: 2-D parameters that are neither the embedding nor
        the output projection. Everything else -- embedding, head, norm gains
        -- goes to AdamW.
        """
        rotational, rest = [], []
        excluded = {id(self.embed.weight), id(self.head.weight)}
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module.weight) not in excluded:
                rotational.append(module)
        owned = {id(m.weight) for m in rotational}
        rest = [p for p in self.parameters() if id(p) not in owned]
        return rotational, rest


    def query_projections(self) -> list[nn.Linear]:
        """The Q projections, which their optimizer rotates one head at a time.

        Megatron carries a fused QKV matrix and slices it; this harness carries
        `wq`, `wk` and `wv` separately, so the same granularity is reached by
        splitting `wq` into `heads` row-blocks. `wk` and `wv` are rotated whole
        in their code too, and need no such treatment.
        """
        return [module for name, module in self.named_modules() if name.endswith(".wq")]


def matrix_parameters(model: Transformer) -> list[nn.Parameter]:
    return [m.weight for m in model.parameter_split()[0]]


# Fraction of the spectrum replaced by the power-law tail. HT-SR fits its
# exponent to the upper tail of the ESD -- typically the largest ten to twenty
# per cent of eigenvalues -- not to the whole spectrum, so a faithful
# construction keeps a bulk and attaches a tail to it.
_TAIL_FRACTION = 0.2


@torch.no_grad()
def _power_law_(w: torch.Tensor, alpha: float, std: float) -> None:
    """A trained-looking singular spectrum: Marchenko-Pastur bulk, power-law tail.

    Two regimes, and they answer two different pieces of literature.

    `alpha = inf` gives a **flat** spectrum -- every singular value equal --
    scaled so the spectral norm is `sqrt(fan_out/fan_in)`. That is a scaled
    semi-orthogonal weight: what dynamical isometry asks for at initialisation
    (arXiv:1711.04735) at the scale feature learning asks for
    (arXiv:2310.17813). It also makes `W^T W` proportional to the identity,
    which Cayley then preserves for the whole run.

    Finite `alpha` gives what HT-SR says a *trained* layer looks like: an iid
    normal matrix's own singular values as the bulk -- which is Marchenko-Pastur
    by construction, not by approximation -- with the largest `_TAIL_FRACTION`
    of them replaced by a power law attached continuously at the break. The
    exponent is theirs: an ESD going as `lam^-alpha` means the i-th largest
    eigenvalue goes as `i^-1/(alpha-1)`, hence `s_i ~ i^-1/(2(alpha-1))`.

    An earlier version applied the power law to the *whole* spectrum, which has
    no bulk at all and is not the object that literature measures. Its
    `alpha = 2` and their `alpha = 2` were different things sharing a name.

    The bulk keeps the scale of the `normal(0, std)` it replaces and the tail
    extends above it, because that is what training does: measured over 2000
    AdamW steps, the largest singular value of an inner layer grows by a factor
    of 2 to 9. This construction gives 2.4x at `alpha = 3` and 7.6x at
    `alpha = 2`, which is the same range.
    """
    if alpha <= 1.0:
        raise ValueError(f"init_pl_alpha must exceed 1, got {alpha}")
    m, n = w.shape
    r = min(m, n)
    U = torch.linalg.qr(torch.randn(m, r, device=w.device))[0]
    V = torch.linalg.qr(torch.randn(n, r, device=w.device))[0]

    if math.isinf(alpha):
        s = torch.full((r,), (m / n) ** 0.5, device=w.device)
    else:
        bulk = torch.linalg.svdvals(torch.randn(m, n, device=w.device) * std)
        k = max(1, min(r - 1, int(round(_TAIL_FRACTION * r))))
        edge = bulk[k]
        beta = 1.0 / (2.0 * (alpha - 1.0))
        i = torch.arange(1, k + 1, dtype=bulk.dtype, device=w.device)
        # anchored so that index k+1 reproduces `edge` exactly: no discontinuity
        tail = edge * (i / (k + 1)) ** (-beta)
        s = torch.cat([tail, bulk[k:]])
    w.copy_(((U * s) @ V.transpose(-1, -2)).to(w.dtype))
