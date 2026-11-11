"""LLaMA-style transformer in the configuration Pion's 60M ablations use.

Values are taken from `opt_llama_60M_pion.sh` in their repository, not from
the paper, which omits several of them: hidden 512, 8 layers, 8 heads,
ffn 1376, kv_channels 64, RMSNorm, sequence length 256, `init_method_std`
0.02.

Plain `nn.Module` and plain `nn.Linear` on purpose. Their code runs on
Megatron-LM, which replaces linear layers with tensor-parallel ones, and none
of that machinery buys anything for a 60M model on a single GPU -- it only
makes the optimizer harder to attach.

Initialisation is Gaussian rather than semi-orthogonal, which matters more
here than it looks: Pion freezes the singular values of every 2-D weight for
the whole run, so this choice fixes the spectrum permanently. Gaussian keeps
the spectrum non-trivial and expressive at the cost of `W^T W != I`, which
sends the optimizer down its general path.
"""

from __future__ import annotations

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
        self.attn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

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
