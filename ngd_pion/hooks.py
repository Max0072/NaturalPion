"""Convenience adapter: feed `nn.Linear` activations to the optimizer.

Deliberately thin and deliberately optional. Other frameworks name their
layers differently -- Megatron's parallel linears, a fused QKV projection --
and each needs its own adapter of about this size. Keeping it out of
`optimizer` is what lets those exist without touching the optimizer at all.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

__all__ = ["ActivationRecorder", "BackwardRecorder", "attach", "attach_backward", "attached"]


class ActivationRecorder:
    """Forwards each layer's input covariance to the optimizer while enabled.

    The switch matters under gradient accumulation. The covariance EMA is
    defined per optimizer step, so letting it fire on every micro-batch would
    silently shorten its horizon by the accumulation factor. Recording one
    micro-batch per step keeps `beta` meaning what it says, and one micro-batch
    is already far more samples than the covariance needs.
    """

    def __init__(self, modules, optimizer) -> None:
        modules = list(modules)
        for module in modules:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"expected nn.Linear, got {type(module).__name__}")
        self.enabled = True
        self._handles = [m.register_forward_pre_hook(self._make(optimizer)) for m in modules]

    def _make(self, optimizer):
        def hook(module, inputs):
            if self.enabled:
                optimizer.observe(module.weight, inputs[0])
            return None

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __len__(self) -> int:
        return len(self._handles)


def attach(modules, optimizer) -> ActivationRecorder:
    """Start recording activations for `modules` into `optimizer`."""
    return ActivationRecorder(modules, optimizer)


@contextmanager
def attached(modules, optimizer):
    """`attach` as a context manager, detaching on exit."""
    recorder = attach(modules, optimizer)
    try:
        yield recorder
    finally:
        recorder.remove()


class BackwardRecorder:
    """Feeds each layer's output gradient to `observe_backward`.

    A forward hook that attaches a hook to the output tensor, rather than
    `register_full_backward_hook`: the latter wraps the module in autograd
    functions and materialises `grad_input`, which is not needed here.

    `enabled` mirrors `ActivationRecorder`'s: the EMA is defined per optimizer
    step, so under gradient accumulation only one micro-batch may feed it.
    """

    def __init__(self, modules, optimizer) -> None:
        self.enabled = True
        self._handles = [
            m.register_forward_hook(self._make(m, optimizer)) for m in list(modules)
        ]

    def _make(self, module, optimizer):
        weight = module.weight

        def forward_hook(mod, inputs, output):
            if not self.enabled or not torch.is_tensor(output) or not output.requires_grad:
                return None

            def grad_hook(grad):
                optimizer.observe_backward(weight, grad.detach())
                return None

            output.register_hook(grad_hook)
            return None

        return forward_hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __len__(self) -> int:
        return len(self._handles)


def attach_backward(modules, optimizer) -> BackwardRecorder:
    """Start recording output gradients for `modules` into `optimizer`."""
    return BackwardRecorder(modules, optimizer)
