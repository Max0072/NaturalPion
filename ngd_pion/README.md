# ngd_pion

**The method is `natural_pion.py`.**

| class | harness name | what it is |
|---|---|---|
| `NaturalPionFast` | `natural-pion-fast` | the method (the paper's Algorithm 1), its defaults: Pion's RMS length rule, floor 1e-3, basis every 25 steps, A and D (EMA 0.5) in the 6 steps before a refresh, one A per input, bf16 statistics inputs, F^-1 in bf16, TF32 before the rotation, the squared Cayley retraction in fp32. Its switches give the paper's exact form (`t_fac` 1, `stats_window` None, `retraction` "cayley") and every ablation of Section 4.4 |
| `NaturalPion` | `natural-pion` | the earlier exact reference with the angle rule c = theta / sigma, kept for the tests |

Primitives: `covariance` (EMA second moments A, D), `direction` (generators, the Fisher operator, its solve in a
basis), `factorization` (the generalized eigenbasis), `linalg` (Cayley, spectral norm, exact-fp32 context). `hooks`
records activations (A) and output gradients (D). `pion_baseline` is published Pion, the only baseline. `reference`
is a numpy oracle for the tests.

The earlier NGD-Pion optimizers (`ngd-pion-*`, the alpha trust region, `shampoo-pion`) were removed on 2026-11-05;
they are in git history and described in `docs/history/`.
