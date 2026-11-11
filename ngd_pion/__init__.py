"""Natural Pion: spectrum-preserving rotations along the natural gradient.

The method is `natural_pion`: `NaturalPionFast` is the method (the paper's Algorithm 1) and, with switches, its exact form
and every ablation; `NaturalPion` is the earlier angle-rule reference kept for the tests. It is built from the primitives in
`covariance`, `direction`, `factorization`, `linalg` and fed by the recorders in `hooks` (activations for A, output
gradients for D). `pion_baseline` is published Pion, the baseline; `reference` is a numpy oracle for the tests.
"""

from .covariance import CovarianceAccumulator
from .direction import fisher_apply, generators, natural_gradient
from .factorization import Basis, basis_congruence, basis_identity_anchor
from .hooks import BackwardRecorder, attach, attach_backward, attached
from .linalg import cayley, floor_eigenvalues, floor_spectrum, skew, spectral_norm
from .natural_pion import NaturalPion, NaturalPionFast
from .pion_baseline import Pion

__all__ = [
    "NaturalPion",
    "NaturalPionFast",
    "Pion",
    "attach",
    "attached",
    "attach_backward",
    "BackwardRecorder",
    "CovarianceAccumulator",
    "Basis",
    "basis_congruence",
    "basis_identity_anchor",
    "generators",
    "natural_gradient",
    "fisher_apply",
    "cayley",
    "spectral_norm",
    "skew",
    "floor_eigenvalues",
    "floor_spectrum",
]
