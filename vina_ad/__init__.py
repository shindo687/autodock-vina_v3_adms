"""Differentiable AutoDock-Vina sidecar with explicit ChainRules rules."""

from .core import DEFAULT_WEIGHTS, energy, score, score_coordinates
from .protocol import NonDifferentiablePoint, RuleNotFound, UnsupportedWrt, ZERO
from .protocol import grad, jvp, value_and_grad, vjp

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_WEIGHTS",
    "score_coordinates",
    "score",
    "energy",
    "jvp",
    "vjp",
    "grad",
    "value_and_grad",
    "ZERO",
    "RuleNotFound",
    "UnsupportedWrt",
    "NonDifferentiablePoint",
]
