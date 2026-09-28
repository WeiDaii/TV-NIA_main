"""TV-NIA: topology-vulnerability-guided node-injection attack framework."""

from .config import InjectionBudget, TVNIAConfig
from .framework import compute_injection_budget, run_tvnia

__all__ = [
    "InjectionBudget",
    "TVNIAConfig",
    "compute_injection_budget",
    "run_tvnia",
]

