"""
HMM-VaR: Adaptive Value-at-Risk with Hidden Markov Regime Detection.

A quantitative risk model that dynamically switches between Calm and Stress
regime parameters based on real-time Markov regime probabilities.
"""

__version__ = "0.1.0"
__author__ = "Dan Redelien"

# Core components (lazy import pattern for heavy dependencies)
from hmm_var.settings import Settings, get_settings

# Exceptions are lightweight, import directly
from hmm_var.data_fetcher import DataQualityError
from hmm_var.data_lake import FrequencyMismatchError


def __getattr__(name: str):
    """Lazy loading for heavy modules to speed up initial import."""
    if name == "RiskEngine":
        from hmm_var.var_model import RiskEngine
        return RiskEngine
    elif name == "RegimeModel":
        from hmm_var.markov_model import RegimeModel
        return RegimeModel
    elif name == "DataEngine":
        from hmm_var.data_fetcher import DataEngine
        return DataEngine
    elif name == "DataLake":
        from hmm_var.data_lake import DataLake
        return DataLake
    elif name == "AnalyticsHub":
        from hmm_var.analytics import AnalyticsHub
        return AnalyticsHub
    elif name == "main":
        from hmm_var.main import main
        return main
    raise AttributeError(f"module 'hmm_var' has no attribute '{name}'")


__all__ = [
    # Core
    "Settings",
    "get_settings",
    "RiskEngine",
    "RegimeModel",
    "DataEngine",
    "DataLake",
    "AnalyticsHub",
    # Entry point
    "main",
    # Exceptions
    "DataQualityError",
    "FrequencyMismatchError",
    # Metadata
    "__version__",
    "__author__",
]

