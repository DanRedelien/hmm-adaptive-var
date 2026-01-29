"""
Configuration module for the Adaptive VaR Risk Model.

Uses pydantic-settings for validation, type safety, and .env file support.
All parameters are flattened (no nesting) for simplicity.
"""

from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Adaptive VaR Risk Model.

    All parameters that affect model behavior are defined here with validation.
    Supports environment variable overrides with VAR_ prefix and .env files.

    Example:
        >>> settings = Settings()  # Load defaults
        >>> settings = Settings(capital=50_000)  # Override capital
        >>> # Or via environment: VAR_CAPITAL=50000

    Attributes:
        exchange_id: CCXT exchange identifier (e.g., 'binance', 'kraken').
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT').
        timeframe: Candle timeframe ('1d', '4h', '1h', '15m').
        start_date: Start of the analysis period (ISO format, UTC assumed).
        end_date: End of the analysis period (None = current time).
        warmup_days: Days before START_DATE for model warm-up (>= window_calm).
        capital: Portfolio capital in USD for VaR conversion.
        var_confidence: VaR confidence level (0.95 = 95% VaR).
        horizons: List of holding period horizons to analyze (in days).
        window_calm: Rolling window for Calm regime VaR (Basel = 252-365 days).
        window_stress: Rolling window for Stress regime VaR (Reactive = 30-180 days).
        markov_smooth_window: Window for smoothing regime probabilities.
        markov_window_size: History length for HMM training.
        markov_refit_interval: Re-train model every N days.
        markov_min_observations: Minimum history to start first fit.
        portfolio_assets: Dictionary of {Symbol: Weight} for Portfolio VaR.
        correlation_window: Lookback for rolling correlation heatmap.
        highlight_asset: Asset to highlight in correlation plot.
        cache_dir: Directory for Parquet data cache.
    """

    model_config = SettingsConfigDict(
        env_prefix="VAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data Source Settings ---
    exchange_id: str = Field(
        default="binance",
        description="CCXT exchange identifier",
    )
    symbol: str = Field(
        default="BTCUSDT",
        description="Default single asset trading pair",
    )
    timeframe: str = Field(
        default="1d",
        pattern=r"^(1d|4h|1h|15m)$",
        description="Candle timeframe",
    )

    # --- Time Settings (UTC) ---
    start_date: str = Field(
        default="2023-01-01",
        description="Analysis start date (ISO format, UTC)",
    )
    end_date: str | None = Field(
        default=None,
        description="Analysis end date (None = current time)",
    )
    warmup_days: int = Field(
        default=300,
        ge=100,
        description="Days before start_date for model initialization",
    )

    # --- Risk Capital ---
    capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Portfolio capital in USD",
    )
    var_confidence: float = Field(
        default=0.95,
        ge=0.9,
        le=0.999,
        description="VaR confidence level (0.95 = 95%)",
    )
    horizons: List[int] = Field(
        default=[1, 5, 21],
        description="Holding period horizons in days",
    )

    # --- Adaptive Risk Settings ---
    window_calm: int = Field(
        default=252,
        ge=100,
        description="Calm regime lookback (Basel standard)",
    )
    window_stress: int = Field(
        default=90,
        ge=30,
        description="Stress regime lookback (Reactive)",
    )

    # --- Weighted Historical Simulation (WHS) Settings ---
    enable_whs: bool = Field(
        default=True,
        description="Enable Regime-Similarity Weighted Historical Simulation",
    )
    ess_threshold: int = Field(
        default=10,
        ge=5,
        description="Minimum Effective Sample Size; alert if ESS < threshold",
    )

    # --- Markov Settings ---
    markov_smooth_window: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Smoothing window for regime probabilities",
    )
    markov_window_size: int = Field(
        default=252,
        ge=100,
        description="History length for HMM training",
    )
    markov_refit_interval: int = Field(
        default=7,
        ge=1,
        description="Re-train model every N days",
    )
    markov_min_observations: int = Field(
        default=100,
        ge=50,
        description="Minimum history to start first fit",
    )

    # --- Portfolio Settings ---
    portfolio_assets: Dict[str, float] = Field(
        default_factory=lambda: {
            "BTCUSDT": 0.2,
            "ETHUSDT": 0.1,
            "SOLUSDT": 0.1,
            "XRPUSDT": 0.1,
            "BNBUSDT": 0.1,
            "ADAUSDT": 0.1,
            "AVAXUSDT": 0.1,
            "DOTUSDT": 0.1,
            "TRXUSDT": 0.1,
        },
        description="Portfolio weights (must sum to 1.0)",
    )
    correlation_window: int = Field(
        default=30,
        ge=5,
        description="Lookback for rolling correlation heatmap",
    )
    highlight_asset: str = Field(
        default="BTCUSDT",
        description="Asset to highlight in correlation plot",
    )

    # --- Rebalancing Settings ---
    enable_rebalancing: bool = Field(
        default=True,
        description="Enable realistic drift and periodic rebalancing mechanics",
    )
    rebalance_interval: str = Field(
        default="1M",
        description="Rebalancing frequency (e.g., '1M' for monthly, '1W' for weekly)",
    )
    transaction_fee_bps: float = Field(
        default=10.0,
        ge=0.0,
        description="Transaction fee in basis points (bps) per rebalance turnover",
    )

    # --- Data Cache ---
    cache_dir: str = Field(
        default="data/cache",
        description="Directory for Parquet data cache",
    )

    @field_validator("portfolio_assets")
    @classmethod
    def validate_portfolio_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure portfolio weights sum to 1.0 (within float tolerance)."""
        if v:
            total_weight = sum(v.values())
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(
                    f"Portfolio weights must sum to 1.0, got {total_weight:.4f}"
                )
        return v

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: List[int]) -> List[int]:
        """Ensure all horizons are positive."""
        if any(h <= 0 for h in v):
            raise ValueError("All horizons must be positive integers")
        return sorted(v)

    def get_cache_path(self) -> Path:
        """Returns the resolved cache directory path."""
        return Path(self.cache_dir).resolve()


# Singleton for convenience (optional usage pattern)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
