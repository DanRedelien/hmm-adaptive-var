"""
Configuration module for the Adaptive VaR Risk Model.

This module defines all configurable parameters for the risk pipeline,
including exchange settings, capital, VaR confidence levels, and
regime switching parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    """
    Central configuration for the Adaptive VaR Risk Model.

    All parameters that affect model behavior should be defined here.
    No hardcoded values should exist in the codebase outside this class.

    Attributes:
        EXCHANGE_ID: CCXT exchange identifier (e.g., 'binance', 'kraken').
        SYMBOL: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT').
        TIMEFRAME: Candle timeframe ('1d', '4h', '1h', etc.).
        START_DATE: Start of the analysis period (backtesting begins here).
        END_DATE: End of the analysis period.
        WARMUP_DAYS: Number of days to load BEFORE START_DATE for model warm-up.
                     This data is used to initialize rolling windows but is
                     excluded from backtest statistics. Should be >= WINDOW_CALM.
        CAPITAL: Portfolio capital in USD for VaR conversion.
        VAR_CONFIDENCE: VaR confidence level (0.95 = 95% VaR).
        HORIZONS: List of holding period horizons to analyze (in days).
        WINDOW_CALM: Rolling window for Calm regime VaR (Basel standard = 252 days).
        WINDOW_STRESS: Rolling window for Stress regime VaR (Reactive = 21-60 days).
        MARKOV_SMOOTH_WINDOW: Window for smoothing regime probabilities.
        PORTFOLIO_ASSETS: Dictionary of {Symbol: Weight} for Portfolio VaR. 
                          If Empty, runs in Single-Asset mode.
        CORRELATION_WINDOW: Lookback window for rolling correlation heatmap.
    """

    # --- Data Source Settings ---
    EXCHANGE_ID: str = 'binance'
    SYMBOL: str = 'BTCUSDT'  # Default single asset
    TIMEFRAME: str = '1d'

    # --- Time Settings ---
    START_DATE: str = '2021-01-01 00:00:00'
    # If None, fetches up to current time
    END_DATE: str = '2026-01-27 00:00:00' 
    
    # Critical for cold-start stability (e.g., 252 days for valid initial rolling stats)
    WARMUP_DAYS: int = 300 

    # --- Risk Capital ---
    CAPITAL: float = 100_000.0  # USD
    
    # 0.95 (1.65 sigma) or 0.99 (2.33 sigma)
    VAR_CONFIDENCE: float = 0.95 
    
    # Horizons to test: 1 day, 3 days, 7 days (Week), 30 days (Month)
    HORIZONS: List[int] = field(default_factory=lambda: [1, 5, 21])

    # --- Portfolio Settings (Multivariate) ---
    # Leave empty for Single Asset Mode.
    # Weights must sum to 1.0.
    PORTFOLIO_ASSETS: Dict[str, float] = field(default_factory=lambda: {
        'BTCUSDT': 0.05, # 1
        'ETHUSDT': 0.05, # 2
        'SOLUSDT': 0.05, # 3
        'ADAUSDT': 0.05, # 4
        'XRPUSDT': 0.05, # 5
        'DOGEUSDT': 0.05, # 6
        'UNIUSDT': 0.05, # 7
        'LUNAUSDT': 0.05, # 8
        'DOTUSDT': 0.05, # 9
        'XLMUSDT': 0.05, # 10
        'AVAXUSDT': 0.05, # 11
        'MATICUSDT': 0.05, # 12
        'LINKUSDT': 0.05, # 13
        'BNBUSDT': 0.05, # 14
        'TRXUSDT': 0.05, # 15
        'ETCUSDT': 0.05, # 16
        'PEPEUSDT': 0.05, # 17
        'LTCUSDT': 0.05, # 18
        'FILUSDT': 0.05, # 19
        'ATOMUSDT': 0.05, # 20
    })
    
    # Lookback for Rolling Correlation Heatmap (Analysis)
    CORRELATION_WINDOW: int = 30
    
    # Asset to highlight in the correlation plot (e.g., 'BTCUSDT')
    HIGHLIGHT_ASSET: str = 'BTCUSDT'

    # --- Adaptive Risk Settings ---
    # Calm regime window (Basel Standard = 1 year)
    # Provides stable VaR that doesn't jump on every news event.
    WINDOW_CALM: int = 365

    # Stress regime window (Reactive = 1 month)
    # When Markov detects stress, we ignore old history and focus on recent data.
    WINDOW_STRESS: int = 180

    # --- Markov Settings ---
    # Smoothing window for regime probabilities.
    # 3 days = Very fast reaction, more false positives.
    # 7-10 days = Fewer false positives, but slower entry.
    MARKOV_SMOOTH_WINDOW: int = 3

    # --- Markov Rolling Estimation (Eliminate Look-Ahead Bias) ---
    MARKOV_WINDOW_SIZE: int = 252       # History length for training (1 year)
    MARKOV_REFIT_INTERVAL: int = 7      # Re-train model every N days (Sparse Re-estimation)
    MARKOV_MIN_OBSERVATIONS: int = 100  # Minimum history to start first fit

    def __post_init__(self):
        """Validate configuration."""
        if self.PORTFOLIO_ASSETS:
            total_weight = sum(self.PORTFOLIO_ASSETS.values())
            # Allow small float error for floating point arithmetic
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight}")