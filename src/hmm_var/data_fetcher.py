"""
Data fetching module for the Adaptive VaR Risk Model.

This module handles OHLCV data retrieval from cryptocurrency exchanges
via the CCXT library. It supports automatic warm-up period loading
to initialize rolling windows before the analysis period.

Includes data quality guardrails for gap detection and spike filtering.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

from hmm_var.settings import Settings

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data fails quality checks."""
    pass


class DataEngine:
    """
    OHLCV data fetcher with automatic warm-up period support.

    Attributes:
        settings: Configuration object containing exchange and date settings.
        exchange: CCXT exchange instance.
    """

    # Data Quality Thresholds
    MAX_GAP_HOURS: int = 24       # Maximum allowed gap between candles
    MAX_SPIKE_PCT: float = 0.20   # Maximum allowed single-candle return (20%)

    def __init__(self, settings: Settings) -> None:
        """
        Initialize DataEngine.
        
        Args:
            settings: Configuration object.
        """
        self.settings = settings
        self._exchange: Optional[ccxt.Exchange] = None

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazy-load CCXT exchange instance."""
        if self._exchange is None:
            self._exchange = getattr(ccxt, self.settings.exchange_id)()
        return self._exchange

    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validates and cleans data for quality issues.
        
        Checks:
        1. Time gaps > MAX_GAP_HOURS
        2. Price spikes > MAX_SPIKE_PCT (potential bad ticks)
        3. Zero/negative prices
        
        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.
            symbol: Symbol name for logging context.
            
        Returns:
            Cleaned DataFrame with anomalies removed.
        """
        if df.empty:
            return df
            
        original_len = len(df)
        issues_found = []
        
        # 1. Check for zero/negative prices
        invalid_prices = (df['close'] <= 0) | (df['open'] <= 0)
        if invalid_prices.any():
            bad_count = invalid_prices.sum()
            issues_found.append(f"Zero/negative prices: {bad_count} rows")
            df = df[~invalid_prices]
            
        # 2. Detect time gaps
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            max_gap = timedelta(hours=self.MAX_GAP_HOURS)
            
            gaps = time_diff[time_diff > max_gap]
            if not gaps.empty:
                for gap_time, gap_size in gaps.items():
                    logger.warning(
                        f"[DataQuality] {symbol}: Gap of {gap_size} detected at {gap_time}"
                    )
                issues_found.append(
                    f"Time gaps > {self.MAX_GAP_HOURS}h: {len(gaps)} detected"
                )
        
        # 3. Detect price spikes (bad ticks)
        if len(df) > 1:
            returns = np.abs(np.log(df['close'] / df['close'].shift(1)))
            spike_mask = returns > self.MAX_SPIKE_PCT
            
            if spike_mask.any():
                spike_count = spike_mask.sum()
                spike_dates = df.index[spike_mask].tolist()
                for date in spike_dates[:5]:
                    logger.warning(
                        f"[DataQuality] {symbol}: Potential bad tick at {date} "
                        f"(>{self.MAX_SPIKE_PCT*100}% move)"
                    )
                
                if spike_count <= len(df) * 0.01:
                    df = df[~spike_mask]
                    issues_found.append(f"Price spikes removed: {spike_count} rows")
                else:
                    issues_found.append(
                        f"Price spikes detected but kept (likely real volatility): {spike_count}"
                    )
        
        # Log summary
        if issues_found:
            removed = original_len - len(df)
            logger.warning(
                f"[DataQuality] {symbol}: {', '.join(issues_found)}. "
                f"Removed {removed} rows total."
            )
        else:
            logger.info(f"[DataQuality] {symbol}: All checks passed. {len(df)} clean rows.")
            
        return df

    def _get_expected_timedelta(self) -> timedelta:
        """Returns expected time delta based on configured timeframe."""
        tf = self.settings.timeframe
        deltas = {
            '1d': timedelta(days=1),
            '4h': timedelta(hours=4),
            '1h': timedelta(hours=1),
            '15m': timedelta(minutes=15),
        }
        return deltas.get(tf, timedelta(days=1))

    def fetch_data(
        self, symbol: Optional[str] = None, include_warmup: bool = True
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT'). If None, uses Settings default.
            include_warmup: If True, fetches extra days before START_DATE.

        Returns:
            DataFrame with Datetime Index and columns ['open', 'high', 'low', 'close', 'volume', 'returns'].
        """
        target_symbol = symbol if symbol else self.settings.symbol
        
        # Calculate start timestamp
        start_dt = datetime.strptime(self.settings.start_date, '%Y-%m-%d')
        start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        if include_warmup:
            start_dt -= timedelta(days=self.settings.warmup_days)
        
        since_ts = int(start_dt.timestamp() * 1000)
        
        limit = 1000 
        all_candles = []
        
        logger.info(f"[DataEngine] Fetching {target_symbol} since {start_dt}...")

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    target_symbol, 
                    timeframe=self.settings.timeframe, 
                    since=since_ts, 
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since_ts = candles[-1][0] + 1
                
                # Check if we reached current time or End Date
                last_candle_time = datetime.fromtimestamp(
                    candles[-1][0] / 1000, tz=timezone.utc
                )
                
                if self.settings.end_date:
                    end_dt = datetime.strptime(
                        self.settings.end_date, '%Y-%m-%d'
                    ).replace(tzinfo=timezone.utc)
                    if last_candle_time >= end_dt:
                        break
                
                now_utc = datetime.now(timezone.utc)
                if last_candle_time >= now_utc:
                    break
                    
                time.sleep(self.exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                logger.error(f"[DataEngine] Network error fetching {target_symbol}: {e}")
                raise
            except ccxt.ExchangeError as e:
                logger.error(f"[DataEngine] Exchange error fetching {target_symbol}: {e}")
                raise
            except Exception as e:
                logger.error(f"[DataEngine] Unexpected error fetching {target_symbol}: {e}")
                raise

        if not all_candles:
            logger.warning(f"[DataEngine] No candles fetched for {target_symbol}")
            return pd.DataFrame()

        # Construct DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Filter strictly by End Date if provided
        if self.settings.end_date:
            end_dt = pd.Timestamp(self.settings.end_date, tz='UTC')
            df = df[df.index <= end_dt]

        # Data Quality Validation
        df = self._validate_data_quality(df, target_symbol)

        # Derived Metrics
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        
        logger.info(f"[DataEngine] {target_symbol}: Loaded {len(df)} clean rows.")

        return df

    def fetch_portfolio_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for all assets defined in Settings.portfolio_assets.
        
        Returns:
            Dictionary {Symbol: DataFrame}
        """
        portfolio_data = {}
        assets = self.settings.portfolio_assets
        
        if not assets:
            logger.warning("[DataEngine] No portfolio assets defined in Settings.")
            return {}
        
        failed_assets = []
            
        for symbol in assets.keys():
            try:
                df = self.fetch_data(symbol=symbol, include_warmup=True)
                if not df.empty:
                    portfolio_data[symbol] = df
                    logger.info(f"[DataEngine] Loaded {symbol}: {len(df)} rows.")
                else:
                    failed_assets.append(symbol)
                    logger.error(
                        f"[DataEngine] Failed to load {symbol}: Empty data returned."
                    )
            except Exception as e:
                failed_assets.append(symbol)
                logger.error(f"[DataEngine] Failed to load {symbol}: {e}")
        
        if failed_assets:
            logger.warning(f"[DataEngine] Failed assets: {failed_assets}")
            
        if not portfolio_data:
            raise DataQualityError("All portfolio assets failed to load. Cannot proceed.")
                
        return portfolio_data

    def fetch_data_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch single-asset data (Legacy Mode) and split into warm-up and analysis.
        
        Returns:
            Tuple of (full_df, analysis_df).
        """
        full_df = self.fetch_data(include_warmup=True)

        if full_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        start_dt = pd.Timestamp(self.settings.start_date, tz='UTC')
        
        warmup_df = full_df[full_df.index < start_dt]
        analysis_df = full_df[full_df.index >= start_dt]
        
        return full_df, analysis_df
