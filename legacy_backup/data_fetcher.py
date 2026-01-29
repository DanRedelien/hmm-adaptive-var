"""
Data fetching module for the Adaptive VaR Risk Model.

This module handles OHLCV data retrieval from cryptocurrency exchanges
via the CCXT library. It supports automatic warm-up period loading
to initialize rolling windows before the analysis period.

Includes data quality guardrails for gap detection and spike filtering.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from config import Config


logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data fails quality checks."""
    pass


class DataEngine:
    """
    OHLCV data fetcher with automatic warm-up period support.

    Attributes:
        cfg: Configuration object containing exchange and date settings.
        exchange: CCXT exchange instance.
    """

    # Data Quality Thresholds
    MAX_GAP_HOURS: int = 24       # Maximum allowed gap between candles (1 day for daily TF)
    MAX_SPIKE_PCT: float = 0.20   # Maximum allowed single-candle return (20%)

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Initialize exchange without API keys (public data only)
        self.exchange: ccxt.Exchange = getattr(ccxt, cfg.EXCHANGE_ID)()

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
            
        Raises:
            DataQualityError: If data has critical issues that cannot be fixed.
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
            # Convert timeframe to expected delta
            expected_delta = self._get_expected_timedelta()
            max_gap = timedelta(hours=self.MAX_GAP_HOURS)
            
            gaps = time_diff[time_diff > max_gap]
            if not gaps.empty:
                for gap_time, gap_size in gaps.items():
                    logger.warning(f"[DataQuality] {symbol}: Gap of {gap_size} detected at {gap_time}")
                issues_found.append(f"Time gaps > {self.MAX_GAP_HOURS}h: {len(gaps)} detected")
        
        # 3. Detect price spikes (bad ticks)
        if len(df) > 1:
            returns = np.abs(np.log(df['close'] / df['close'].shift(1)))
            spike_mask = returns > self.MAX_SPIKE_PCT
            
            if spike_mask.any():
                spike_count = spike_mask.sum()
                spike_dates = df.index[spike_mask].tolist()
                for date in spike_dates[:5]:  # Log first 5
                    logger.warning(f"[DataQuality] {symbol}: Potential bad tick at {date} (>{self.MAX_SPIKE_PCT*100}% move)")
                
                if spike_count <= len(df) * 0.01:  # Remove if < 1% of data
                    df = df[~spike_mask]
                    issues_found.append(f"Price spikes removed: {spike_count} rows")
                else:
                    # Too many spikes - might be legitimate volatility
                    issues_found.append(f"Price spikes detected but kept (likely real volatility): {spike_count}")
        
        # Log summary
        if issues_found:
            removed = original_len - len(df)
            logger.warning(f"[DataQuality] {symbol}: {', '.join(issues_found)}. Removed {removed} rows total.")
        else:
            logger.info(f"[DataQuality] {symbol}: All checks passed. {len(df)} clean rows.")
            
        return df

    def _get_expected_timedelta(self) -> timedelta:
        """Returns expected time delta based on configured timeframe."""
        tf = self.cfg.TIMEFRAME
        if tf == '1d':
            return timedelta(days=1)
        elif tf == '4h':
            return timedelta(hours=4)
        elif tf == '1h':
            return timedelta(hours=1)
        elif tf == '15m':
            return timedelta(minutes=15)
        else:
            return timedelta(days=1)  # Default

    def fetch_data(self, symbol: str = None, include_warmup: bool = True) -> pd.DataFrame:
        """
        Fetches OHLCV data for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT'). If None, uses Config default.
            include_warmup: If True, fetches extra days before START_DATE.

        Returns:
            DataFrame with Datetime Index and columns ['open', 'high', 'low', 'close', 'volume', 'returns'].
            
        Raises:
            ccxt.NetworkError: If exchange connection fails.
            ccxt.ExchangeError: If exchange returns an error.
        """
        target_symbol = symbol if symbol else self.cfg.SYMBOL
        
        # Calculate start timestamp
        start_str = self.cfg.START_DATE
        start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        
        if include_warmup:
            start_dt -= timedelta(days=self.cfg.WARMUP_DAYS)
        
        since_ts = int(start_dt.timestamp() * 1000)
        
        # Limit fetches to avoid bans (safety mechanism)
        limit = 1000 
        all_candles = []
        
        logger.info(f"[DataEngine] Fetching {target_symbol} since {start_dt}...")

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    target_symbol, 
                    timeframe=self.cfg.TIMEFRAME, 
                    since=since_ts, 
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since_ts = candles[-1][0] + 1
                
                # Check if we reached current time or End Date
                last_candle_time = datetime.fromtimestamp(candles[-1][0] / 1000)
                if self.cfg.END_DATE:
                    end_dt = datetime.strptime(self.cfg.END_DATE, '%Y-%m-%d %H:%M:%S')
                    if last_candle_time >= end_dt:
                        break
                
                if last_candle_time >= datetime.now():
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
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter strictly by End Date if provided
        if self.cfg.END_DATE:
            df = df[df.index <= self.cfg.END_DATE]

        # Data Quality Validation
        df = self._validate_data_quality(df, target_symbol)

        # Derived Metrics
        # Log Returns: ln(P_t / P_{t-1}) (Continuous compounding assumption)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        df.dropna(inplace=True)
        
        logger.info(f"[DataEngine] {target_symbol}: Loaded {len(df)} clean rows.")

        return df

    def fetch_portfolio_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for all assets defined in Config.PORTFOLIO_ASSETS.
        
        Returns:
            Dictionary {Symbol: DataFrame}
            
        Raises:
            DataQualityError: If any critical asset fails to load.
        """
        portfolio_data = {}
        assets = self.cfg.PORTFOLIO_ASSETS
        
        if not assets:
            logger.warning("[DataEngine] No portfolio assets defined in Config.")
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
                    logger.error(f"[DataEngine] Failed to load {symbol}: Empty data returned.")
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

        start_dt = pd.Timestamp(self.cfg.START_DATE)
        
        warmup_df = full_df[full_df.index < start_dt]
        analysis_df = full_df[full_df.index >= start_dt]
        
        return full_df, analysis_df