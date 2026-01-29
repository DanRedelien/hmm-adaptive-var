"""
Data Lake module for robust Parquet-based OHLCV caching.

Implements safe delta-fetch logic:
1. Check for cached Parquet file
2. Load cached data with UTC enforcement
3. Validate frequency matches config
4. Fetch only new data from CCXT (delta)
5. Append and save
6. Log all operations with row counts and timestamp ranges
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hmm_var.settings import Settings

logger = logging.getLogger(__name__)


class FrequencyMismatchError(Exception):
    """Raised when cached data frequency doesn't match config."""
    pass


class DataLake:
    """
    Parquet-based OHLCV cache with CCXT delta-fetch.

    Provides robust data storage with:
    - Explicit UTC timezone enforcement (safe for tz-naive and tz-aware)
    - Frequency validation (rejects mismatched data)
    - Delta-fetch (only fetches new data from exchange)
    - Comprehensive logging

    Attributes:
        settings: Configuration object.
        cache_path: Resolved path to cache directory.
    """

    # Expected time deltas for each timeframe
    TIMEFRAME_DELTAS = {
        "1d": timedelta(days=1),
        "4h": timedelta(hours=4),
        "1h": timedelta(hours=1),
        "15m": timedelta(minutes=15),
    }

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize DataLake.

        Args:
            settings: Configuration object. If None, uses default Settings.
        """
        self.settings = settings or Settings()
        self.cache_path = Path(self.settings.cache_dir).resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._exchange: Optional[ccxt.Exchange] = None

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazy-load CCXT exchange instance."""
        if self._exchange is None:
            self._exchange = getattr(ccxt, self.settings.exchange_id)()
        return self._exchange

    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """
        Returns path to Parquet file for a symbol/timeframe pair.

        Format: {cache_dir}/{symbol}_{timeframe}.parquet
        Example: data/cache/BTCUSDT_1d.parquet
        """
        safe_symbol = symbol.replace("/", "")
        return self.cache_path / f"{safe_symbol}_{timeframe}.parquet"

    def _enforce_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely enforce UTC timezone on DataFrame index.

        Logic:
        - If tz-naive → localize to UTC
        - If tz-aware → convert to UTC

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            DataFrame with UTC-localized index.
        """
        if df.empty:
            return df

        if df.index.tz is None:
            # tz-naive: localize to UTC
            df.index = df.index.tz_localize("UTC")
            logger.debug("[DataLake] Localized tz-naive index to UTC")
        elif str(df.index.tz) != "UTC":
            # tz-aware but not UTC: convert to UTC
            df.index = df.index.tz_convert("UTC")
            logger.debug(f"[DataLake] Converted index from {df.index.tz} to UTC")
        # else: already UTC, no action needed

        return df

    def _validate_frequency(self, df: pd.DataFrame, expected_tf: str) -> None:
        """
        Validate that data frequency matches expected timeframe.

        Calculates mode of time deltas and compares to expected.
        Raises FrequencyMismatchError if mismatch detected.

        Args:
            df: DataFrame with DatetimeIndex.
            expected_tf: Expected timeframe string ('1d', '4h', etc.)

        Raises:
            FrequencyMismatchError: If frequency doesn't match.
        """
        if len(df) < 2:
            return  # Cannot validate with < 2 rows

        expected_delta = self.TIMEFRAME_DELTAS.get(expected_tf)
        if expected_delta is None:
            logger.warning(f"[DataLake] Unknown timeframe: {expected_tf}, skipping validation")
            return

        # Calculate time differences
        time_diffs = df.index.to_series().diff().dropna()
        
        if time_diffs.empty:
            return

        # Get mode (most common delta)
        mode_delta = time_diffs.mode()
        if mode_delta.empty:
            return

        actual_delta = mode_delta.iloc[0]
        
        # Allow 10% tolerance for edge cases (DST, exchange maintenance gaps)
        tolerance = expected_delta * 0.1
        if abs(actual_delta - expected_delta) > tolerance:
            raise FrequencyMismatchError(
                f"Data frequency mismatch. Expected: {expected_delta}, "
                f"Got: {actual_delta}. Cache may be corrupted or from different timeframe."
            )

    def _log_dataframe_stats(
        self, symbol: str, df: pd.DataFrame, operation: str
    ) -> None:
        """Log DataFrame statistics (rows, timestamp range)."""
        if df.empty:
            logger.info(f"[DataLake] {symbol}: {operation} - Empty DataFrame")
            return

        start_ts = df.index.min()
        end_ts = df.index.max()
        logger.info(
            f"[DataLake] {symbol}: {operation} - "
            f"{len(df)} rows, range: {start_ts.strftime('%Y-%m-%d')} → {end_ts.strftime('%Y-%m-%d')}"
        )

    def load_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load cached data from Parquet file.

        Args:
            symbol: Trading pair symbol.

        Returns:
            DataFrame with UTC index, or None if no cache exists.
        """
        parquet_path = self._get_parquet_path(symbol, self.settings.timeframe)

        if not parquet_path.exists():
            logger.info(f"[DataLake] {symbol}: No cache found at {parquet_path}")
            return None

        try:
            df = pd.read_parquet(parquet_path)
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
                else:
                    logger.warning(f"[DataLake] {symbol}: Cache has invalid index structure")
                    return None

            # Enforce UTC
            df = self._enforce_utc(df)

            # Validate frequency
            self._validate_frequency(df, self.settings.timeframe)

            self._log_dataframe_stats(symbol, df, "Cache loaded")
            return df

        except FrequencyMismatchError as e:
            logger.error(f"[DataLake] {symbol}: {e}. Invalidating cache.")
            parquet_path.unlink()  # Delete corrupted cache
            return None
        except Exception as e:
            logger.error(f"[DataLake] {symbol}: Failed to load cache: {e}")
            return None

    def _fetch_from_exchange(
        self,
        symbol: str,
        since_timestamp: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange via CCXT.

        Args:
            symbol: Trading pair symbol.
            since_timestamp: Start timestamp in milliseconds. If None, fetches from settings.start_date.

        Returns:
            DataFrame with UTC index and OHLCV columns.
        """
        # Calculate start timestamp
        if since_timestamp is None:
            start_dt = datetime.strptime(self.settings.start_date, "%Y-%m-%d")
            start_dt = start_dt.replace(tzinfo=timezone.utc)
            start_dt -= timedelta(days=self.settings.warmup_days)
            since_timestamp = int(start_dt.timestamp() * 1000)

        limit = 1000
        all_candles = []

        logger.info(f"[DataLake] {symbol}: Fetching from exchange since {datetime.fromtimestamp(since_timestamp / 1000, tz=timezone.utc)}")

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.settings.timeframe,
                    since=since_timestamp,
                    limit=limit,
                )

                if not candles:
                    break

                all_candles.extend(candles)
                since_timestamp = candles[-1][0] + 1

                # Check if reached current time or end_date
                last_ts = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
                
                if self.settings.end_date:
                    end_dt = datetime.strptime(self.settings.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if last_ts >= end_dt:
                        break

                now_utc = datetime.now(timezone.utc)
                if last_ts >= now_utc - timedelta(hours=1):
                    break

                # Rate limiting
                import time
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                logger.error(f"[DataLake] {symbol}: Network error: {e}")
                raise
            except ccxt.ExchangeError as e:
                logger.error(f"[DataLake] {symbol}: Exchange error: {e}")
                raise

        if not all_candles:
            logger.warning(f"[DataLake] {symbol}: No candles fetched from exchange")
            return pd.DataFrame()

        # Build DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Filter by end_date
        if self.settings.end_date:
            end_dt = pd.Timestamp(self.settings.end_date, tz="UTC")
            df = df[df.index <= end_dt]

        self._log_dataframe_stats(symbol, df, "Fetched from exchange")
        return df

    def save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to Parquet cache.

        Args:
            symbol: Trading pair symbol.
            df: DataFrame with UTC index to save.
        """
        if df.empty:
            logger.warning(f"[DataLake] {symbol}: Skipping save of empty DataFrame")
            return

        parquet_path = self._get_parquet_path(symbol, self.settings.timeframe)
        
        # Ensure UTC before saving
        df = self._enforce_utc(df)
        
        df.to_parquet(parquet_path, engine="pyarrow", index=True)
        self._log_dataframe_stats(symbol, df, "Saved to cache")

    def load_or_fetch(self, symbol: str) -> pd.DataFrame:
        """
        Main entry point: Load from cache or fetch from exchange.

        Logic:
        1. Try to load cached Parquet
        2. If no cache → full fetch
        3. If cache exists → delta fetch (only new rows)
        4. Append and save

        Args:
            symbol: Trading pair symbol.

        Returns:
            Complete DataFrame with UTC index and OHLCV + returns columns.
        """
        cached_df = self.load_cached(symbol)

        if cached_df is None:
            # No cache: full fetch
            logger.info(f"[DataLake] {symbol}: No cache, performing full fetch")
            df = self._fetch_from_exchange(symbol)
        elif cached_df.empty:
            # Empty cache: full fetch
            df = self._fetch_from_exchange(symbol)
        else:
            # Cache exists: delta fetch
            last_cached_ts = cached_df.index.max()
            # Add 1 ms to avoid duplicate
            since_ts = int(last_cached_ts.timestamp() * 1000) + 1

            logger.info(f"[DataLake] {symbol}: Delta fetch from {last_cached_ts}")
            delta_df = self._fetch_from_exchange(symbol, since_timestamp=since_ts)

            if delta_df.empty:
                logger.info(f"[DataLake] {symbol}: Cache is up-to-date, no new data")
                df = cached_df
            else:
                # Append delta to cache
                pre_rows = len(cached_df)
                df = pd.concat([cached_df, delta_df])
                df = df[~df.index.duplicated(keep="last")]  # Remove any duplicates
                df = df.sort_index()
                post_rows = len(df)

                logger.info(
                    f"[DataLake] {symbol}: Appended {post_rows - pre_rows} new rows "
                    f"(pre: {pre_rows}, post: {post_rows})"
                )

        # Save updated cache
        if not df.empty:
            self.save_to_cache(symbol, df)

        # Calculate log returns
        if not df.empty and "returns" not in df.columns:
            df["returns"] = np.log(df["close"] / df["close"].shift(1))
            df.dropna(subset=["returns"], inplace=True)

        self._log_dataframe_stats(symbol, df, "Final output")
        return df
