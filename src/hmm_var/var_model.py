"""
Core Risk Calculation Engine for the Adaptive VaR Risk Model.

Handles:
1. Adaptive VaR calculations (Calm/Stress windows).
2. Weighted Historical Simulation (WHS) with Regime Similarity.
3. Synthetic Portfolio construction (Historical Simulation).
4. Performance/Backtest metrics.
5. Correlation Analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from hmm_var.settings import Settings

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Core Risk Calculation Engine.
    
    Handles:
    1. Adaptive VaR calculations (Calm/Stress windows).
    2. Synthetic Portfolio construction (Historical Simulation).
    3. Performance/Backtest metrics.
    4. Correlation Analysis.
    """
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize RiskEngine.
        
        Args:
            settings: Configuration object containing risk parameters.
        """
        self.settings = settings
        self.alpha = 1 - settings.var_confidence

    def construct_synthetic_portfolio(
        self, assets_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructs a Synthetic Portfolio History based on current weights.
        
        Methodology:
        1. If enable_rebalancing=False (Standard VaR):
           - "Daily Reset" assumption: Weights are reset to target daily.
           - Returns = Sum(w_target * r_asset).
           - Consistent with standard parametric/HS VaR assumptions (static weights).
           
        2. If enable_rebalancing=True (Realistic Backtest):
           - "Drift" assumption: Weights drift with market moves.
           - Periodic Rebalancing (e.g., Monthly) with Transaction Fees.
           - Captures drag from fees and drift risk (concentration in winners).
           
        CRITICAL MATH NOTE:
        You CANNOT sum weighted log returns directly!
        ln(1 + Σw_i*R_i) ≠ Σw_i*ln(1+R_i)
        This approximation error explodes during stress regimes (-20% moves).
        
        Args:
            assets_data: Dictionary {Symbol: DataFrame} containing OHLCV.
            
        Returns:
            Tuple(portfolio_df, components_returns_df):
                - portfolio_df: DataFrame with 'returns' (log), 'close' (synthetic).
                - components_returns_df: DataFrame with individual asset SIMPLE returns.
        """
        if not self.settings.portfolio_assets:
            raise ValueError("Portfolio Mode invoked but no assets defined in Config.")

        # 1. Align Data (Inner Join) using SIMPLE returns
        aligned_simple_returns = pd.DataFrame()
        
        first_asset = True
        for symbol, df in assets_data.items():
            asset_ret = pd.DataFrame(index=df.index)
            # SIMPLE returns: R = (P_t / P_{t-1}) - 1
            asset_ret[symbol] = (df['close'] / df['close'].shift(1)) - 1
            
            if first_asset:
                aligned_simple_returns = asset_ret
                first_asset = False
            else:
                aligned_simple_returns = aligned_simple_returns.join(asset_ret, how='inner')
        
        aligned_simple_returns.dropna(inplace=True)

        # 2. Check Rebalancing Mode
        if self.settings.enable_rebalancing:
             # Realistic Mode: Drift + Periodic Rebalancing + Fees
            return self._construct_dynamic_portfolio(aligned_simple_returns)
        
        # 3. Standard VaR Mode: Daily Reset to Target (Static Weights)
        portfolio_simple_returns = pd.Series(0.0, index=aligned_simple_returns.index)
        
        for symbol, weight in self.settings.portfolio_assets.items():
            if symbol in aligned_simple_returns.columns:
                portfolio_simple_returns += aligned_simple_returns[symbol] * weight
            else:
                raise ValueError(f"Asset {symbol} missing in aligned data.")

        # 4. Convert to Log Returns for HMM/VaR compatibility
        # Guard against extreme negative returns (can't take log of <=0)
        portfolio_log_returns = np.log(1 + portfolio_simple_returns.clip(lower=-0.9999))
        
        # 5. Construct Synthetic Price Series
        cumulative_log_returns = portfolio_log_returns.cumsum()
        synthetic_prices = self.settings.capital * np.exp(cumulative_log_returns)
        
        # 6. Build Output DataFrame
        portfolio_df = pd.DataFrame(index=aligned_simple_returns.index)
        portfolio_df['returns'] = portfolio_log_returns
        portfolio_df['close'] = synthetic_prices
        
        return portfolio_df, aligned_simple_returns

    def _construct_dynamic_portfolio(
        self,
        aligned_returns: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructs portfolio with Drift, Periodic Rebalancing, and Transaction Fees.
        
        Methodology:
        1. Initialize positions at Target Weights.
        2. Let positions drift with market sets between rebalance dates.
        3. On Rebalance Date:
           - Calc Turnover = sum(|current_weight - target_weight|)
           - Deduct Fee = Value * Turnover * FeeBps
           - Reset to Target Weights
           
        Args:
            aligned_returns: DataFrame of component SIMPLE returns.
            
        Returns:
            Tuple(portfolio_df, aligned_returns)
        """
        # Configuration
        targets = self.settings.portfolio_assets
        interval = self.settings.rebalance_interval
        fee_rate = self.settings.transaction_fee_bps / 10000.0  # bps to decimal
        
        # We process manually to track exact value path
        # Map targets to array for speed
        assets = list(aligned_returns.columns)
        target_weights = np.array([targets.get(a, 0.0) for a in assets])
        
        # Validate assets match targets
        if not np.isclose(target_weights.sum(), 1.0, atol=0.01):
             # If subset of assets, re-normalize or warn? 
             # Assuming aligned_returns contains all portfolio assets as per validation
             pass

        current_value = self.settings.capital
        
        # Result containers
        portfolio_values = []
        portfolio_indices = []
        
        # Group by rebalance interval
        # If interval is trivial or None, we treat whole history as one period (Drift only)
        # Using pd.Grouper to segment time
        try:
            grouper = aligned_returns.groupby(pd.Grouper(freq=interval))
        except ValueError as e:
            logger.warning(f"Invalid rebalance interval '{interval}', defaulting to Monthly '1ME'")
            grouper = aligned_returns.groupby(pd.Grouper(freq='1ME'))

        for period_end, chunk in grouper:
            if chunk.empty:
                continue
                
            # 1. Initialize positions for this chunk (from current_value)
            # positions: array of value allocated to each asset
            positions = current_value * target_weights
            
            # 2. Calculate growth for this chunk (Vectorized within chunk)
            # chunk_returns + 1 -> cumulative prd
            # (1 + r).cumprod()
            cumulative_growth = (1 + chunk).cumprod()
            
            # Position values over time in this chunk
            # shape: (n_days, n_assets)
            chunk_positions = positions * cumulative_growth
            
            # Portfolio value sum for each day
            chunk_values = chunk_positions.sum(axis=1)
            
            # Store results
            portfolio_values.append(chunk_values)
            portfolio_indices.append(chunk.index)
            
            # 3. End of Period Rebalance Logic
            # Get values at end of chunk
            end_positions = chunk_positions.iloc[-1].values
            pv_end = end_positions.sum()
            
            # Calculate drifted weights
            drifted_weights = end_positions / pv_end
            
            # Calculate Turnover: sum(|w_drift - w_target|)
            turnover = np.sum(np.abs(drifted_weights - target_weights))
            
            # Calculate Fee
            fee_cost = pv_end * turnover * fee_rate
            
            # Update Current Value (after fee)
            current_value = pv_end - fee_cost
            
            # Apply fee impact to the LAST day's recorded value?
            # Standard practice: The rebalance happens "on" the close, so the close price
            # for that day effectively includes the fee deduction or the open next day does.
            # To be conservative/visible, we reduce the closing value of the rebalance date.
            portfolio_values[-1].iloc[-1] = current_value

        # Concatenate all chunks
        full_idx = pd.DatetimeIndex(np.concatenate([idx for idx in portfolio_indices]))
        full_vals = np.concatenate([v.values for v in portfolio_values])
        
        # Create Series
        portfolio_series = pd.Series(full_vals, index=full_idx).sort_index()
        
        # Remove duplicates if any (grouped chunks shouldn't overlap but good safety)
        portfolio_series = portfolio_series[~portfolio_series.index.duplicated(keep='last')]
        
        # Compute Log Returns from Value Series: r_t = ln(V_t / V_{t-1})
        portfolio_log_returns = np.log(portfolio_series / portfolio_series.shift(1))
        
        # Build DataFrame
        portfolio_df = pd.DataFrame(index=portfolio_series.index)
        portfolio_df['returns'] = portfolio_log_returns
        portfolio_df['close'] = portfolio_series
        
        return portfolio_df, aligned_returns

    def get_rolling_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Correlation Matrix over the *last* N days.
        
        Used for visualization (Heatmap), NOT for VaR calculation.
        
        Args:
            returns_df: DataFrame with asset returns as columns.
            
        Returns:
            Correlation matrix as DataFrame.
        """
        window = self.settings.correlation_window
        
        if len(returns_df) < window:
            return returns_df.corr()
            
        recent_slice = returns_df.iloc[-window:]
        return recent_slice.corr()

    def get_raw_horizon_returns(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Returns raw returns series for N days (for distribution plotting).
        
        Args:
            df: DataFrame with 'close' column.
            horizon: Holding period in days.
            
        Returns:
            Series of log returns over the horizon.
        """
        return np.log(df['close'] / df['close'].shift(horizon)).dropna()

    def _calc_similarity_weights(
        self,
        current_prob: float,
        historical_probs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Regime Similarity Weights for Weighted Historical Simulation.
        
        Methodology (Variant A - Similarity):
        - Similarity_k = 1 - |P(Stress_t) - P(Stress_{t-k})|
        - Weight w_k ∝ Similarity_k (normalized to sum to 1)
        
        Financial Rationale:
        Days with similar regime probability to today are more relevant for
        risk estimation. This weights recent similar regimes higher than
        distant dissimilar ones, providing adaptive tail estimation.
        
        Args:
            current_prob: Current P(Stress) at time t.
            historical_probs: Array of historical P(Stress) values from t-k.
            
        Returns:
            Normalized weight array (sums to 1.0).
        """
        # Similarity = 1 - |P(Stress_t) - P(Stress_{t-k})|
        similarities = 1.0 - np.abs(current_prob - historical_probs)
        
        # Ensure non-negative (edge case protection)
        similarities = np.maximum(similarities, 0.0)
        
        # Normalize to sum to 1
        total = similarities.sum()
        if total > 0:
            weights = similarities / total
        else:
            # Fallback: equal weights if all zeros (degenerate case)
            weights = np.ones(len(similarities)) / len(similarities)
        
        return weights

    def _calc_effective_sample_size(self, weights: np.ndarray) -> float:
        """
        Calculate Effective Sample Size (ESS) for weight quality monitoring.
        
        Formula: ESS = 1 / Σ(w_i^2)
        
        Financial Rationale:
        ESS measures how concentrated the weights are. If ESS is much smaller
        than the actual sample size, it indicates that only a few observations
        are effectively contributing to the VaR estimate, which can lead to
        unstable risk estimates.
        
        Args:
            weights: Normalized weight array (sums to 1.0).
            
        Returns:
            Effective Sample Size.
        """
        sum_squared = np.sum(weights ** 2)
        if sum_squared > 0:
            return 1.0 / sum_squared
        return 0.0

    def _calc_weighted_quantile(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        quantile: float
    ) -> Tuple[float, float]:
        """
        Calculate weighted VaR and CVaR from the weighted return distribution.
        
        Uses weighted percentile calculation via linear interpolation.
        
        Args:
            returns: Historical returns array.
            weights: Normalized weights array (same length as returns).
            quantile: Target quantile (e.g., 0.05 for 95% VaR).
            
        Returns:
            Tuple(var_value, cvar_value).
        """
        # Sort returns and corresponding weights
        sorted_indices = np.argsort(returns)
        sorted_returns = returns[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        
        # Find weighted quantile via interpolation
        var_idx = np.searchsorted(cumsum_weights, quantile)
        
        # Handle edge cases
        if var_idx >= len(sorted_returns):
            var_idx = len(sorted_returns) - 1
        if var_idx == 0:
            var_value = sorted_returns[0]
        else:
            # Linear interpolation between adjacent values
            w0 = cumsum_weights[var_idx - 1]
            w1 = cumsum_weights[var_idx]
            if w1 > w0:
                t = (quantile - w0) / (w1 - w0)
                var_value = (1 - t) * sorted_returns[var_idx - 1] + t * sorted_returns[var_idx]
            else:
                var_value = sorted_returns[var_idx]
        
        # Weighted CVaR: weighted mean of returns <= VaR
        tail_mask = sorted_returns <= var_value
        if tail_mask.sum() > 0:
            tail_weights = sorted_weights[tail_mask]
            tail_returns = sorted_returns[tail_mask]
            # Renormalize tail weights
            tail_sum = tail_weights.sum()
            if tail_sum > 0:
                cvar_value = np.sum(tail_returns * tail_weights) / tail_sum
            else:
                cvar_value = var_value
        else:
            cvar_value = var_value
        
        return var_value, cvar_value


    def calc_adaptive_var(
        self,
        df: pd.DataFrame,
        regime_probs: pd.Series,
        horizon: int
    ) -> pd.DataFrame:
        """
        Calculates Adaptive VaR using Historical Simulation.
        
        METHODOLOGY:
        1. If enable_whs=True (Weighted Historical Simulation):
           - Uses single lookback window (window_calm).
           - Weights historical returns by regime similarity.
           - Similarity_k = 1 - |P(Stress_t) - P(Stress_{t-k})|
           - Calculates weighted quantile for VaR/CVaR.
        
        2. If enable_whs=False (Legacy Regime Blending):
           - VaR_calm using WINDOW_CALM (365 days) - stable, Basel-standard
           - VaR_stress using WINDOW_STRESS (180 days) - reactive, crisis-aware
           - Blend: VaR_t = P_calm * VaR_calm + P_stress * VaR_stress
        
        Args:
            df: DataFrame with 'returns' and 'close' (Synthetic or Single).
            regime_probs: Series of Stress Probabilities P(Stress) from HMM (0..1).
            horizon: Holding period in days.
            
        Returns:
            DataFrame with columns: realized_return, var_forecast, cvar_forecast,
                                    var_calm, var_stress, regime_prob, var_usd, ess, etc.
        """
        # Realized Forward Return (for Backtesting comparison)
        future_close = df['close'].shift(-horizon)
        data = pd.DataFrame(index=df.index)
        data['realized_return'] = np.log(future_close / df['close'])
        data['close'] = df['close']
        data['returns'] = df['returns']
        
        # Output columns
        data['var_forecast'] = np.nan
        data['cvar_forecast'] = np.nan
        data['var_calm'] = np.nan
        data['var_stress'] = np.nan
        data['regime_prob'] = np.nan
        data['ess'] = np.nan  # Effective Sample Size (WHS mode only)
        
        # Start iteration after window has sufficient data
        start_idx = max(self.settings.window_calm, self.settings.window_stress) + horizon
        
        use_whs = self.settings.enable_whs
        
        if use_whs:
            logger.info(
                f"[VaR] Using Weighted Historical Simulation (WHS) with "
                f"window={self.settings.window_calm}, ESS threshold={self.settings.ess_threshold}"
            )
        else:
            logger.info("[VaR] Using Legacy Regime Blending (Calm/Stress windows)")
        
        for i in range(start_idx, len(df)):
            curr_date = df.index[i]
            if curr_date not in regime_probs.index:
                continue
            
            stress_prob = regime_probs.loc[curr_date]
            calm_prob = 1.0 - stress_prob
            data.loc[curr_date, 'regime_prob'] = stress_prob
            
            # 1. ALWAYS calculate Legacy VaR (Reference/Visualization)
            var_calm, cvar_calm = self._calc_var_for_window(
                df, i, self.settings.window_calm, horizon
            )
            var_stress, cvar_stress = self._calc_var_for_window(
                df, i, self.settings.window_stress, horizon
            )
            
            # Store legacy baselines if available
            if var_calm is not None:
                data.loc[curr_date, 'var_calm'] = var_calm
            if var_stress is not None:
                data.loc[curr_date, 'var_stress'] = var_stress

            if use_whs:
                # WHS Mode: Use similarity-weighted historical returns
                whs_result = self._calc_whs_var(
                    df, i, self.settings.window_calm, horizon,
                    stress_prob, regime_probs
                )
                
                if whs_result[0] is None:
                    # Fallback to legacy blending if WHS fails
                    if var_calm is None or var_stress is None:
                        continue
                    
                    blended_var = calm_prob * var_calm + stress_prob * var_stress
                    blended_cvar = calm_prob * cvar_calm + stress_prob * cvar_stress
                else:
                    var_val, cvar_val, ess = whs_result
                    blended_var = var_val
                    blended_cvar = cvar_val
                    data.loc[curr_date, 'ess'] = ess
            else:
                # Legacy Mode: Probability-weighted blend
                if var_calm is None or var_stress is None:
                    continue
                
                blended_var = calm_prob * var_calm + stress_prob * var_stress
                blended_cvar = calm_prob * cvar_calm + stress_prob * cvar_stress
            
            data.loc[curr_date, 'var_forecast'] = blended_var
            data.loc[curr_date, 'cvar_forecast'] = blended_cvar
            
        # Convert to USD
        data['var_usd'] = self.settings.capital * (np.exp(data['var_forecast']) - 1)
        data['realized_usd'] = self.settings.capital * (np.exp(data['realized_return']) - 1)

        return data
    
    def _calc_var_for_window(
        self, 
        df: pd.DataFrame, 
        current_idx: int, 
        window: int, 
        horizon: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Helper: Calculates VaR and CVaR for a specific lookback window.
        
        Args:
            df: Price DataFrame.
            current_idx: Current position in DataFrame.
            window: Lookback window size (e.g., 365 or 180).
            horizon: Holding period in days.
            
        Returns:
            Tuple(var_value, cvar_value) or (None, None) if insufficient data.
        """
        history_start = current_idx - window - horizon
        if history_start < 0:
            return None, None
        
        hist_closes = df['close'].iloc[history_start:current_idx].values
        
        if len(hist_closes) <= horizon:
            return None, None
        
        # Calculate overlapping horizon returns: ln(P_t / P_{t-h})
        hist_series = pd.Series(hist_closes)
        hist_h_returns = np.log(hist_series / hist_series.shift(horizon)).dropna().values
        
        if len(hist_h_returns) < 10:
            return None, None
        
        # VaR: Left-tail percentile
        var_val = np.percentile(hist_h_returns, (1 - self.settings.var_confidence) * 100)
        
        # CVaR (Expected Shortfall): Mean of returns below VaR
        tail_losses = hist_h_returns[hist_h_returns <= var_val]
        cvar_val = tail_losses.mean() if len(tail_losses) > 0 else var_val
        
        return var_val, cvar_val

    def _calc_whs_var(
        self,
        df: pd.DataFrame,
        current_idx: int,
        window: int,
        horizon: int,
        current_prob: float,
        regime_probs: pd.Series
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate VaR using Weighted Historical Simulation with Regime Similarity.
        
        Methodology (Variant A - Similarity Weighting):
        1. Extract historical returns over lookback window.
        2. Get historical P(Stress) for each observation.
        3. Calculate similarity weights: w_k = 1 - |P(Stress_t) - P(Stress_{t-k})|
        4. Normalize weights and calculate weighted quantile.
        5. Monitor ESS and log warning if below threshold.
        
        Financial Rationale:
        Traditional Historical Simulation treats all observations equally,
        ignoring regime context. WHS weights observations by how similar
        their market regime was to today, making tail estimates more
        relevant to current conditions.
        
        Args:
            df: Price DataFrame.
            current_idx: Current position in DataFrame.
            window: Lookback window size.
            horizon: Holding period in days.
            current_prob: Current P(Stress) at time t.
            regime_probs: Full series of historical P(Stress) values.
            
        Returns:
            Tuple(var_value, cvar_value, ess) or (None, None, None) if insufficient data.
        """
        history_start = current_idx - window - horizon
        if history_start < 0:
            return None, None, None
        
        hist_closes = df['close'].iloc[history_start:current_idx].values
        hist_dates = df.index[history_start:current_idx]
        
        if len(hist_closes) <= horizon:
            return None, None, None
        
        # Calculate overlapping horizon returns
        hist_series = pd.Series(hist_closes)
        hist_h_returns_raw = np.log(hist_series / hist_series.shift(horizon))
        
        # Get corresponding dates for valid returns (after shift)
        valid_mask = ~hist_h_returns_raw.isna()
        hist_h_returns = hist_h_returns_raw[valid_mask].values
        valid_dates = hist_dates[valid_mask.values]
        
        if len(hist_h_returns) < 10:
            return None, None, None
        
        # Get historical regime probabilities for these dates
        historical_probs = []
        valid_indices = []
        
        for idx, date in enumerate(valid_dates):
            if date in regime_probs.index:
                historical_probs.append(regime_probs.loc[date])
                valid_indices.append(idx)
        
        if len(valid_indices) < 10:
            # Not enough historical probs, fallback to unweighted
            return None, None, None
        
        # Filter returns and probs to matched observations
        matched_returns = hist_h_returns[valid_indices]
        matched_probs = np.array(historical_probs)
        
        # Calculate similarity weights
        weights = self._calc_similarity_weights(current_prob, matched_probs)
        
        # Calculate ESS
        ess = self._calc_effective_sample_size(weights)
        
        # ESS check and warning
        if ess < self.settings.ess_threshold:
            logger.warning(
                f"[WHS] Low ESS detected: {ess:.1f} < {self.settings.ess_threshold}. "
                f"Risk estimates may be unstable due to weight concentration."
            )
        
        # Calculate weighted VaR and CVaR
        var_val, cvar_val = self._calc_weighted_quantile(
            matched_returns,
            weights,
            self.alpha
        )
        
        return var_val, cvar_val, ess


    def get_performance_summary(self, data: pd.DataFrame) -> Dict:
        """
        Calculates backtest statistics (Breakouts, Drawdowns, etc).
        
        Args:
            data: DataFrame from calc_adaptive_var.
            
        Returns:
            Dictionary with performance metrics.
        """
        if data.empty:
            return {}
        
        valid_data = data.dropna(subset=['var_forecast', 'realized_return'])
        breakouts = valid_data[valid_data['realized_return'] < valid_data['var_forecast']]
        
        total = len(valid_data)
        count = len(breakouts)
        pct = (count / total) * 100 if total > 0 else 0
        target = (1 - self.settings.var_confidence) * 100
        
        cvar_realized_usd = breakouts['realized_usd'].mean() if count > 0 else 0
        avg_var_usd = valid_data['var_usd'].mean()

        neg_ret = valid_data[valid_data['realized_return'] < 0]['realized_return']
        avg_dd_pct = (np.exp(neg_ret.mean()) - 1) * 100 if not neg_ret.empty else 0.0

        return {
            'total_obs': total,
            'breakouts': count,
            'breakout_pct': pct,
            'target_pct': target,
            'avg_var_usd': avg_var_usd,
            'cvar_realized_usd': cvar_realized_usd,
            'avg_dd_pct': avg_dd_pct
        }
