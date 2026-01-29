import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from config import Config

class RiskEngine:
    """
    Core Risk Calculation Engine.
    
    Handles:
    1. Adaptive VaR calculations (Calm/Stress windows).
    2. Synthetic Portfolio construction (Historical Simulation).
    3. Performance/Backtest metrics.
    4. Correlation Analysis.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.alpha = 1 - cfg.VAR_CONFIDENCE

    def construct_synthetic_portfolio(self, assets_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructs a Synthetic Portfolio History based on current weights.
        
        Methodology (Mathematically Correct Full Revaluation):
        1. Inner Join all asset dates to ensure alignment.
        2. Calculate SIMPLE Returns for each asset: R_i = (P_t / P_{t-1}) - 1
        3. Compute Weighted Portfolio Simple Return: R_port = Sum(w_i * R_i)
        4. Convert to Log Return for HMM compatibility: ln(1 + R_port)
        5. Reconstruct Synthetic Price from cumulative simple returns.
        
        CRITICAL MATH NOTE:
        You CANNOT sum weighted log returns directly!
        ln(1 + Σw_i*R_i) ≠ Σw_i*ln(1+R_i)
        This approximation error explodes during stress regimes (-20% moves).
        
        Args:
            assets_data: Dictionary {Symbol: DataFrame} containing OHLCV.
            
        Returns:
            Tuple(portfolio_df, components_returns_df):
                - portfolio_df: DataFrame with 'returns' (log), 'close' (synthetic).
                - components_returns_df: DataFrame with individual asset SIMPLE returns (for correlation).
        """
        if not self.cfg.PORTFOLIO_ASSETS:
            raise ValueError("Portfolio Mode invoked but no assets defined in Config.")

        # 1. Align Data (Inner Join) using SIMPLE returns
        aligned_simple_returns = pd.DataFrame()
        
        first_asset = True
        for symbol, df in assets_data.items():
            asset_ret = pd.DataFrame(index=df.index)
            # SIMPLE returns: R = (P_t / P_{t-1}) - 1
            # This is the mathematically correct form for portfolio aggregation
            asset_ret[symbol] = (df['close'] / df['close'].shift(1)) - 1
            
            if first_asset:
                aligned_simple_returns = asset_ret
                first_asset = False
            else:
                aligned_simple_returns = aligned_simple_returns.join(asset_ret, how='inner')
        
        aligned_simple_returns.dropna(inplace=True)
        
        # 2. Apply Weights to SIMPLE Returns (Mathematically Correct)
        # Formula: R_portfolio,t = Σ(w_i * R_i,t)
        # This is the ONLY correct way to aggregate portfolio returns
        portfolio_simple_returns = pd.Series(0.0, index=aligned_simple_returns.index)
        
        for symbol, weight in self.cfg.PORTFOLIO_ASSETS.items():
            if symbol in aligned_simple_returns.columns:
                portfolio_simple_returns += aligned_simple_returns[symbol] * weight
            else:
                raise ValueError(f"Asset {symbol} missing in aligned data.")

        # 3. Convert to Log Returns for HMM/VaR compatibility
        # CORRECT: ln(1 + R_simple) where R_simple is the weighted portfolio return
        # Guard against extreme negative returns (can't take log of <=0)
        portfolio_log_returns = np.log(1 + portfolio_simple_returns.clip(lower=-0.9999))
        
        # 4. Construct Synthetic Price Series
        # Use cumulative SIMPLE returns for price reconstruction (more accurate)
        # P_t = P_0 * Π(1 + R_i) = P_0 * exp(Σ ln(1 + R_i))
        cumulative_log_returns = portfolio_log_returns.cumsum()
        synthetic_prices = self.cfg.CAPITAL * np.exp(cumulative_log_returns)
        
        # 5. Build Output DataFrame
        portfolio_df = pd.DataFrame(index=aligned_simple_returns.index)
        portfolio_df['returns'] = portfolio_log_returns  # Log returns for HMM
        portfolio_df['close'] = synthetic_prices
        
        # Return simple returns for correlation analysis (more intuitive)
        return portfolio_df, aligned_simple_returns

    def get_rolling_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Correlation Matrix over the *last* N days (Rolling Window).
        
        Used for visualization (Heatmap), NOT for VaR calculation.
        """
        window = self.cfg.CORRELATION_WINDOW
        
        if len(returns_df) < window:
            return returns_df.corr()
            
        # Select last N days
        recent_slice = returns_df.iloc[-window:]
        return recent_slice.corr()

    def get_raw_horizon_returns(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Returns raw returns series for N days (for distribution plotting).
        """
        # ln(P_t / P_{t-n})
        return np.log(df['close'] / df['close'].shift(horizon)).dropna()

    def calc_adaptive_var(self, df: pd.DataFrame, regime_probs: pd.Series, horizon: int) -> pd.DataFrame:
        """
        Calculates Probability-Weighted Adaptive VaR using Historical Simulation.
        
        METHODOLOGY (Continuous Regime Blending):
        Instead of hard-switching between windows based on a threshold, we:
        1. Calculate VaR_calm using WINDOW_CALM (252 days) - stable, Basel-standard
        2. Calculate VaR_stress using WINDOW_STRESS (60 days) - reactive, crisis-aware
        3. Blend: VaR_t = P_calm * VaR_calm + P_stress * VaR_stress
        
        This approach:
        - Produces CONTINUOUS, STABLE risk limits (no flickering)
        - Is mathematically correct (weighted average of quantiles)
        - Respects trader preferences (no sudden limit changes)
        - Uses HMM probabilities directly without arbitrary thresholds
        
        Args:
            df: DataFrame with 'returns' and 'close' (Synthetic or Single).
            regime_probs: Series of Stress Probabilities P(Stress) from HMM (0..1).
            horizon: Holding period in days.
            
        Returns:
            DataFrame with columns: realized_return, var_forecast, cvar_forecast,
                                    var_calm, var_stress, regime_prob, var_usd, etc.
        """
        # Realized Forward Return (for Backtesting comparison)
        # Shift(-H) aligns "Return over next H days" to "Today's row"
        future_close = df['close'].shift(-horizon)
        data = pd.DataFrame(index=df.index)
        data['realized_return'] = np.log(future_close / df['close'])
        data['close'] = df['close']
        data['returns'] = df['returns']
        
        # Output columns
        data['var_forecast'] = np.nan      # Final blended VaR
        data['cvar_forecast'] = np.nan     # Final blended CVaR
        data['var_calm'] = np.nan          # VaR from calm window (for diagnostics)
        data['var_stress'] = np.nan        # VaR from stress window (for diagnostics)
        data['regime_prob'] = np.nan       # P(Stress) for transparency
        
        # Start iteration after both windows have sufficient data
        start_idx = max(self.cfg.WINDOW_CALM, self.cfg.WINDOW_STRESS) + horizon
        
        for i in range(start_idx, len(df)):
            curr_date = df.index[i]
            if curr_date not in regime_probs.index:
                continue
            
            # Get stress probability from HMM
            stress_prob = regime_probs.loc[curr_date]
            calm_prob = 1.0 - stress_prob
            data.loc[curr_date, 'regime_prob'] = stress_prob
            
            # ====== CALCULATE BOTH VaRs IN PARALLEL ======
            
            # A. VaR_calm (Long window = stable, Basel-standard)
            var_calm, cvar_calm = self._calc_var_for_window(
                df, i, self.cfg.WINDOW_CALM, horizon
            )
            
            # B. VaR_stress (Short window = reactive, crisis-aware)
            var_stress, cvar_stress = self._calc_var_for_window(
                df, i, self.cfg.WINDOW_STRESS, horizon
            )
            
            if var_calm is None or var_stress is None:
                continue
            
            # Store component VaRs for diagnostic/visualization
            data.loc[curr_date, 'var_calm'] = var_calm
            data.loc[curr_date, 'var_stress'] = var_stress
            
            # ====== PROBABILITY-WEIGHTED BLEND ======
            # VaR_t = P_calm * VaR_calm + P_stress * VaR_stress
            # 
            # Note: Since VaR values are negative (losses), this weighted average
            # correctly interpolates between the two risk estimates.
            # When P_stress is high, we weight more toward the reactive (usually worse) VaR.
            
            blended_var = calm_prob * var_calm + stress_prob * var_stress
            blended_cvar = calm_prob * cvar_calm + stress_prob * cvar_stress
            
            data.loc[curr_date, 'var_forecast'] = blended_var
            data.loc[curr_date, 'cvar_forecast'] = blended_cvar
            
        # Convert to USD
        data['var_usd'] = self.cfg.CAPITAL * (np.exp(data['var_forecast']) - 1)
        data['realized_usd'] = self.cfg.CAPITAL * (np.exp(data['realized_return']) - 1)

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
            window: Lookback window size (e.g., 252 or 60).
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
        var_val = np.percentile(hist_h_returns, (1 - self.cfg.VAR_CONFIDENCE) * 100)
        
        # CVaR (Expected Shortfall): Mean of returns below VaR
        tail_losses = hist_h_returns[hist_h_returns <= var_val]
        cvar_val = tail_losses.mean() if len(tail_losses) > 0 else var_val
        
        return var_val, cvar_val

    def get_performance_summary(self, data: pd.DataFrame):
        """Calculates backtest statistics (Breakouts, Drawdowns, etc)."""
        if data.empty: return {}
        
        valid_data = data.dropna(subset=['var_forecast', 'realized_return'])
        breakouts = valid_data[valid_data['realized_return'] < valid_data['var_forecast']]
        
        total = len(valid_data)
        count = len(breakouts)
        pct = (count / total) * 100 if total > 0 else 0
        target = (1 - self.cfg.VAR_CONFIDENCE) * 100
        
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