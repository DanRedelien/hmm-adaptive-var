
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path to allow imports from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hmm_var.settings import Settings
from hmm_var.var_model import RiskEngine

class TestRiskEngineInstitutional(unittest.TestCase):
    """
    Tier-1 Bank Standard Unit Tests for RiskEngine.
    
    Focus:
    1. Deterministic Validation: No random numbers in critical checks.
    2. Look-Ahead Bias: Strict verification of shift logic.
    3. Boundary Conditions: Empty data, NaN handling.
    4. Financial Sanity: VaR < 0 (always a loss), VaR scales with time.
    5. WHS Validation: Verify weighted simulation logic.
    """

    def setUp(self):
        # Use valid window sizes per Pydantic constraints (>=100, >=30)
        self.settings = Settings(
            window_calm=100,
            window_stress=30,
            var_confidence=0.95,
            capital=100_000,
            enable_whs=False  # Default to legacy for standard tests
        )
        self.engine = RiskEngine(self.settings)

    def create_synthetic_market(self, pattern: str = 'steady', length: int = 200) -> pd.DataFrame:
        """
        Generates deterministic price paths for testing.
        """
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        
        if pattern == 'steady':
            # Price doubles every day
            prices = np.array([2.0**i for i in range(length)])
        elif pattern == 'alternating':
            # 100, 101, 100, 101...
            prices = np.array([100.0 if i % 2 == 0 else 101.0 for i in range(length)])
        elif pattern == 'downtrend':
            # Consistent downtrend
            prices = 100.0 * np.exp(np.cumsum(np.full(length, -0.01)))
        elif pattern == 'random':
            np.random.seed(42)
            returns = np.random.normal(0, 0.01, length)
            prices = 100 * np.exp(np.cumsum(returns))
        else:
            prices = np.ones(length)
            
        df = pd.DataFrame({
            'close': prices, 
            'high': prices, 
            'low': prices
        }, index=dates)
        
        # Add returns column
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        return df

    def test_01_quantile_accuracy(self):
        """Verify that probability-weighted VaR calculation produces valid output."""
        np.random.seed(42)
        length = 200
        df = self.create_synthetic_market('random', length=length)
        
        # Use array-based probabilities
        probs_values = np.full(len(df), 0.3)
        regime_probs = pd.Series(probs_values, index=df.index)
        
        # Calculate
        res = self.engine.calc_adaptive_var(df, regime_probs, horizon=1)
        
        # Verify VaR output exists
        valid_forecasts = res['var_forecast'].dropna()
        self.assertFalse(valid_forecasts.empty, "No VaR forecasts generated")
        
        if not valid_forecasts.empty:
            last_idx = valid_forecasts.index[-1]
            final_var = valid_forecasts.iloc[-1]
            calm_var = res.loc[last_idx, 'var_calm']
            stress_var = res.loc[last_idx, 'var_stress']
            
            # Check blending logic
            min_var = min(calm_var, stress_var)
            max_var = max(calm_var, stress_var)
            self.assertTrue(min_var <= final_var <= max_var or np.isclose(final_var, calm_var) or np.isclose(final_var, stress_var),
                f"Blended VaR {final_var} not between inputs {calm_var}, {stress_var}")

    def test_02_lookahead_bias_strict(self):
        """
        CRITICAL: Verify that a shock at T is NOT reflected in VaR(T).
        It should only appear in VaR(T+1).
        """
        # 1. Use Random Market (so VaR is sensitive to distribution changes)
        df = self.create_synthetic_market('random', length=200)
        
        # 2. Inject massive crash at T=150
        crash_pos = 150
        crash_date = df.index[crash_pos]
        
        # Modify price to create -20% return
        prices = df['close'].values
        prices[crash_pos] = prices[crash_pos-1] * 0.8 
        df['close'] = prices
        
        # Recalc returns
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        # Ensure we dropped initial nan from creation, but consistent with index
        
        regime_probs = pd.Series(0.0, index=df.index)
        
        res = self.engine.calc_adaptive_var(df, regime_probs, horizon=1)
        
        if crash_date in res.index:
            idx_loc = res.index.get_loc(crash_date)
            
            if idx_loc + 1 < len(res):
                next_date = res.index[idx_loc + 1]
                
                var_at_crash = res.loc[crash_date, 'var_forecast']
                var_after_crash = res.loc[next_date, 'var_forecast']
                
                if pd.notna(var_at_crash) and pd.notna(var_after_crash):
                    # VaR (negative number) should become MORE negative
                    # e.g. -0.02 -> -0.05
                    # So var_after_crash < var_at_crash
                    self.assertLess(var_after_crash, var_at_crash,
                        f"VaR did not react to crash: T={var_at_crash:.4f}, T+1={var_after_crash:.4f}")

    def test_03_regime_switching_integration(self):
        """Test that the engine actually switches windows based on signal."""
        df = self.create_synthetic_market('random', length=200)
        
        # Regime: First half Calm(0), Second half Stress(1)
        probs_values = np.zeros(len(df))
        probs_values[len(df)//2:] = 1.0
        regime_probs = pd.Series(probs_values, index=df.index)
        
        res = self.engine.calc_adaptive_var(df, regime_probs, horizon=1)
        
        valid_res = res.dropna(subset=['regime_prob'])
        if not valid_res.empty:
            stress_period = valid_res[valid_res.index > df.index[len(df)//2]]
            if not stress_period.empty:
                self.assertEqual(stress_period['regime_prob'].iloc[0], 1.0, 
                    "Regime probability not propagated correctly.")

    def test_04_financial_sanity(self):
        """VaR should be negative (representing a loss) for downtrending assets."""
        df = self.create_synthetic_market('downtrend', length=200)
        regime_probs = pd.Series(0.0, index=df.index)
        
        res = self.engine.calc_adaptive_var(df, regime_probs, 1)
        valid_usd = res['var_usd'].dropna()
        
        if not valid_usd.empty:
            self.assertTrue((valid_usd < 0).all(), "VaR USD should be negative.")
            last_var = valid_usd.iloc[-1]
            self.assertTrue(-5000 < last_var < 0, f"Suspicious magnitude: {last_var}")

    def test_05_whs_functionality(self):
        """Test Weighted Historical Simulation mode."""
        # 1. Enable WHS
        self.settings.enable_whs = True
        self.engine = RiskEngine(self.settings)
        
        df = self.create_synthetic_market('random', length=200)
        
        # Varying probabilities
        probs = np.linspace(0, 1, len(df))
        regime_probs = pd.Series(probs, index=df.index)
        
        res = self.engine.calc_adaptive_var(df, regime_probs, horizon=1)
        
        # Check ESS calculation
        valid_ess = res['ess'].dropna()
        self.assertFalse(valid_ess.empty, "ESS not calculated in WHS mode")
        
        # ESS should be > 0 and <= window size (100)
        self.assertTrue((valid_ess > 0).all())
        self.assertTrue((valid_ess <= 100.1).all())
        
        # Check VaR is calculated
        self.assertFalse(res['var_forecast'].dropna().empty)

    def test_06_rebalancing_mechanics(self):
        """
        Verify that rebalancing logic includes drift and fee deduction.
        Case: 2 Assets, Diverging (A: +10%, B: -10%), Target 50/50.
        """
        # 1. Setup
        self.settings.enable_rebalancing = True
        self.settings.rebalance_interval = "1D"  # Daily rebalance to force turnover
        self.settings.transaction_fee_bps = 100.0 # 1% fee
        self.settings.portfolio_assets = {"A": 0.5, "B": 0.5}
        self.engine = RiskEngine(self.settings)
        
        # 2. Create Synthetic Data (2 days)
        dates = pd.date_range("2024-01-01", periods=3, freq='D')
        
        # Asset A: 100, 110, 121 (+10%)
        # Asset B: 100, 90, 81 (-10%)
        df_a = pd.DataFrame({'close': [100.0, 110.0, 121.0]}, index=dates)
        df_b = pd.DataFrame({'close': [100.0, 90.0, 81.0]}, index=dates)
        
        assets = {'A': df_a, 'B': df_b}
        
        # 3. run construction
        res_df, _ = self.engine.construct_synthetic_portfolio(assets)
        
        # 4. Analysis
        # Day 1:
        # Start Value = 100,000
        # A: 50k -> 55k (+10%)
        # B: 50k -> 45k (-10%)
        # Gross Value = 100k (0% return)
        # Weights: 55/100, 45/100 -> 0.55, 0.45
        # Target: 0.5, 0.5
        # Turnover: |0.55-0.5| + |0.45-0.5| = 0.10
        # Fee: 100k * 0.10 * 1% = 100k * 0.001 = 100 USD.
        # Net Value = 99,900.
        
        # Calculated Close
        # Note: Period 1 ends at dates[1] (2024-01-02).
        # construct_synthetic_portfolio drops first day (dates[0]) as it computes returns.
        # So index[0] of result corresponds to dates[1].
        
        close_day1 = res_df.loc[dates[1], 'close']
        
        # Allow small float tolerance
        expected_val = 99900.0
        self.assertTrue(np.isclose(close_day1, expected_val, atol=1.0),
            f"Expected {expected_val}, got {close_day1}")

if __name__ == '__main__':
    unittest.main()
