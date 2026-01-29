import sys
import logging
import pandas as pd
import numpy as np
from config import Config
from data_fetcher import DataEngine, DataQualityError
from markov_model import RegimeModel
from var_model import RiskEngine
from analyze_log import AnalyticsHub

# Initialize Config as a singleton-like for main execution
config = Config()

def main():
    hub = AnalyticsHub(config)
    hub.log("=== STARTING ADAPTIVE RISK PIPELINE ===")
    
    try:
        # 1. Initialization
        data_engine = DataEngine(config)
        risk_engine = RiskEngine(config)
        regime_model = RegimeModel(config)
        
        # Data Containers
        main_df = None            # The primary OHLCV (Single or Synthetic Portfolio)
        components_df = None      # Individual asset returns (for Correlation Matrix)
        is_portfolio = bool(config.PORTFOLIO_ASSETS)

        # 2. Data Loading & Construction
        if is_portfolio:
            hub.log(f"--- PORTFOLIO MODE ACTIVATED ---")
            hub.log(f"Assets: {config.PORTFOLIO_ASSETS}")
            
            # A. Fetch All Assets
            raw_assets = data_engine.fetch_portfolio_data()
            if not raw_assets:
                hub.log("Critical: Failed to load portfolio assets.", "error")
                return

            # B. Construct Synthetic Portfolio (Historical Simulation)
            # This accounts for 'Linear Mix' of returns but preserves higher moments (fat tails)
            # inherently present in the historical series.
            hub.log("Constructing Synthetic Portfolio History...")
            main_df, components_df = risk_engine.construct_synthetic_portfolio(raw_assets)
            
            hub.log(f"Synthetic Portfolio constructed. Rows: {len(main_df)}")
            
        else:
            hub.log(f"--- SINGLE ASSET MODE ({config.SYMBOL}) ---")
            main_df, _ = data_engine.fetch_data_split()
            if main_df.empty:
                hub.log("Critical: No data fetched.", "error")
                return
        
        # 3. Regime Detection (HMM)
        # CRITICAL: We run HMM on the *Active Entity* (Portfolio or Single Asset).
        # This ensures 'Stress' is defined by OUR PnL volatility, not just BTC's.
        hub.log("Detecting Volatility Regimes (Markov Switching)...")
        regime_probs = regime_model.detect_regimes(main_df['returns'])
        
        # 4. Risk Calculation (Adaptive VaR)
        hub.log(f"Calculating VaR (Confidence: {config.VAR_CONFIDENCE*100}%, Capital: ${config.CAPITAL:,.0f})...")
        
        backtest_results = {}
        dist_data = {}
        
        for h in config.HORIZONS:
            # A. Backtest (Adaptive VaR)
            # The 'regime_probs' passed here dictate the lookback window (60 vs 252)
            # effectively conditioning VaR on the Portfolio's Volatility State.
            res_df = risk_engine.calc_adaptive_var(main_df, regime_probs, horizon=h)
            perf = risk_engine.get_performance_summary(res_df)
            backtest_results[h] = res_df
            
            # B. Raw Data (For Distribution Plot)
            raw_ret = risk_engine.get_raw_horizon_returns(main_df, horizon=h)
            dist_data[h] = raw_ret
            
            # Logging Stats
            bo_pct = perf['breakout_pct']
            target = perf['target_pct']
            
            status = "ACCEPTABLE"
            if bo_pct > 1.5 * target: status = "DANGEROUS"
            elif bo_pct < 0.5 * target: status = "TOO SAFE"
            elif 0.8 * target <= bo_pct <= 1.2 * target: status = "EXCELLENT"

            hub.log(f"Horizon {h}D | Breakouts: {bo_pct:5.2f}% (Target {target:.1f}%) | "
                    f"Avg VaR: ${abs(perf['avg_var_usd']):,.0f} | CVaR: ${abs(perf['cvar_realized_usd']):,.0f} | "
                    f"Drawdown: {perf['avg_dd_pct']:.2f}% | {status}")

        hub.log("="*80)

        # 5. Correlation Analysis (Portfolio Only)
        corr_matrix = None
        if is_portfolio and components_df is not None:
            hub.log(f"Calculating Rolling Correlation ({config.CORRELATION_WINDOW} days)...")
            corr_matrix = risk_engine.get_rolling_correlation(components_df)
            hub.log("Correlation Matrix Ready.")

        # 6. Visualization & Reporting
        # We select the shortest horizon (usually 1D) for the main time-series chart
        primary_horizon = config.HORIZONS[0]
        primary_results = backtest_results[primary_horizon]
        
        hub.plot_dashboard(
            results_df=primary_results,
            regime_probs=regime_probs,
            dist_data_dict=dist_data,
            corr_matrix=corr_matrix,
            title=f"Adaptive VaR Model | {'Portfolio' if is_portfolio else config.SYMBOL}"
        )
        
        hub.log("Pipeline Completed Successfully.")

    except KeyboardInterrupt:
        hub.log("Pipeline stopped by user.", "warning")
        sys.exit(0)
    except DataQualityError as e:
        hub.log(f"Critical Data Error: {e}", "critical")
        sys.exit(1)
    except ValueError as e:
        hub.log(f"Configuration or Data Error: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        hub.log(f"Model Computation Error: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Catch-all for truly unexpected errors
        hub.log(f"UNEXPECTED Pipeline Crash: {type(e).__name__}: {e}", "critical")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()