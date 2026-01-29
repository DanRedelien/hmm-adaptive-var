"""
Main entry point for the Adaptive VaR Risk Model.

This module orchestrates the risk analysis pipeline:
1. Data loading (single asset or portfolio) with Parquet caching
2. Regime detection (HMM)
3. VaR calculation (probability-weighted)
4. Visualization and reporting
"""

import logging
import sys

import numpy as np
import pandas as pd

from hmm_var.analytics import AnalyticsHub
from hmm_var.data_fetcher import DataEngine, DataQualityError
from hmm_var.data_lake import DataLake
from hmm_var.markov_model import RegimeModel
from hmm_var.settings import Settings
from hmm_var.var_model import RiskEngine
from hmm_var.visualizer import Visualizer

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)


def main() -> None:
    """
    Main pipeline entry point.
    
    Orchestrates the full risk analysis workflow:
    1. Load configuration
    2. Fetch market data (with Parquet caching)
    3. Detect market regimes
    4. Calculate adaptive VaR
    5. Generate visualizations
    """
    # Initialize settings
    settings = Settings()
    
    hub = AnalyticsHub(settings)
    visualizer = Visualizer(settings)
    
    hub.log("=== STARTING ADAPTIVE RISK PIPELINE ===")
    
    try:
        # 1. Initialization
        data_lake = DataLake(settings)  # Parquet cache with delta-fetch
        risk_engine = RiskEngine(settings)
        regime_model = RegimeModel(settings)
        
        # Data Containers
        main_df = None
        components_df = None
        is_portfolio = bool(settings.portfolio_assets)

        # 2. Data Loading & Construction (with Parquet caching)
        if is_portfolio:
            hub.log("--- PORTFOLIO MODE ACTIVATED ---")
            hub.log(f"Assets: {list(settings.portfolio_assets.keys())}")
            
            # Load each asset via DataLake (cached)
            raw_assets = {}
            for symbol in settings.portfolio_assets.keys():
                hub.log(f"Loading {symbol} (with cache)...")
                df = data_lake.load_or_fetch(symbol)
                if not df.empty:
                    raw_assets[symbol] = df
                else:
                    hub.log(f"Warning: {symbol} returned empty data", "warning")
            
            if not raw_assets:
                hub.log("Critical: Failed to load portfolio assets.", "error")
                return

            hub.log("Constructing Synthetic Portfolio History...")
            main_df, components_df = risk_engine.construct_synthetic_portfolio(raw_assets)
            hub.log(f"Synthetic Portfolio constructed. Rows: {len(main_df)}")
            
        else:
            hub.log(f"--- SINGLE ASSET MODE ({settings.symbol}) ---")
            main_df = data_lake.load_or_fetch(settings.symbol)
            if main_df.empty:
                hub.log("Critical: No data fetched.", "error")
                return
        
        # 3. Regime Detection (HMM)
        hub.log("Detecting Volatility Regimes (Markov Switching)...")
        regime_probs = regime_model.detect_regimes(main_df['returns'])
        
        # 4. Risk Calculation (Adaptive VaR)
        hub.log(
            f"Calculating VaR (Confidence: {settings.var_confidence*100}%, "
            f"Capital: ${settings.capital:,.0f})..."
        )
        
        backtest_results = {}
        dist_data = {}
        
        for h in settings.horizons:
            res_df = risk_engine.calc_adaptive_var(main_df, regime_probs, horizon=h)
            perf = risk_engine.get_performance_summary(res_df)
            backtest_results[h] = res_df
            
            raw_ret = risk_engine.get_raw_horizon_returns(main_df, horizon=h)
            dist_data[h] = raw_ret
            
            bo_pct = perf['breakout_pct']
            target = perf['target_pct']
            
            status = "ACCEPTABLE"
            if bo_pct > 1.5 * target:
                status = "DANGEROUS"
            elif bo_pct < 0.5 * target:
                status = "TOO SAFE"
            elif 0.8 * target <= bo_pct <= 1.2 * target:
                status = "EXCELLENT"

            hub.log(
                f"Horizon {h}D | Breakouts: {bo_pct:5.2f}% (Target {target:.1f}%) | "
                f"Avg VaR: ${abs(perf['avg_var_usd']):,.0f} | "
                f"CVaR: ${abs(perf['cvar_realized_usd']):,.0f} | "
                f"Drawdown: {perf['avg_dd_pct']:.2f}% | {status}"
            )

        hub.log("=" * 80)

        # 5. Correlation Analysis (Portfolio Only)
        corr_matrix = None
        if is_portfolio and components_df is not None:
            hub.log(f"Calculating Rolling Correlation ({settings.correlation_window} days)...")
            corr_matrix = risk_engine.get_rolling_correlation(components_df)
            hub.log("Correlation Matrix Ready.")
        
        # Print Stats Table via Hub (Console Report)
        primary_horizon = settings.horizons[0]
        primary_results = backtest_results[primary_horizon]
        hub.print_stats_table(primary_results)

        # 6. Visualization & Reporting
        hub.log("Generating graphical report...")
        visualizer.plot_dashboard(
            results_df=primary_results,
            regime_probs=regime_probs,
            dist_data_dict=dist_data,
            corr_matrix=corr_matrix,
            title=f"Adaptive VaR Model | {'Portfolio' if is_portfolio else settings.symbol}"
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
        hub.log(f"UNEXPECTED Pipeline Crash: {type(e).__name__}: {e}", "critical")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
