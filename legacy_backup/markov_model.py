"""
Markov Regime Switching Model for Volatility Clustering Detection.

This module implements a Hidden Markov Model (HMM) to detect market regimes
(Calm vs Stress) based on absolute returns as a volatility proxy.

CRITICAL: Uses filtered_marginal_probabilities (real-time, no look-ahead)
instead of smoothed_marginal_probabilities (full-sample, look-ahead bias).
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitchingResultsWrapper
from config import Config

logger = logging.getLogger(__name__)

class RegimeModel:
    """
    A 2-state Markov Regime Switching model for detecting volatility regimes.
    
    Uses Rolling Window Estimation to prevent Look-Ahead Bias:
    - Fits model on [t-WINDOW : t]
    - Updates parameters sparsely (every REFIT_INTERVAL days)
    - Filters probabilities for [t : t+interval] using fixed parameters
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.last_fit_result: Optional[MarkovSwitchingResultsWrapper] = None
        self.last_params = None

    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes using Rolling Window & Sparse Re-estimation.
        
        Args:
            returns: Log-returns series
            
        Returns:
            pd.Series of Stress probabilities (0..1), aligned with input index.
        """
        # 1. Prepare Data
        y = np.abs(returns * 100) + 1e-6
        if y.index.freq is None:
            y = y.asfreq('D').ffill()
            
        # Result container
        probs = pd.Series(0.0, index=y.index)
        # Mark warmup period as 0 or NaN? 0 is safer for pipeline.
        
        start_idx = self.cfg.MARKOV_MIN_OBSERVATIONS
        window_size = self.cfg.MARKOV_WINDOW_SIZE
        refit_interval = self.cfg.MARKOV_REFIT_INTERVAL
        
        total_obs = len(y)
        if total_obs < start_idx:
            logger.warning(f"[Markov] Not enough data ({total_obs} < {start_idx}). Returning zeros.")
            return probs

        logger.info(f"[Markov] Starting Rolling Estimation (Window={window_size}, Interval={refit_interval})...")
        
        # 2. Rolling Loop
        # We iterate in chunks of `refit_interval`
        for t in range(start_idx, total_obs, refit_interval):
            
            # Define Training Window: [t - window : t]
            train_start = max(0, t - window_size)
            train_data = y.iloc[train_start:t]
            
            # Define Prediction/Filter Window: [t : t + interval]
            # We will generate probabilities for these days using the model fitted on train_data
            pred_end = min(t + refit_interval, total_obs)
            pred_data = y.iloc[t:pred_end]
            
            if len(train_data) < self.cfg.MARKOV_MIN_OBSERVATIONS:
                continue

            # 3. Fit or Update Model
            try:
                # k_regimes=2, trend='c' (intercept), switching_variance=True
                model = MarkovRegression(train_data, k_regimes=2, trend='c', switching_variance=True)
                
                # Fit the model
                # Use last_params as starting point to speed up convergence
                res = model.fit(start_params=self.last_params, disp=False, maxiter=200)
                
                # Store parameters for next iteration
                self.last_params = res.params
                self.last_fit_result = res
                
                # Identify Stress Regime (Higher Volatility / Intercept)
                # Note: We re-check this every time because label 0/1 might flip
                params_const = res.params[['const[0]', 'const[1]']].values
                stress_idx = int(np.argmax(params_const))
                
                # 4. Filter Probabilities for Prediction Window
                # We need to apply the fitted parameters to fresh data (pred_data)
                # Construct a new model instance for the extended period or just filter?
                # Statsmodels creates a new model object for the new dataset but fixes params.
                 
                # Optimization: To get filtered probs for [t : t+interval], 
                # we conceptually append pred_data to history.
                # However, full filter run is O(N). We want O(interval).
                # simpler approach: Create model for `pred_data` but use `res.params`? 
                # No, Markov dependance requires state from `t-1`.
                
                # Correction: "Sparse Re-estimation" usually means:
                # 1. Fit params on [0..t] (or rolling [t-W..t])
                # 2. Use these params to filter [t..t+interval]
                # To do (2) correctly in statsmodels, we usually have to run the filter on the sequence.
                
                # Practical Approach for this scale:
                # Create a model instance for (train_data + pred_data). 
                # Fix params to `res.params`. 
                # Extract filtered_marginal_probabilities for the last `len(pred_data)` steps.
                
                combined_data = y.iloc[train_start:pred_end]
                apply_model = MarkovRegression(combined_data, k_regimes=2, trend='c', switching_variance=True)
                
                # Apply parameters (no fitting, just filtering)
                # smooth_results=False -> filtered only
                apply_res = apply_model.smooth(res.params) 
                
                # Get probabilities for the prediction chunk
                chunk_probs = apply_res.filtered_marginal_probabilities[stress_idx]
                
                # Slice out only the new predictions (last `len(pred_data)` points)
                # Beware of indexing alignment
                probs.iloc[t:pred_end] = chunk_probs.iloc[-len(pred_data):].values
                
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                # KNOWN FAILURE MODES:
                # - ValueError: Insufficient variance in training window
                # - LinAlgError: Singular covariance matrix (pathological data)
                # - RuntimeError: Optimization failed to converge
                # FALLBACK: Keep probabilities at 0.0 (Calm regime default)
                # This is conservative - model assumes calm until proven otherwise.
                logger.warning(f"[Markov] Fit/Filter failed at t={t} ({type(e).__name__}): {e}. "
                               f"Falling back to Calm regime (P=0) for interval [{t}:{t + refit_interval}].")
                # probs[t:pred_end] remains at initialized 0.0 value
                
            except Exception as e:
                # Unexpected errors should be visible for debugging
                logger.error(f"[Markov] UNEXPECTED error at t={t}: {type(e).__name__}: {e}. "
                             f"Falling back to Calm regime.", exc_info=True)

        # 5. Smoothing (Optional cleanup)
        if self.cfg.MARKOV_SMOOTH_WINDOW > 1:
            probs = probs.rolling(window=self.cfg.MARKOV_SMOOTH_WINDOW).mean().fillna(0)
            
        return probs