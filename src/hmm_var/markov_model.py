"""
Markov Regime Switching Model for Volatility Clustering Detection.

This module implements a Hidden Markov Model (HMM) to detect market regimes
(Calm vs Stress) based on absolute returns as a volatility proxy.

CRITICAL: Uses filtered_marginal_probabilities (real-time, no look-ahead)
instead of smoothed_marginal_probabilities (full-sample, look-ahead bias).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitchingResultsWrapper
from statsmodels.tools.sm_exceptions import EstimationWarning
import warnings

# Suppress benign HMM warnings during rolling fit
warnings.simplefilter('ignore', EstimationWarning)

from hmm_var.settings import Settings

logger = logging.getLogger(__name__)


class RegimeModel:
    """
    A 2-state Markov Regime Switching model for detecting volatility regimes.
    
    Uses Rolling Window Estimation to prevent Look-Ahead Bias:
    - Fits model on [t-WINDOW : t]
    - Updates parameters sparsely (every REFIT_INTERVAL days)
    - Filters probabilities for [t : t+interval] using fixed parameters
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize RegimeModel.
        
        Args:
            settings: Configuration object containing Markov parameters.
        """
        self.settings = settings
        self.last_fit_result: Optional[MarkovSwitchingResultsWrapper] = None
        self.last_params = None

    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes using Rolling Window & Sparse Re-estimation.
        
        Args:
            returns: Log-returns series.
            
        Returns:
            pd.Series of Stress probabilities (0..1), aligned with input index.
        """
        # 1. Prepare Data - Absolute returns as volatility proxy
        abs_returns = np.abs(returns * 100) + 1e-6
        
        # Normalize returns for HMM stability (z-score standardization)
        abs_mean = abs_returns.mean()
        abs_std = abs_returns.std()
        if abs_std > 0:
            y = (abs_returns - abs_mean) / abs_std
        else:
            y = abs_returns - abs_mean
        
        if y.index.freq is None:
            y = y.asfreq('D').ffill()
            
        # Result container
        probs = pd.Series(0.0, index=y.index)
        
        start_idx = self.settings.markov_min_observations
        window_size = self.settings.markov_window_size
        refit_interval = self.settings.markov_refit_interval
        
        total_obs = len(y)
        if total_obs < start_idx:
            logger.warning(
                f"[Markov] Not enough data ({total_obs} < {start_idx}). Returning zeros."
            )
            return probs

        logger.info(
            f"[Markov] Starting Rolling Estimation "
            f"(Window={window_size}, Interval={refit_interval})..."
        )
        
        # 2. Rolling Loop
        for t in range(start_idx, total_obs, refit_interval):
            
            # Define Training Window: [t - window : t]
            train_start = max(0, t - window_size)
            train_data = y.iloc[train_start:t]
            
            # Define Prediction/Filter Window: [t : t + interval]
            pred_end = min(t + refit_interval, total_obs)
            pred_data = y.iloc[t:pred_end]
            
            if len(train_data) < self.settings.markov_min_observations:
                continue

            # 3. Fit or Update Model
            try:
                model = MarkovRegression(
                    train_data, k_regimes=2, trend='c', switching_variance=True
                )
                
                # Fit with last params as starting point
                res = model.fit(start_params=self.last_params, disp=False, maxiter=200)
                
                # Store parameters
                self.last_params = res.params
                self.last_fit_result = res
                
                # Identify Stress Regime (Higher Volatility)
                params_const = res.params[['const[0]', 'const[1]']].values
                stress_idx = int(np.argmax(params_const))
                
                # 4. Filter Probabilities for Prediction Window
                combined_data = y.iloc[train_start:pred_end]
                apply_model = MarkovRegression(
                    combined_data, k_regimes=2, trend='c', switching_variance=True
                )
                
                apply_res = apply_model.smooth(res.params)
                
                chunk_probs = apply_res.filtered_marginal_probabilities[stress_idx]
                probs.iloc[t:pred_end] = chunk_probs.iloc[-len(pred_data):].values
                
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                # Fallback: Use previous probability if available, else 0 (Calm)
                # This prevents massive jumps in risk estimates due to single-day solver failures
                prev_prob = 0.0
                if t > start_idx:
                    # Get last calculated probability
                    prev_prob = probs.iloc[t-1]
                
                logger.warning(
                    f"[Markov] Fit/Filter failed at t={t} ({type(e).__name__}): {e}. "
                    f"Holding previous probability (P={prev_prob:.2f})."
                )
                
                # Fill prediction window with LAST KNOWN probability
                probs.iloc[t:pred_end] = prev_prob
                
            except Exception as e:
                logger.error(
                    f"[Markov] UNEXPECTED error at t={t}: {type(e).__name__}: {e}. "
                    f"Falling back to Calm regime.",
                    exc_info=True
                )


        # 5. Smoothing
        if self.settings.markov_smooth_window > 1:
            probs = probs.rolling(window=self.settings.markov_smooth_window).mean().fillna(0)
            
        return probs
