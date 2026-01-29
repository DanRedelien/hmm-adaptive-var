"""
Analytics and Logging Hub for the Adaptive VaR Risk Model.

This module is responsible for:
1. Structured Logging (Console output).
2. Statistical validation tests (Kupiec, Christoffersen).
3. Performance metrics calculation.

PLOTTING IS NOW MOVED TO visualizer.py.
"""

import logging
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import colorlog

from hmm_var.settings import Settings


class AnalyticsHub:
    """
    Central hub for logging and statistical validation.
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize AnalyticsHub."""
        self.settings = settings
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup colored console output logger."""
        logger = logging.getLogger("RiskManager")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = colorlog.StreamHandler(sys.stdout)
            fmt = '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s'
            handler.setFormatter(colorlog.ColoredFormatter(
                fmt, datefmt='%H:%M:%S',
                log_colors={
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'
                }
            ))
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    def log(self, msg: str, level: str = 'info') -> None:
        """Logging wrapper for unified output."""
        if hasattr(self.logger, level): 
            getattr(self.logger, level)(msg)
        else: 
            self.logger.info(msg)

    def print_stats_table(self, df: pd.DataFrame) -> None:
        """Prints a clean metrics table to the console."""
        
        if df.empty:
            return

        total_ret = (np.exp(df['realized_return'].sum()) - 1) * 100
        vol_ann = df['realized_return'].std() * np.sqrt(252) * 100
        sharpe = (total_ret / vol_ann) if vol_ann > 0 else 0
        
        valid_df = df.dropna(subset=['var_forecast', 'realized_return'])
        mask_var = valid_df['realized_return'] < valid_df['var_forecast']
        var_breaks = valid_df[mask_var]
        
        # Calculate Validation Metrics
        total_obs = len(valid_df)
        breaches_count = len(var_breaks)
        var_breach_pct = (breaches_count / total_obs) * 100 if total_obs > 0 else 0
        
        # Kupiec Test (Unconditional Coverage)
        kupiec_p = self._kupiec_test(total_obs, breaches_count, self.settings.var_confidence)
        kupiec_res = "PASS" if kupiec_p > 0.05 else "FAIL"
        
        # Christoffersen Test (Conditional Coverage / Clustering)
        breach_series = pd.Series(0, index=valid_df.index)
        breach_series.loc[var_breaks.index] = 1
        christ_p = self._christoffersen_test(breach_series)
        christ_res = "PASS" if christ_p > 0.05 else "FAIL"

        print("\n" + "="*65)
        print(f" RISK REPORT (CONF: {self.settings.var_confidence*100:.0f}%) | {total_obs} Observations")
        print("="*65)
        print(f"{'METRIC':<35} | {'VALUE':<15} | {'STATUS'}")
        print("-" * 65)
        print(f"{'Total Return':<35} | {total_ret:>6.2f} %        |")
        print(f"{'Annualized Volatility':<35} | {vol_ann:>6.2f} %        |")
        print("-" * 65)
        print(f"{'VaR Expected Breach Rate':<35} | {(1-self.settings.var_confidence)*100:>6.2f} %        |")
        print(f"{'VaR Actual Breach Rate':<35} | {var_breach_pct:>6.2f} %        |")
        print(f"{'Kupiec POF Test (p-val)':<35} | {kupiec_p:>6.4f}          | {kupiec_res}")
        print(f"{'Christoffersen Ind. Test (p-val)':<35} | {christ_p:>6.4f}          | {christ_res}")
        print("="*65 + "\n")

    def _kupiec_test(self, total: int, breaches: int, confidence: float) -> float:
        """Kupiec POF Test (Likelihood Ratio)."""
        if total == 0: return 0.0
        
        p_exp = 1.0 - confidence
        p_obs = breaches / total
        
        if breaches == 0:
            lr = -2 * np.log( (1 - p_exp)**total )
        elif breaches == total:
            lr = -2 * np.log( p_exp**total )
        else:
            null_log_lik = (total - breaches) * np.log(1 - p_exp) + breaches * np.log(p_exp)
            alt_log_lik = (total - breaches) * np.log(1 - p_obs) + breaches * np.log(p_obs)
            lr = -2 * (null_log_lik - alt_log_lik)
            
        return stats.chi2.sf(lr, df=1)

    def _christoffersen_test(self, breach_series: pd.Series) -> float:
        """Christoffersen Independence Test."""
        if len(breach_series) < 2: return 0.0
        
        hits = breach_series.values.astype(int)
        prev = hits[:-1]
        curr = hits[1:]
        
        n00 = ((prev == 0) & (curr == 0)).sum()
        n01 = ((prev == 0) & (curr == 1)).sum()
        n10 = ((prev == 1) & (curr == 0)).sum()
        n11 = ((prev == 1) & (curr == 1)).sum()
        
        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
        
        if pi_hat == 0 or pi_hat == 1:
            return 1.0 # Pass
            
        def log_lik(p, n_good, n_bad):
            if p == 0: return 0 if n_bad == 0 else -1e9
            if p == 1: return 0 if n_good == 0 else -1e9
            return n_good * np.log(1 - p) + n_bad * np.log(p)

        l_null = log_lik(pi_hat, n00 + n10, n01 + n11)
        l_alt  = log_lik(pi_0, n00, n01) + log_lik(pi_1, n10, n11)
        
        lr = -2 * (l_null - l_alt)
        lr = max(0.0, lr)
        
        return stats.chi2.sf(lr, df=1)
