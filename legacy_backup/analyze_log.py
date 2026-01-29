"""
Analytics and Visualization Hub for the Adaptive VaR Risk Model.

This module is responsible ONLY for data presentation (charts and console output).
Design: Clean, light theme, informative.
"""

import logging
import colorlog
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional
from config import Config


class AnalyticsHub:
    """
    Central hub for logging and visualization.
    Responsible ONLY for data presentation (charts and console output).
    
    Design: Clean, light theme, informative.
    """
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = self._setup_logger()
        
        # === THEME (Colors and Styles) ===
        self.c = {
            # Interface colors
            'bg_fig': '#ffffff',       # White figure background
            'bg_ax':  '#ffffff',       # White axes background
            'text':   '#2c3e50',       # Dark blue-gray (main text)
            'grid':   '#ecf0f1',       # Very light gray (grid)
            'spine':  '#bdc3c7',       # Border color
            
            # Data colors
            'price':    '#2c3e50',     # Dark blue (Portfolio Price)
            'stress':   '#e74c3c',     # Red (Stress Regime)
            'calm':     '#2ecc71',     # Green (Calm Regime)
            
            # Risk metrics - Updated for probability-weighted VaR
            'var_blended': '#e67e22',  # Orange (Blended VaR - main line)
            'var_calm':    '#27ae60',  # Green (VaR Calm window)
            'var_stress':  '#c0392b',  # Dark red (VaR Stress window)
            'cvar_line':   '#7f8c8d',  # Gray dashed (Expected Shortfall)
            'realized':    '#95a5a6',  # Light gray (Daily Returns)
            
            'breakout_var':  '#c0392b', # Dark red (VaR Breakout)
            'breakout_cvar': '#8e44ad', # Purple (ES Breakout)
            
            # Distribution colors
            'dist_colors': ['#2980b9', '#8e44ad', '#27ae60'] 
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup colored console output logger."""
        logger = logging.getLogger("RiskManager")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = colorlog.StreamHandler(sys.stdout)
            fmt = '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s'
            handler.setFormatter(colorlog.ColoredFormatter(fmt, datefmt='%H:%M:%S',
                log_colors={'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}))
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    def log(self, msg: str, level: str = 'info') -> None:
        """Logging wrapper for unified output."""
        if hasattr(self.logger, level): 
            getattr(self.logger, level)(msg)
        else: 
            self.logger.info(msg)

    def plot_dashboard(self, 
                       results_df: pd.DataFrame, 
                       regime_probs: pd.Series, 
                       dist_data_dict: Dict[int, pd.Series],
                       corr_matrix: Optional[pd.DataFrame] = None,
                       title: str = "Risk Dashboard") -> None:
        """
        Main dashboard rendering method.
        Accepts only pre-calculated data. No computations inside.
        
        Updated for Probability-Weighted VaR visualization:
        - Shows VaR_calm, VaR_stress, and Blended VaR
        - Uses continuous color gradient for regime probability
        - No more threshold-based binary switching visualization
        
        Args:
            results_df: Backtest results with 'close', 'realized_return', 'var_forecast', 
                        'cvar_forecast', 'var_calm', 'var_stress', 'regime_prob'.
            regime_probs: Series of Stress probabilities (0..1).
            dist_data_dict: Dictionary {horizon: returns_series} for distribution plots.
            corr_matrix: Optional correlation matrix for heatmap.
            title: Dashboard title.
        """
        self.log("Generating graphical report (Probability-Weighted VaR)...", "info")

        # Global matplotlib style settings
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.facecolor'] = self.c['bg_ax']
        plt.rcParams['figure.facecolor'] = self.c['bg_fig']
        plt.rcParams['text.color'] = self.c['text']
        plt.rcParams['axes.labelcolor'] = self.c['text']
        plt.rcParams['xtick.color'] = self.c['text']
        plt.rcParams['ytick.color'] = self.c['text']
        
        # === LAYOUT: GridSpec 3x2 ===
        # Row 1: Price Chart (full width)
        # Row 2: Risk Chart (VaR/ES) | Regime Probability Chart
        # Row 3: Distributions | Correlation
        
        fig = plt.figure(figsize=(16, 12), layout='constrained')
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        ax1 = fig.add_subplot(gs[0, :])     # 1. Price + Regimes
        ax2 = fig.add_subplot(gs[1, 0])     # 2. Adaptive VaR (Calm/Stress/Blended)
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2) # 3. Regime Probability
        ax4 = fig.add_subplot(gs[2, 0])     # 4. Distributions
        ax5 = fig.add_subplot(gs[2, 1])     # 5. Correlation

        # Common axis styling (Grid, Spines)
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.grid(True, color=self.c['grid'], linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(self.c['spine'])
            ax.spines['left'].set_color(self.c['spine'])
            ax.tick_params(colors=self.c['text'], which='both')

        # === 1. PORTFOLIO VALUE + HMM REGIMES (Top) ===
        price_series = results_df['close']
        ax1.plot(price_series.index, price_series, color=self.c['price'], lw=2, label='Portfolio Value')
        
        # Continuous gradient background based on P(Stress)
        # Instead of binary Red/Green, use intensity mapping
        valid_idx = price_series.index.intersection(regime_probs.index)
        if not valid_idx.empty:
            p_probs = regime_probs.loc[valid_idx]
            y_min, y_max = price_series.min() * 0.95, price_series.max() * 1.05
            ax1.set_ylim(y_min, y_max)

            # Create continuous color gradient based on P(Stress)
            # Higher P(Stress) = more red, Lower = more green
            for i in range(len(valid_idx) - 1):
                prob = p_probs.iloc[i]
                # Blend between green (calm) and red (stress)
                color = self._blend_color(self.c['calm'], self.c['stress'], prob)
                ax1.axvspan(valid_idx[i], valid_idx[i+1], 
                           facecolor=color, alpha=0.15, linewidth=0)

        ax1.set_title("1. Portfolio Value & Regime Intensity", fontweight='bold', loc='left', fontsize=12)
        ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # === 2. PROBABILITY-WEIGHTED VaR MODEL (Middle Left) ===
        # Return bars
        ax2.bar(results_df.index, results_df['realized_return'] * 100, 
                color=self.c['realized'], alpha=0.5, label='Daily Returns %', width=1.0)
        
        # VaR Blended (Orange, thick solid) - Final probability-weighted
        if 'var_forecast' in results_df.columns:
            var_pct = results_df['var_forecast'] * 100
            ax2.plot(results_df.index, var_pct, 
                    color=self.c['var_blended'], lw=2, 
                    label='VaR Blended (P-Weighted)')
            
            # VaR Breakouts (Red dots)
            mask_var = (results_df['realized_return'] < results_df['var_forecast'])
            breaks_var = results_df[mask_var]
            if not breaks_var.empty:
                ax2.scatter(breaks_var.index, breaks_var['realized_return']*100, 
                            color=self.c['breakout_var'], s=20, zorder=5, label='VaR Breakout')

        ax2.set_title("2. Daily PnL vs VaR Limits", fontweight='bold', loc='left', fontsize=12)
        ax2.set_ylabel("Returns / Risk (%)")
        ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=7, ncol=2)

        # === 3. VAR DYNAMICS: Calm vs Stress vs Blended (Middle Right) ===
        # Show how VaR limits EVOLVE over time - the key insight of probability-weighting
        
        # VaR in USD (absolute risk) for better interpretability
        if 'var_calm' in results_df.columns and 'var_stress' in results_df.columns:
            # Convert log returns to USD
            var_calm_usd = self.cfg.CAPITAL * (np.exp(results_df['var_calm']) - 1).abs()
            var_stress_usd = self.cfg.CAPITAL * (np.exp(results_df['var_stress']) - 1).abs()
            var_blended_usd = self.cfg.CAPITAL * (np.exp(results_df['var_forecast']) - 1).abs()
            
            # 3A. Secondary Axis for Regime Intensity (Background)
            ax3b = ax3.twinx()
            ax3b.fill_between(regime_probs.index, 0, regime_probs, 
                              color=self.c['stress'], alpha=0.1, label='Regime Intensity')
            ax3b.set_ylim(0, 1.0)
            ax3b.set_yticks([]) # Hide ticks to keep it clean, or keep if requested
            # Move main ax to front so lines are visible over the fill
            ax3.set_zorder(ax3b.get_zorder() + 1)
            ax3.patch.set_visible(False)

            # 3B. VaR Lines (Main Axis)
            # VaR Calm (Green, thin, transparent)
            ax3.plot(results_df.index, var_calm_usd / 1000, 
                    color=self.c['var_calm'], lw=1, linestyle='-', 
                    alpha=0.3, label=f'VaR Calm')
            
            # VaR Stress (Red, thin, transparent)
            ax3.plot(results_df.index, var_stress_usd / 1000, 
                    color=self.c['var_stress'], lw=1, linestyle='-', 
                    alpha=0.3, label=f'VaR Stress')
            
            # Blended VaR (Orange, thick) - Final P-weighted result
            ax3.plot(results_df.index, var_blended_usd / 1000, 
                    color=self.c['var_blended'], lw=2.5, 
                    label='VaR Blended (Final)')
        else:
            # Fallback to P(Stress) if VaR columns missing
            ax3.plot(regime_probs.index, regime_probs, color=self.c['text'], lw=1.5)
            ax3.fill_between(regime_probs.index, 0, regime_probs, 
                            color=self.c['stress'], alpha=0.3, label='P(Stress)')
        
        ax3.set_title("3. VaR Limit Dynamics ($K) - Calm vs Stress", fontweight='bold', loc='left', fontsize=12)
        ax3.set_ylabel("VaR ($ thousands)")
        ax3.legend(loc='upper left', fontsize=7)

        # Date formatting for middle row
        for ax in [ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # === 4. RETURN DISTRIBUTIONS (Bottom Left) ===
        for i, horizon in enumerate(self.cfg.HORIZONS):
            raw_ret = dist_data_dict.get(horizon)
            if raw_ret is None or raw_ret.empty: 
                continue
            
            series_pct = raw_ret * 100
            color = self.c['dist_colors'][i % len(self.c['dist_colors'])]
            
            try:
                sns.kdeplot(series_pct, ax=ax4, color=color, fill=True, alpha=0.1, lw=2, label=f'{horizon}D Returns')
            except Exception:
                pass 
            
            # Static VaR line for comparison
            static_var = np.percentile(series_pct, (1 - self.cfg.VAR_CONFIDENCE)*100)
            ax4.axvline(static_var, color=color, linestyle='--', alpha=0.5)

        ax4.set_title("4. Return Distributions (Fat Tails)", fontweight='bold', loc='left', fontsize=12)
        ax4.set_xlabel("Return (%)")
        ax4.legend(loc='upper left', fontsize=8)

        # === 5. ROLLING CORRELATION (Bottom Right) ===
        # === 5. PAIRWISE CORRELATION STRIP PLOT (Bottom Right) ===
        if corr_matrix is not None and not corr_matrix.empty:
            # 1. Sort assets by Mean Correlation (most systematic first)
            # This makes the graph look organized (e.g. highly correlated cluster vs decoupled assets)
            mean_corr = corr_matrix.mean().sort_values(ascending=False)
            sorted_assets = mean_corr.index.tolist()
            
            # 2. Prepare Scatter Data (Flattened)
            all_xs = []
            all_ys = []
            all_labels = []
            
            for i, asset in enumerate(sorted_assets):
                # Get correlations for this asset
                series = corr_matrix[asset]
                # Filter out self-correlation (1.0)
                series = series[series.index != asset]
                
                for other_asset, corr_val in series.items():
                    all_xs.append(i)
                    all_ys.append(corr_val)
                    all_labels.append(f"{asset} vs {other_asset}")

            # 3. Plotting
            if all_xs:
                # Separate data into "Normal" and "Highlight" to ensure z-order (Highlights on TOP)
                norm_xs, norm_ys = [], []
                high_xs, high_ys = [], []
                
                # Keep track of indices for original labels if needed, or just rebuild labels for hover?
                # The hover function uses `sc.contains(event)`. If we have two scatters, we need to handle both 
                # or merge them visually but keep data accessible.
                # EASIER WAY: Sort the data so highlights are LAST in the list.
                # This ensures they are drawn last within the single scatter call.
                
                combined_data = []
                highlight_asset = self.cfg.HIGHLIGHT_ASSET
                high_color = '#e67e22'
                default_color = self.c['price']

                for i in range(len(all_xs)):
                    label = all_labels[i]
                    x = all_xs[i]
                    y = all_ys[i]
                    
                    # Logic: Highlight if 'other_asset' is BTC (meaning this dot represents correlation WITH BTC)
                    # But if the Column (asset) IS BTC, we don't highlight the dots inside it (as per user request "return cell to color it was")
                    
                    is_highlight = False
                    parts = label.split(" vs ")
                    if len(parts) == 2:
                        asset_col, other_col = parts[0], parts[1]
                        if other_col == highlight_asset:
                            is_highlight = True
                    
                    # Store tuple: (is_highlight, x, y, color, label)
                    if is_highlight:
                        combined_data.append((1, x, y, high_color, label))
                    else:
                        combined_data.append((0, x, y, default_color, label))
                
                # Sort: Non-highlights (0) first, Highlights (1) last.
                # This guarantees highlights are drawn ON TOP of others.
                combined_data.sort(key=lambda x: x[0])
                
                # Unzip
                sorted_xs = [d[1] for d in combined_data]
                sorted_ys = [d[2] for d in combined_data]
                sorted_colors = [d[3] for d in combined_data]
                sorted_labels = [d[4] for d in combined_data] # We need this for hover

                # Update global all_labels for the hover function to work correctly with the new order
                all_labels = sorted_labels 
                all_ys = sorted_ys # Update for hover referencing

                # Plot
                sc = ax5.scatter(sorted_xs, sorted_ys, c=sorted_colors, s=40, edgecolors='none', alpha=0.85, zorder=5)
                
                # Reference Lines
                ax5.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
                ax5.axhline(1.0, color=self.c['grid'], linestyle='-', alpha=0.3, zorder=1)
                for y in [-0.5, 0.5]:
                    ax5.axhline(y, color=self.c['grid'], linestyle=':', alpha=0.5, zorder=1)

                # Axes Setup
                ax5.set_ylim(-1.1, 1.1)
                
                # X-Axis: Show Asset Names, but maybe sparsely if too many? 
                # Converting index to labels
                ax5.set_xticks(range(len(sorted_assets)))
                if len(sorted_assets) > 10:
                    # Rotate if many assets
                    ax5.set_xticklabels(sorted_assets, rotation=90, fontsize=8)
                else:
                    ax5.set_xticklabels(sorted_assets, rotation=45, fontsize=9)
                
                ax5.set_ylabel("Correlation Coefficient", fontsize=10)
                ax5.set_title(f"5. Pairwise Correlation Distribution (Sorted)\n(Hover for Pair Details)", fontweight='bold', loc='center', fontsize=11)
                
                # --- Interactive Hover Annotation ---
                quant_annot = ax5.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w", alpha=0.95),
                                    arrowprops=dict(arrowstyle="->"),
                                    zorder=100) # this thing
                quant_annot.set_visible(False)
                
                def hover_func(event):
                    if event.inaxes == ax5:
                        cont, ind = sc.contains(event)
                        if cont:
                            # Use the first point found (simplification for dense plots)
                            idx = ind["ind"][0]
                            pos = sc.get_offsets()[idx]
                            quant_annot.xy = pos
                            
                            # Get data from lists
                            lbl = all_labels[idx]
                            val = all_ys[idx]
                            
                            text = f"{lbl}\nCorr: {val:.2f}"
                            quant_annot.set_text(text)
                            quant_annot.set_visible(True)
                            fig.canvas.draw_idle()
                        else:
                            if quant_annot.get_visible():
                                quant_annot.set_visible(False)
                                fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", hover_func)
                
                ax5.set_aspect('auto')
                ax5.set_anchor('C')
            else:
                 ax5.text(0.5, 0.5, "No Pairwise Data", ha='center', va='center')
        else:
            ax5.text(0.5, 0.5, "Single Asset Mode\n(No Correlation Data)", 
                     ha='center', va='center', fontsize=12, color=self.c['text'])
            ax5.set_title("5. Correlation", fontweight='bold', loc='left')

        plt.show()
        
        # === 6. PRINT STATS TABLE TO TERMINAL ===
        self._print_stats_table(results_df)

    def _blend_color(self, color1: str, color2: str, weight: float) -> str:
        """
        Blends two hex colors based on weight (0..1).
        weight=0 → color1, weight=1 → color2
        """
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        
        blended = tuple(int(c1 * (1 - weight) + c2 * weight) for c1, c2 in zip(rgb1, rgb2))
        return rgb_to_hex(blended)

    def _print_stats_table(self, df: pd.DataFrame) -> None:
        """Prints a clean metrics table to the console."""
        
        if df.empty:
            return

        # 1. Calculate metrics
        total_ret = (np.exp(df['realized_return'].sum()) - 1) * 100
        vol_ann = df['realized_return'].std() * np.sqrt(252) * 100
        
        # Sharpe (simplified, risk-free=0)
        sharpe = (total_ret / vol_ann) if vol_ann > 0 else 0
        
        # VaR Stats - CRITICAL: Filter to valid VaR rows only (exclude warm-up period)
        valid_df = df.dropna(subset=['var_forecast', 'realized_return'])
        mask_var = valid_df['realized_return'] < valid_df['var_forecast']
        var_breaks = valid_df[mask_var]
        var_breach_pct = (len(var_breaks) / len(valid_df)) * 100 if len(valid_df) > 0 else 0
        
        # ES Stats (if available)
        cvar_col = 'cvar_forecast' if 'cvar_forecast' in df.columns else None
        es_text = "N/A"
        es_breaks_count = 0
        
        if cvar_col:
            mask_cvar = valid_df['realized_return'] < valid_df[cvar_col]
            es_breaks = valid_df[mask_cvar]
            es_breaks_count = len(es_breaks)
            es_breach_pct = (es_breaks_count / len(valid_df)) * 100 if len(valid_df) > 0 else 0
            es_text = f"{es_breach_pct:.2f} %"

        # 2. Print table
        print("\n" + "="*55)
        print(f" RISK REPORT (CONF: {self.cfg.VAR_CONFIDENCE*100:.0f}%)")
        print("="*55)
        print(f"{'METRIC':<30} | {'VALUE':<15}")
        print("-" * 48)
        print(f"{'Total Return':<30} | {total_ret:>.2f} %")
        print(f"{'Annualized Volatility':<30} | {vol_ann:>.2f} %")
        print(f"{'Sharpe Ratio (Approx)':<30} | {sharpe:>.2f}")
        print("-" * 48)
        print(f"{'VaR Breaches (Count)':<30} | {len(var_breaks)}")
        print(f"{'VaR Breach Rate':<30} | {var_breach_pct:>.2f} %")
        
        if cvar_col:
            print(f"{'ES/CVaR Breaches (Count)':<30} | {es_breaks_count}")
            print(f"{'ES Breach Rate':<30} | {es_text}")
            
        print("="*55 + "\n")