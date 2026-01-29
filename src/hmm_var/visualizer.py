"""
Visualization Module for the Adaptive VaR Risk Model.

Handles all graphical output, adhering to the VISUALIZATION_SPEC.md standards.
Separated from analytics logic for clean architecture.
"""
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hmm_var.settings import Settings


class Visualizer:
    """
    Dedicated Visualization Engine.
    
    Adheres to Institutional Styling:
    - Background: White
    - Profit/Calm: Green (#2ECC71)
    - Loss/Stress: Red (#E74C3C)
    - Fonts: Sans-Serif
    """
    
    # Institutional Color Palette (The "Jane Street" Theme)
    C = {
        'bg_fig': '#ffffff',
        'bg_ax':  '#ffffff',
        'text':   '#2c3e50',
        'grid':   '#ecf0f1',
        'spine':  '#bdc3c7',
        
        'price':    '#2c3e50',
        'stress':   '#e74c3c',
        'calm':     '#2ecc71',
        
        'var_blended': '#e67e22',
        'var_calm':    '#27ae60',
        'var_stress':  '#c0392b',
        'realized':    '#95a5a6',
        
        'breakout_var':  '#c0392b',
        'dist_colors': ['#2980b9', '#8e44ad', '#27ae60']
    }

    def __init__(self, settings: Settings) -> None:
        """Initialize Visualizer with project settings."""
        self.settings = settings
        self._apply_global_styles()

    def _apply_global_styles(self) -> None:
        """Apply matplotlib global configuration."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.facecolor'] = self.C['bg_ax']
        plt.rcParams['figure.facecolor'] = self.C['bg_fig']
        plt.rcParams['text.color'] = self.C['text']
        plt.rcParams['axes.labelcolor'] = self.C['text']
        plt.rcParams['xtick.color'] = self.C['text']
        plt.rcParams['ytick.color'] = self.C['text']

    def plot_dashboard(
        self,
        results_df: pd.DataFrame, 
        regime_probs: pd.Series, 
        dist_data_dict: Dict[int, pd.Series],
        corr_matrix: Optional[pd.DataFrame] = None,
        title: str = "Risk Dashboard"
    ) -> None:
        """
        Main dashboard rendering method.
        
        Layout: GridSpec 3x2
        1. Portfolio Value & Regime overlay
        2. Daily PnL vs VaR Breaches
        3. VaR Dynamics (Calm vs Stress levels)
        4. Return Distributions (Fat tails)
        5. Correlation Matrix (if portfolio)
        """
        # === LAYOUT ===
        fig = plt.figure(figsize=(16, 12), layout='constrained')
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        # Common axis styling
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            self._style_axis(ax)

        # === 1. PORTFOLIO VALUE + HMM REGIMES ===
        price_series = results_df['close']
        ax1.plot(
            price_series.index, price_series,
            color=self.C['price'], lw=1.5, label='Portfolio Value'
        )
        
        valid_idx = price_series.index.intersection(regime_probs.index)
        if not valid_idx.empty:
            p_probs = regime_probs.loc[valid_idx]
            y_min, y_max = price_series.min() * 0.95, price_series.max() * 1.05
            ax1.set_ylim(y_min, y_max)

            for i in range(len(valid_idx) - 1):
                prob = p_probs.iloc[i]
                color = self._blend_color(self.C['calm'], self.C['stress'], prob)
                ax1.axvspan(
                    valid_idx[i], valid_idx[i+1], 
                    facecolor=color, alpha=0.15, linewidth=0
                )

        ax1.set_title("1. Portfolio Value & Regime Intensity", fontweight='bold', loc='left')
        ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # === 2. PROBABILITY-WEIGHTED VaR MODEL ===
        ax2.bar(
            results_df.index, results_df['realized_return'] * 100, 
            color=self.C['realized'], alpha=0.5, label='Daily Returns %', width=1.0
        )
        
        if 'var_forecast' in results_df.columns:
            var_pct = results_df['var_forecast'] * 100
            ax2.plot(
                results_df.index, var_pct, 
                color=self.C['var_blended'], lw=2, 
                label='VaR Blended'
            )
            
            mask_var = (results_df['realized_return'] < results_df['var_forecast'])
            breaks_var = results_df[mask_var]
            if not breaks_var.empty:
                ax2.scatter(
                    breaks_var.index, breaks_var['realized_return']*100, 
                    color=self.C['breakout_var'], s=20, zorder=5, label='VaR Breakout'
                )

        ax2.set_title("2. Daily PnL vs VaR Limits", fontweight='bold', loc='left')
        ax2.set_ylabel("Returns / Risk (%)")
        ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

        # === 3. VAR DYNAMICS ===
        if 'var_calm' in results_df.columns and 'var_stress' in results_df.columns:
            var_calm_usd = self.settings.capital * (np.exp(results_df['var_calm']) - 1).abs()
            var_stress_usd = self.settings.capital * (np.exp(results_df['var_stress']) - 1).abs()
            var_blended_usd = self.settings.capital * (np.exp(results_df['var_forecast']) - 1).abs()
            
            # Hidden twin axis for regime shading reference
            ax3b = ax3.twinx()
            ax3b.fill_between(regime_probs.index, 0, regime_probs, color=self.C['stress'], alpha=0.0)
            ax3b.set_yticks([])
            ax3.set_zorder(ax3b.get_zorder() + 1)
            ax3.patch.set_visible(False)

            ax3.plot(results_df.index, var_calm_usd / 1000, 
                     color=self.C['var_calm'], lw=1, alpha=0.3, label='VaR Calm')
            ax3.plot(results_df.index, var_stress_usd / 1000, 
                     color=self.C['var_stress'], lw=1, alpha=0.3, label='VaR Stress')
            ax3.plot(results_df.index, var_blended_usd / 1000, 
                     color=self.C['var_blended'], lw=2.5, label='VaR Blended')
        
        ax3.set_title("3. VaR Limit Dynamics ($K)", fontweight='bold', loc='left')
        ax3.set_ylabel("VaR ($ thousands)")
        ax3.legend(loc='upper left', fontsize=8)

        for ax in [ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # === 4. RETURN DISTRIBUTIONS ===
        for i, horizon in enumerate(self.settings.horizons):
            raw_ret = dist_data_dict.get(horizon)
            if raw_ret is None or raw_ret.empty: continue
            
            series_pct = raw_ret * 100
            color = self.C['dist_colors'][i % len(self.C['dist_colors'])]
            
            try:
                sns.kdeplot(
                    series_pct, ax=ax4, color=color, fill=True, 
                    alpha=0.1, lw=2, label=f'{horizon}D Returns'
                )
            except Exception:
                pass 
            
            static_var = np.percentile(series_pct, (1 - self.settings.var_confidence)*100)
            ax4.axvline(static_var, color=color, linestyle='--', alpha=0.5)

        ax4.set_title("4. Return Fat Tails", fontweight='bold', loc='left')
        ax4.set_xlabel("Return (%)")
        ax4.legend(loc='upper left', fontsize=8)

        # === 5. CORRELATION PLOT ===
        self._plot_correlation_scatter(ax5, corr_matrix)

        plt.show()

    def _style_axis(self, ax: plt.Axes) -> None:
        """Apply strict grid and spine styling."""
        ax.grid(True, color=self.C['grid'], linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.C['spine'])
        ax.spines['left'].set_color(self.C['spine'])
        ax.tick_params(colors=self.C['text'], which='both')

    def _plot_correlation_scatter(self, ax: plt.Axes, corr_matrix: Optional[pd.DataFrame]) -> None:
        """Plots the sorted correlation scatter chart."""
        if corr_matrix is None or corr_matrix.empty:
            ax.text(0.5, 0.5, "Single Asset Mode\n(No Correlation Data)",
                    ha='center', va='center', fontsize=12, color=self.C['text'])
            ax.set_title("5. Correlation", fontweight='bold', loc='left')
            return

        mean_corr = corr_matrix.mean().sort_values(ascending=False)
        sorted_assets = mean_corr.index.tolist()
        
        all_xs, all_ys, all_labels = [], [], []
        
        for i, asset in enumerate(sorted_assets):
            series = corr_matrix[asset]
            series = series[series.index != asset]
            for other_asset, corr_val in series.items():
                all_xs.append(i)
                all_ys.append(corr_val)
                all_labels.append(f"{asset} vs {other_asset}")

        if not all_xs:
            return

        combined_data = []
        highlight_asset = self.settings.highlight_asset
        
        for i in range(len(all_xs)):
            label = all_labels[i]
            x, y = all_xs[i], all_ys[i]
            
            # Highlight only when OTHER asset is the highlight_asset
            # (not when the column itself is highlight_asset)
            is_high = False
            parts = label.split(" vs ")
            if len(parts) == 2:
                asset_col, other_col = parts[0], parts[1]
                if other_col == highlight_asset:
                    is_high = True
            
            color = '#e67e22' if is_high else self.C['price']
            combined_data.append((int(is_high), x, y, color, label))
        
        combined_data.sort(key=lambda x: x[0])
        
        sorted_xs = [d[1] for d in combined_data]
        sorted_ys = [d[2] for d in combined_data]
        sorted_colors = [d[3] for d in combined_data]

        ax.scatter(sorted_xs, sorted_ys, c=sorted_colors, s=40, edgecolors='none', alpha=0.85, zorder=5)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks(range(len(sorted_assets)))
        ax.set_xticklabels(sorted_assets, rotation=90, fontsize=8)
        ax.set_title("5. Pairwise Correlations", fontweight='bold', loc='left')

    def _blend_color(self, color1: str, color2: str, weight: float) -> str:
        """Blends two hex colors based on weight (0..1)."""
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        blended = tuple(int(c1 * (1 - weight) + c2 * weight) for c1, c2 in zip(rgb1, rgb2))
        return rgb_to_hex(blended)
