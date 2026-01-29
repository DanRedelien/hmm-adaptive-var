# HMM-Driven Adaptive VaR Model

A regime-switching Value at Risk (VaR) engine that blends Weighted Historical Simulation (WHS) with Hidden Markov Model (HMM) state probabilities. Designed to reduce lag in risk estimation during volatility regime transitions while maintaining stability during calm periods.

---

## Problem Statement

Standard Historical Simulation VaR (e.g., fixed 252-day window) suffers from a critical lag:
1.  **Ghost Effect**: A crisis event remains in the window for a year, keeping VaR artificially high long after volatility subsides.
2.  **Slow Reaction**: Emerging crises are diluted by a year of calm data, leading to underestimation of risk at the onset of a crash.

This model addresses these issues by dynamically weighting historical scenarios based on their *regime similarity* to the current market state.

---

## Methodology

### 1. Regime Detection (Unsupervised)
We employ a 2-state **Markov Switching Model** (`statsmodels.MarkovRegression`) on absolute returns (volatility proxy).
-   **States**: "Calm" (Low Variance) vs "Stress" (High Variance).
-   **Estimation**: Rolling window (252 days) with sparse re-estimation (every 7 days) to prevent parameter drift.
-   **No Look-Ahead**: We strictly use **Filtered Marginal Probabilities** $P(S_t | I_t)$, not Smoothed Probabilities $P(S_t | I_T)$. This ensures the model only uses information available at time $t$.

### 2. Weighted Historical Simulation (WHS)
Instead of a simple percentile, we implement a **Regime-Similarity WHS**:
1.  Extract historical returns over a long window (e.g., 365 days).
2.  Assign a weight $w_k$ to each historical day $t-k$ based on how similar its regime probability $P(Stress_{t-k})$ is to today's $P(Stress_t)$.
    $$ w_k \propto 1 - |P(Stress_t) - P(Stress_{t-k})| $$
3.  Calculate the weighted percentile of the return distribution.

This approach ensures that if the market enters a high-volatility regime today, historical high-volatility days are weighted more heavily, regardless of how long ago they occurred.

### 3. Portfolio Construction
-   **Aggregation**: Component returns are fetched as valid OHLCV, converted to simple returns for portfolio aggregation, and then to log returns for statistical modeling.
    $$ R_{port} = \ln(1 + \sum w_i \times (e^{r_i} - 1)) $$
    *Assumption: Daily rebalance, long-only, no leverage. Shorts and intraday leverage require separate treatment.*
-   **Drift & Rebalancing**: Supports realistic "Buy & Hold" drift or periodic rebalancing (e.g., Monthly) with transaction fee models.

---

## Risk Controls

| Control | Implementation |
| :--- | :--- |
| **Walk-Forward** | Training on $[t-W, t]$, application on $[t, t+k]$. No insample leakage. |
| **Data Quality** | Strict validation of OHLCV gaps (>24h) and outliers (>20% daily moves). |
| **Effective Sample Size** | Monitoring of ESS to ensure WHS weights do not degenerate to single-scenario reliance. |
| **Validation Tests** | Kupiec POF (Unconditional Coverage) and Christoffersen (Independence) tests. |

---

## Assumptions & Limitations

### Gaussian vs. t-Student Emissions
We utilize **Gaussian emissions** for the HMM states. 
*   **Why?** While t-Student distributions theoretically model fat tails better, they introduce significant numerical instability and convergence failures in automated rolling-window fitting. 
*   **Mitigation**: The 2-state switching mechanism itself approximates a fat-tailed distribution (mixture of Gaussians), capturing the "volatility clustering" effect robustly without the fragility of t-Student inference.

### Known Issues
-   **Aggressive WHS**: The current Similarity WHS implementation can be overly reactive, leading to breach rates above target (~10% vs 5% at 95% confidence, crypto spot, 2022-2025). Tuning of the similarity decay function is ongoing.
-   **Long-Only Bias**: The risk metrics primarily focus on downside tail risk for long positions.

---

## Outputs & Diagnostics

The pipeline generates a comprehensive dashboard:

1.  **Regime Overlay**: Price actions colored by stress probability.
2.  **Breach Analysis**: Daily PnL vs VaR limits.
3.  **Correlation Matrix**: Rolling heatmap of asset correlations.
4.  **Distribution**: Return histograms with VaR cutoffs.

<img width="1599" height="800" alt="Screenshot_1" src="https://github.com/user-attachments/assets/ec616ef9-308b-4f87-a33b-5738a7560868" />
<img width="872" height="127" alt="Screenshot_3" src="https://github.com/user-attachments/assets/b9304f65-b8b3-4219-85c9-965cde4d259d" />
<img width="477" height="230" alt="Screenshot_2" src="https://github.com/user-attachments/assets/21599da2-0c2d-4a44-b114-351290d8572c" />

---

## Project Structure

```bash
├── settings.py      # Single Source of Truth (Pydantic)
├── main.py          # Pipeline Orchestraion
├── data_fetcher.py  # CCXT + Defensvie Validation
├── markov_model.py  # Rolling HMM Estimation
├── var_model.py     # WHS & Portfolio Engines
├── visualizer.py    # Plotly/Matplotlib Dashboarding
└── dev_context/     # MCP Protocol & Audits
```

---

## Usage

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Configure (or use .env)
# Edit src/hmm_var/settings.py to set capital, portfolio_assets, refit_interval

# 3. Run Pipeline
python main.py
```

---

## Future Improvements
-   [ ] **Monte Carlo Simulation**: Add covariance-based MC for better multi-asset stress testing.
-   [ ] **WHS Tuning**: Optimize the weighting function to bring breach rates closer to the 5% target.
-   [ ] **Dynamic Optimization**: Implement Mean-Variance or Risk Parity weight optimization.

---

## Credits

Developed by **DanRedelien** with architectural assistance from Gemini 3.0 Pro and Claude Opus 4.5.
