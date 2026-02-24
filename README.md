# Do Financial Transfer Entropy Networks Recover Meaningful Structure?

**A Matched-DGP Audit of Node-Level Estimation Reliability**

## Overview

This repository contains the replication package for our working paper examining whether Transfer Entropy (TE) and Granger Causality (GC) networks can reliably recover node-level structure at low T/N ratios typical in financial applications.

**Key Finding**: At T/N < 5, network topology recovery is unreliable. OLS pairwise TE achieves ~11% precision; LASSO-TE reaches 72% on raw returns but only 67% with factor-neutral preprocessing. The T/N ratio dominates—factor adjustment does not materially improve recovery.

## Repository Structure

```
.
├── run_all_experiments.py    # 🚀 ONE-CLICK RUNNER (start here!)
├── paper/
│   ├── main.tex              # LaTeX source
│   └── references.bib        # Bibliography
├── paper_assets/             # Figures for paper
├── src/                      # Python simulation & empirical code
│   ├── all_experiments_v2.py         # Table 4 (Oracle vs Estimated)
│   ├── extended_dgp.py               # GARCH+t5 DGP
│   ├── lasso_simulation.py           # LASSO-TE estimation
│   ├── run_factor_neutral_sim.py     # Table 2 (Main Results)
│   ├── empirical_portfolio_sort.py   # Table 5 (Portfolio Sort)
│   └── oracle_nio_power.py           # Table 6 (Power Analysis)
├── data/
│   └── empirical/
│       ├── te_features_weekly.csv    # S&P 500 NIO data (2005-2025)
│       └── universe_500.csv          # Stock universe (monthly rebalanced)
└── results/                  # Generated tables and figures
```

## Data

### Simulated Data (Tables 2, 4, 6)
All simulations are generated on-the-fly using the DGP in `src/extended_dgp.py`:
- **GARCH(1,1) + t(5) innovations** (volatility clustering + fat tails)
- **Common factor structure** (K=3 factors, mimicking Fama-French)
- **Sparse VAR(1) network** (10% density)

No external data required for simulation experiments.

### Empirical Data (Table 5)
Real market data for portfolio sort analysis:
- **Source**: S&P 500 constituent stocks
- **Period**: 2021-2026 (full sample), 2021-2023 & 2023-2026 (sub-periods)
- **Universe**: Top ~100 stocks by 60-day average dollar volume (monthly rebalanced)
- **Factor data**: Fama-French 5 factors + Momentum (Kenneth French Data Library)
- **Returns**: Factor-neutral returns (residuals from FF5+Mom regression)

**Data files** (included in `data/empirical/`):
- `te_features_weekly.csv`: Pre-computed NIO and forward returns (33 MB)
- `universe_500.csv`: Stock universe with monthly rebalancing (4.8 MB)

**Data generation pipeline** (if you want to rebuild from scratch):
1. Download S&P 500 daily prices (2005-2025) from yfinance or WRDS
2. Compute rolling TE networks (T=60 days, 5-day step)
3. Factor-neutralize returns using FF5+Mom
4. Compute NIO and forward returns
5. Save to `data/empirical/te_features_weekly.csv`

(Script: `te-network-research/weekly_te_pipeline_500_v2.py` - see archived project)

## Replication Instructions

### Requirements
- Python 3.8+
- NumPy, SciPy, scikit-learn, pandas, matplotlib
- LaTeX (for paper compilation)

### 🚀 One-Click Replication

**Run ALL experiments** (simulation + empirical):
```bash
python run_all_experiments.py
```

This will:
1. Generate Table 2 (Main Results: GARCH+Factor DGP)
2. Generate Table 4 (Oracle vs Estimated Factor-Neutral)
3. Generate Table 5 (Portfolio Sort on NIO)
4. Generate Table 6 (Oracle NIO Power Analysis)
5. Save all results to `results/` as CSV + formatted tables

**Quick mode** (10 trials instead of 100, for testing):
```bash
python run_all_experiments.py --quick
```

**Expected runtime**:
- Quick mode: ~5 minutes
- Full mode: ~30-60 minutes (depending on CPU)

### Manual Replication (Individual Tables)

**Table 2 (Main Results)**:
```bash
cd src
python run_factor_neutral_sim.py --trials 100
```

**Table 4 (Oracle vs Estimated)**:
```bash
python all_experiments_v2.py --trials 100 --experiments 3
```

**Table 5 (Portfolio Sort)**:
```bash
python empirical_portfolio_sort.py
```

**Table 6 (Power Analysis)**:
```bash
python oracle_nio_power.py --trials 50
```

Results will be saved to `results/` as CSV files.

### Compile Paper
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Results Summary

| Estimator | Preprocessing | T/N=2 | T/N=5 |
|-----------|---------------|-------|-------|
| OLS-TE | Raw | 11.3% | 11.5% |
| LASSO-TE | Raw | 45.1% | 72.3% |
| LASSO-TE | Factor-neutral (PCA) | 35.5% | 66.7% |
| LASSO-TE | Factor-neutral (Oracle) | 25.3% | 74.6% |

**Precision** = True positives / (True positives + False positives)

Factor-neutral preprocessing does not materially improve precision; the T/N barrier persists regardless of preprocessing choice.

## Citation

```bibtex
@unpublished{te-network-audit-2026,
  author = {[Your Name]},
  title  = {Do Financial Transfer Entropy Networks Recover Meaningful Structure? 
            A Matched-DGP Audit of Node-Level Estimation Reliability},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

[Your contact information]

---

**Note**: This is the clean replication branch. Full experimental history is available in the private `master` branch.
