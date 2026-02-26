# Do Financial Transfer Entropy Networks Recover Meaningful Structure?

**A Matched-DGP Audit of Node-Level Estimation Reliability**

## 🎯 Overview

This repository contains the complete replication package for our working paper examining whether Transfer Entropy (TE) and Granger Causality (GC) networks can reliably recover node-level structure at low T/N ratios typical in financial applications.

**Key Finding**: At T/N < 5, network topology recovery is unreliable. OLS pairwise TE achieves ~11% precision; LASSO-TE reaches 72% on raw returns but only 67% with factor-neutral preprocessing. **The T/N ratio dominates—factor adjustment does not materially improve recovery.**

---

## 🚀 Quick Start

### One-Click: Run ALL experiments
```bash
python run_experiments_modular.py --quick
```
**Output**: All 4 tables in `results/<timestamp>/` (~5 min)

### Run Individual Tables
```bash
# Only Table 2 (main simulation)
python run_experiments_modular.py --tables table2 --quick

# Only Table 5 (empirical)
python run_experiments_modular.py --tables table5

# Multiple tables
python run_experiments_modular.py --tables table2 table4 --quick
```

### Alternative: Direct Script Execution
```bash
# Table 2
python src/run_factor_neutral_sim.py --trials 10

# Table 5
python src/empirical_portfolio_sort.py
```

**Runtime**:
- Quick mode (`--quick`): ~5 minutes
- Full mode (100 trials): ~30-60 minutes

---

## 📂 Repository Structure

```
.
├── run_experiments_modular.py  # 🚀 ONE-CLICK MODULAR RUNNER (start here!)
├── run_all_experiments.py      # Legacy one-click runner (deprecated)
├── compare_runs.py             # Compare results across different runs
├── results_manager.py          # Results versioning system
├── simulation_config.py        # Dual-mode seed configuration
├── experiment_metadata.py      # SHA256 fingerprinting & lineage tracking
│
├── paper/
│   ├── main.tex              # LaTeX source
│   └── references.bib        # Bibliography
│
├── src/                      # Python code (all experiments)
│   ├── te_core.py            # ⭐ CORE: Unified TE implementations
│   ├── extended_dgp.py       # GARCH+t5+Factor DGP
│   ├── run_factor_neutral_sim.py     # Table 2 (Main Results)
│   ├── all_experiments_v2.py         # Table 4 (Oracle vs Estimated)
│   ├── empirical_portfolio_sort.py   # Table 5 (Portfolio Sort)
│   └── oracle_nio_power.py           # Table 6 (Power Analysis)
│
├── data/
│   └── empirical/
│       ├── te_features_weekly.csv    # S&P 500 NIO data (2005-2025, 33 MB)
│       └── universe_500.csv          # Stock universe (4.8 MB)
│
├── results/                  # Versioned experiment results
│   ├── <run_id>/             # Each run gets its own directory
│   │   ├── run_metadata.json # Git commit, timestamp, params
│   │   ├── README.txt        # Human-readable summary
│   │   ├── table2.csv
│   │   ├── table4.csv
│   │   └── ...
│   └── ...
│
└── docs/
    ├── DATA_SOURCES.md       # Complete data lineage
    ├── REPRODUCIBILITY.md    # Dual-mode workflow guide
    └── CODE_CONSOLIDATION_PLAN.md  # Code audit notes
```

---

## 📊 Data

### Simulated Data (Tables 2, 4, 6)
All simulations generated on-the-fly using `src/extended_dgp.py`:
- **GARCH(1,1)**: α=0.08, β=0.90 (Engle & Bollerslev 1986)
- **t(5) innovations**: Fat tails (kurtosis ≈ 9, matching real equity)
- **K=3 common factors**: Mimicking Fama-French structure
- **Sparse VAR(1)**: 10% density, uniformly distributed

**No external data required** for simulation experiments.

### Empirical Data (Table 5)
S&P 500 portfolio sort analysis:
- **Period**: 2021-2026 (full sample), split into 2 sub-periods
- **Universe**: Top ~100 stocks by 60-day dollar volume (monthly rebalanced)
- **Factor adjustment**: Fama-French 5 factors + Momentum
- **TE estimation**: 60-day rolling windows, 5-day steps

**Data files** (included):
- `data/empirical/te_features_weekly.csv` (33 MB)
- `data/empirical/universe_500.csv` (4.8 MB)

**Data source**: CRSP (users must have WRDS access to replicate from scratch)
**Full pipeline**: See `DATA_SOURCES.md`

---

## 🔬 Run Experiments

### Method 1: Modular Workflow (Recommended)

**Run specific tables**:
```bash
# Single table
python run_experiments_modular.py --tables table2 --run-id test1

# Multiple tables
python run_experiments_modular.py --tables table2 table4 --quick
```

**Results**: Auto-saved to `results/<run_id>/` with metadata

---

### Method 2: Direct Execution

**Run scripts directly** (no versioning):
```bash
# Table 2 (main simulation)
python src/run_factor_neutral_sim.py --trials 100

# Table 4 (oracle vs estimated)
python src/all_experiments_v2.py --trials 100 --experiments 3

# Table 5 (empirical portfolio sort)
python src/empirical_portfolio_sort.py

# Table 6 (power analysis)
python src/oracle_nio_power.py --trials 50
```

**Results**: Saved to `results/*.csv` (may overwrite)

---

## 🎯 Key Results Summary

### Table 2: Main Simulation Results (GARCH+Factor DGP)

| Estimator | Preprocessing | T/N=2 | T/N=5 | T/N=10 |
|-----------|---------------|-------|-------|--------|
| OLS-TE | Raw | 8.1% | 11.5% | 12.6% |
| OLS-TE | Factor-neutral (Est.) | 7.0% | 11.0% | 11.8% |
| LASSO-TE | Raw | 17.3% | 23.1% | 9.4% |
| LASSO-TE | Factor-neutral (Est.) | 14.2% | 28.1% | 10.1% |

**Precision** = TP / (TP + FP)

### Table 4: Oracle vs Estimated Factor-Neutral (T/N=5)

| Method | Raw | Oracle FN | Estimated FN |
|--------|-----|-----------|--------------|
| LASSO-TE | 44.4% | **74.0%** | 68.9% |

**Key insight**: Factor-neutral helps ONLY if you know the true factors (Oracle). Estimated factors (PCA) don't improve much.

### Table 5: Empirical Portfolio Sort (NEW RESULTS)

| Quintile | Ann. Return | t-stat |
|----------|-------------|--------|
| Q1 (Low NIO) | +18.71% | 1.88 |
| Q5 (High NIO) | +6.03% | 0.59 |
| **L/S** | **-12.68%** | **-2.40** |

**Significant negative spread**: High NIO stocks UNDERPERFORM. Signal reversal suggests estimation noise dominates.

---

## 📋 Requirements
```bash
pip install -r requirements.txt
```
Python 3.8+, NumPy, SciPy, scikit-learn, pandas

---

## 🔐 Reproducibility

### Fixed Seed (Paper Version)
```bash
python run_experiments_modular.py --run-id paper_final
```
Generates exact same results every time.

### Robustness Check
```bash
python compare_runs.py run1 run2 --table table2
```
Compare results across different runs. Expected CV < 5%.

**Metadata tracking**: Every run generates `run_metadata.json` with git commit, timestamp, SHA256 hashes.

---

## 🏗️ Code Architecture

**Core Module**: `src/te_core.py` (single source of truth)

```python
from te_core import compute_linear_te_matrix, compute_nio

# OLS-TE
te_matrix, adj = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)

# LASSO-TE
te_matrix, adj = compute_linear_te_matrix(R, method='lasso')

# Net Information Outflow
nio = compute_nio(te_matrix, method='binary')
```

**All experiments import from `te_core.py`** (no duplicate implementations).

---

## 📚 Documentation

- `DATA_SOURCES.md`: Complete data lineage (CRSP → TE features)
- `REPRODUCIBILITY.md`: Dual-mode workflow & robustness validation
- `CODE_CONSOLIDATION_PLAN.md`: Code audit notes

---

## 🧪 Code Verification

```bash
# Check implementation consistency
python audit_code_consistency.py

# Expected output:
# TE difference: 0.00e+00
# All imports: OK
```
