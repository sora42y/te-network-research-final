# Migration & One-Click Replication

## What Changed

### 1. Data Migration
- ✅ Migrated empirical data from `te-network-research/` to `te-network-research-final/data/empirical/`
- ✅ Files: `te_features_weekly.csv` (33 MB), `universe_500.csv` (4.8 MB)
- ✅ Time period: 2005-2025 (20 years of S&P 500 data)

### 2. New Scripts
- ✅ `run_all_experiments.py`: One-click runner for ALL experiments
- ✅ `src/empirical_portfolio_sort.py`: Table 5 (Portfolio Sort)
- ✅ `src/oracle_nio_power.py`: Table 6 (Power Analysis)

### 3. Documentation
- ✅ Updated `README.md` with data sources and one-click instructions
- ✅ Added detailed data description (S&P 500, FF5+Mom, 2021-2026)

## How to Use

```bash
# Run all experiments (simulation + empirical)
python run_all_experiments.py

# Quick test (10 trials)
python run_all_experiments.py --quick
```

## What This Fixes

**Before**:
- ❌ Empirical data scattered across old project folder
- ❌ No clear entry point for replication
- ❌ Manual step-by-step instructions

**After**:
- ✅ All data in final repo
- ✅ Single command runs everything
- ✅ Clear, reproducible pipeline

## Files Changed
- `README.md` (major rewrite)
- `run_all_experiments.py` (new)
- `src/empirical_portfolio_sort.py` (new)
- `src/oracle_nio_power.py` (new)
- `data/empirical/` (new directory with data)

## Next Steps
1. Test `python run_all_experiments.py --quick` (should finish in ~5 min)
2. Verify all tables match paper
3. Commit and push to GitHub
