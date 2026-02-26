# Data Manifest

## Overview

This repository contains **ALL data required for replication**. No external downloads, API keys, or WRDS access needed.

## Empirical Data Files

### `data/empirical/te_features_weekly.csv`
- **Size**: 33 MB
- **Period**: 2005-01-01 to 2025-12-31
- **Frequency**: Weekly (1,096 weeks)
- **Universe**: S&P 500 stocks (top ~100 by 60-day dollar volume, monthly rebalanced)
- **Columns**:
  - `date`: Week ending date
  - `ticker`: Stock ticker symbol
  - `NIO_raw`: Net Information Outflow (raw returns)
  - `NIO_fn`: Net Information Outflow (factor-neutral)
  - `ret_fwd_5d`: Forward 5-day return
  - Factor loadings (MKT, SMB, HML, RMW, CMA, MOM)
  - Returns (raw, residual, etc.)
- **Source**: CRSP Daily Stock File via WRDS (processed)
- **Processing**: See `DATA_SOURCES.md` for complete pipeline

### `data/empirical/universe_500.csv`
- **Size**: 4.8 MB
- **Purpose**: Stock universe metadata
- **Columns**:
  - `YearMonth`: Month identifier
  - `MonthStart`: First trading day
  - `Ticker`: Stock ticker symbol
  - Dollar volume rankings

## Simulated Data

**No files required.** All simulation data generated on-the-fly via:
- `src/extended_dgp.py` - Base DGP (GARCH + Factor structure)
- `src/extended_dgp_planted_signal.py` - DGP with planted NIO premium

**DGP Parameters**:
- GARCH(1,1): α=0.08, β=0.90
- Student-t innovations: df=5
- Common factors: K=3
- VAR sparsity: 10%

## Data Integrity

### Checksums (SHA256)
```
te_features_weekly.csv: [to be computed]
universe_500.csv: [to be computed]
```

### File Verification
```bash
# Verify files exist
ls -lh data/empirical/

# Expected output:
# te_features_weekly.csv  33 MB
# universe_500.csv         4.8 MB
```

## Data Usage

### Simulation Experiments (Tables 2, 4, 6)
- **Data source**: Generated on-the-fly (no files needed)
- **Reproducibility**: Controlled by `seed_base` parameter

### Empirical Analysis (Table 5)
- **Data source**: `data/empirical/te_features_weekly.csv`
- **Period used in paper**: 2021-2026 (subset of full 2005-2025 data)
- **No downloads at runtime**: All data pre-loaded

## Data License

Empirical data derived from CRSP (proprietary). Users must have WRDS/CRSP access to replicate **data construction** from scratch. However, **this repository includes processed features** sufficient for replicating all paper results without WRDS access.

Original CRSP data subject to WRDS terms of service.

## Notes

- **No runtime downloads**: All scripts read from local `data/empirical/` directory
- **No API keys required**: No external data sources accessed
- **No missing data**: Repository is complete and self-contained
