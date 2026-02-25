# Data Sources Documentation

## Section 5: Empirical Analysis

### 5.1 Stock Universe Construction

**Source**: CRSP Daily Stock File  
**Download date**: February 13, 2026  
**Original file**: `cus249hhsqzrn47s.csv.gz` (CRSP export)

**Universe Selection Script**: `build_universe_500.py`

**Selection Methodology**:
1. **Rolling universe**: Monthly rebalanced
2. **Selection criterion**: Top 500 stocks by 60-day dollar volume
3. **Dollar volume calculation**: `|Price| × Volume`
4. **Rebalancing frequency**: First trading day of each month
5. **Lookback window**: 60 calendar days prior to month start

**Time Coverage**:
- Start: February 2005
- End: December 2025
- Total months: 251
- Total observations: 125,500 (251 months × 500 stocks)
- Unique tickers over full period: 2,320

**Data Fields**:
- `YearMonth`: Month identifier (YYYY-MM)
- `MonthStart`: First trading day of month
- `Ticker`: Stock ticker symbol
- `DollarVol60d`: Total dollar volume over 60-day window

**Output**: `data/empirical/universe_500.csv` (4.8 MB)

---

### 5.2 Factor-Neutral Returns

**Factor Data Source**: Fama-French Data Library (Kenneth French)
- Fama-French 5 factors: Mkt-RF, SMB, HML, RMW, CMA
- Momentum factor: UMD

**Residualization**:
```
r_residual_{i,t} = r_{i,t} - β_i · F_t
```
where β_i estimated via OLS regression on full sample.

**Script**: `weekly_te_pipeline_500.py` or equivalent

---

### 5.3 Transfer Entropy Features

**TE Estimation**:
- Method: Linear Gaussian TE (Barnett & Seth 2014)
- Window size: T = 60 trading days (~3 months)
- Step size: 5 days (rolling window)
- Lag order: p = 1

**Network-to-Individual Imbalance (NIO)**:
```
NIO_i = (out-degree_i - in-degree_i) / (N-1)
```
where edges defined by LASSO-selected TE coefficients.

**Output**: `data/empirical/te_features_weekly.csv` (33 MB, 357,417 observations)

---

### 5.4 Sample Period for Paper

**Full dataset**: 2005-02 to 2025-12 (251 months)  
**Paper sample**: 2021-01 to 2026-01 (61 months)

**Rationale**:
- Recent AI-driven market regime (2021+)
- Post-COVID normalization
- Sufficient sample size for portfolio sort (61 months × 5 quintiles)

---

## Replication Instructions

### Step 1: Obtain CRSP Data
```bash
# Download CRSP Daily Stock File from WRDS
# Columns needed: PERMNO, Ticker, DlyCalDt, DlyPrc, DlyVol, DlyRet
# Save as: cus249hhsqzrn47s.csv.gz
```

### Step 2: Build Universe
```bash
python build_universe_500.py
# Output: data/empirical/universe_500.csv
```

### Step 3: Download Factor Data
```bash
# Fama-French 5 factors + Momentum
# Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
```

### Step 4: Compute Factor-Neutral Returns & TE
```bash
python weekly_te_pipeline_500.py
# Output: data/empirical/te_features_weekly.csv
```

### Step 5: Run Portfolio Sort
```bash
python src/empirical_portfolio_sort.py
# Output: results/table5_portfolio_sort.txt
```

---

## Data Integrity

**Universe file hash** (SHA256):
```
universe_500.csv: [Same as migrated file]
```

**TE features file hash** (SHA256):
```
te_features_weekly.csv: [Same as migrated file]
```

Both files migrated from original research folder on 2026-02-24.

---

## Notes

1. **CRSP PERMNO stability**: Tickers may change over time; PERMNO used as stable identifier
2. **Survivorship bias**: Not present (universe rebalanced monthly, includes delisted stocks in historical windows)
3. **ETF filtering**: SPY, QQQQ, IWM, SMH appear in early months but filtered out in later analysis
4. **Corporate actions**: CRSP data already adjusted for splits/dividends

---

## Citation

**CRSP Data**:
> Center for Research in Security Prices (CRSP), University of Chicago Booth School of Business. CRSP Daily Stock File. Accessed February 13, 2026.

**Fama-French Factors**:
> Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1-22.

**Momentum Factor**:
> Carhart, M. M. (1997). On persistence in mutual fund performance. The Journal of Finance, 52(1), 57-82.
