"""
Factor-Neutral Transfer Entropy Network
========================================

Residualize stock returns against Fama-French 5 factors + Momentum,
then compute TE network on idiosyncratic returns only.

Author: Sora
Date: 2026-02-20
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Step 1: Download Fama-French Factors
# ============================================================================

def download_fama_french(start_date='2020-01-01', end_date='2025-12-31'):
    """
    Download Fama-French 5 factors + Momentum from Kenneth French's data library.
    
    Returns:
        pd.DataFrame: Daily factor returns (Mkt-RF, SMB, HML, RMW, CMA, Mom, RF)
    """
    print("Downloading Fama-French 5 factors...")
    ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 
                          'famafrench', 
                          start=start_date, 
                          end=end_date)[0]
    
    print("Downloading Momentum factor...")
    mom = pdr.DataReader('F-F_Momentum_Factor_daily', 
                         'famafrench', 
                         start=start_date, 
                         end=end_date)[0]
    
    # Merge factors
    factors = ff5.join(mom)
    factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
    
    # Convert from % to decimal
    factors = factors / 100.0
    
    print(f"Factors shape: {factors.shape}")
    print(f"Date range: {factors.index[0]} to {factors.index[-1]}")
    
    return factors


# ============================================================================
# Step 2: Download Stock Returns
# ============================================================================

def download_stock_returns(tickers, start_date='2020-01-01', end_date='2025-12-31'):
    """
    Download daily adjusted close prices and compute returns.
    
    Args:
        tickers (list): List of ticker symbols
        
    Returns:
        pd.DataFrame: Daily returns (aligned with factor data)
    """
    print(f"Downloading {len(tickers)} stocks...")
    
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    
    # Compute returns
    returns = prices.pct_change().dropna()
    
    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns


# ============================================================================
# Step 3: Residualize Returns (Factor-Neutral)
# ============================================================================

def residualize_returns(returns, factors):
    """
    Residualize stock returns against Fama-French factors.
    
    For each stock i:
        r_i,t = α + β1·(Mkt-RF) + β2·SMB + β3·HML + β4·RMW + β5·CMA + β6·Mom + ε_i,t
    
    Returns ε_i,t (idiosyncratic return).
    
    Args:
        returns (pd.DataFrame): Stock returns (T x N)
        factors (pd.DataFrame): Factor returns (T x 7)
        
    Returns:
        pd.DataFrame: Residualized returns (T x N)
    """
    print("Residualizing returns against factors...")
    
    # Align dates
    common_dates = returns.index.intersection(factors.index)
    returns_aligned = returns.loc[common_dates]
    factors_aligned = factors.loc[common_dates]
    
    # Prepare factor matrix (add constant)
    X = sm.add_constant(factors_aligned)
    
    residuals = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns)
    
    for stock in returns_aligned.columns:
        y = returns_aligned[stock].dropna()
        X_stock = X.loc[y.index]
        
        # OLS regression
        model = sm.OLS(y, X_stock, missing='drop')
        result = model.fit()
        
        # Store residuals
        residuals.loc[y.index, stock] = result.resid
    
    # Drop rows with any NaN
    residuals = residuals.dropna()
    
    print(f"Residuals shape: {residuals.shape}")
    print(f"Mean residual (should be ~0): {residuals.mean().mean():.6f}")
    
    return residuals


# ============================================================================
# Step 4: Compute Transfer Entropy (placeholder - use your existing code)
# ============================================================================

def compute_transfer_entropy(returns, lag=1, bins=10):
    """
    Compute pairwise Transfer Entropy matrix using vectorized linear TE.
    
    TE(j -> i) measures how much knowing r_j,t-1 reduces uncertainty about r_i,t
    beyond what r_i,t-1 already tells us.
    
    Linear TE formula: TE[j→i] = 0.5 * ln(var_restricted / var_full)
    - Restricted: r_i(t) ~ r_i(t-1)
    - Full: r_i(t) ~ r_i(t-1) + r_j(t-1)
    
    Args:
        returns (pd.DataFrame): Returns (T x N)
        lag (int): Time lag (currently only lag=1 is supported)
        bins (int): Not used (linear TE doesn't need discretization)
        
    Returns:
        pd.DataFrame: TE matrix (N x N), where TE[i,j] = TE from i to j
    """
    print(f"Computing Transfer Entropy (vectorized linear, lag={lag})...")
    
    if lag != 1:
        print(f"Warning: lag={lag} requested, but only lag=1 is supported. Using lag=1.")
    
    # Convert to numpy
    R = returns.values  # (T, N)
    T, N = R.shape
    
    # Prepare lagged data
    R_t = R[1:]       # (T-1, N) current
    R_lag = R[:-1]    # (T-1, N) lagged
    T_eff = T - 1
    
    # Step 1: Compute all restricted residuals (vectorized)
    # r_i(t) = alpha + beta * r_i(t-1) + e
    
    R_t_dm = R_t - R_t.mean(axis=0)           # (T-1, N)
    R_lag_dm = R_lag - R_lag.mean(axis=0)     # (T-1, N)
    
    var_lag = (R_lag_dm ** 2).sum(axis=0)     # (N,)
    var_lag = np.where(var_lag < 1e-10, 1e-10, var_lag)  # prevent division by zero
    
    cov_t_lag = (R_t_dm * R_lag_dm).sum(axis=0)  # (N,)
    beta_r = cov_t_lag / var_lag                  # (N,)
    
    # Restricted residuals: e_i = r_i(t) - beta * r_i(t-1)
    E_restricted = R_t_dm - R_lag_dm * beta_r    # (T-1, N)
    var_restricted = (E_restricted ** 2).sum(axis=0) / T_eff  # (N,)
    
    # Step 2: Standardize
    R_lag_std = R_lag_dm / (np.sqrt(var_lag) + 1e-10)  # (T-1, N)
    E_std = E_restricted / (np.sqrt(var_restricted * T_eff) + 1e-10)  # (T-1, N)
    
    # Step 3: Correlation between residuals and lagged returns
    # corr(e_i, r_j_lag) for all i,j
    corr_e_rlag = E_std.T @ R_lag_std / T_eff  # (N, N): [i, j] = corr(e_i, r_j_lag)
    
    # Step 4: Compute TE matrix
    # TE[j→i] = 0.5 * ln(var_r / var_f) = -0.5 * ln(1 - corr^2)
    corr_sq = corr_e_rlag ** 2
    corr_sq = np.clip(corr_sq, 0, 0.9999)  # prevent log(0)
    
    te_matrix_np = -0.5 * np.log(1 - corr_sq)  # (N, N): [i, j] = TE from j to i
    
    # Transpose to match convention: te_matrix[j, i] = TE from j to i
    te_matrix_np = te_matrix_np.T
    
    # Zero out diagonal
    np.fill_diagonal(te_matrix_np, 0)
    
    # Convert back to DataFrame
    te_matrix = pd.DataFrame(
        te_matrix_np,
        index=returns.columns,
        columns=returns.columns
    )
    
    print(f"TE matrix shape: {te_matrix.shape}")
    print(f"Mean TE: {te_matrix.mean().mean():.6f}")
    print(f"Max TE: {te_matrix.max().max():.6f}")
    print(f"Min TE: {te_matrix.min().min():.6f}")
    
    return te_matrix


# ============================================================================
# Step 5: Visualize TE Networks
# ============================================================================

def plot_te_comparison(te_raw, te_neutral, threshold=0.01):
    """
    Plot side-by-side heatmaps of raw TE vs factor-neutral TE.
    
    Args:
        te_raw (pd.DataFrame): Raw TE matrix
        te_neutral (pd.DataFrame): Factor-neutral TE matrix
        threshold (float): Edge threshold for visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw TE
    sns.heatmap(te_raw, cmap='Reds', vmax=threshold*3, ax=axes[0], cbar_kws={'label': 'TE'})
    axes[0].set_title('Raw TE Network')
    axes[0].set_xlabel('To Stock')
    axes[0].set_ylabel('From Stock')
    
    # Factor-Neutral TE
    sns.heatmap(te_neutral, cmap='Blues', vmax=threshold*3, ax=axes[1], cbar_kws={'label': 'TE'})
    axes[1].set_title('Factor-Neutral TE Network')
    axes[1].set_xlabel('To Stock')
    axes[1].set_ylabel('From Stock')
    
    plt.tight_layout()
    plt.savefig('te_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: te_comparison.png")


def compute_network_stats(te_matrix, threshold=0.01):
    """
    Compute basic network statistics.
    
    Args:
        te_matrix (pd.DataFrame): TE matrix
        threshold (float): Edge threshold
        
    Returns:
        dict: Network statistics
    """
    # Threshold network
    adj = (te_matrix > threshold).astype(int)
    
    stats = {
        'density': adj.sum().sum() / (adj.shape[0] * (adj.shape[0] - 1)),
        'mean_te': te_matrix.mean().mean(),
        'median_te': np.median(te_matrix.values),
        'max_te': te_matrix.max().max(),
        'num_edges': adj.sum().sum(),
        'avg_degree': adj.sum(axis=1).mean()
    }
    
    return stats


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    
    # Configuration
    START_DATE = '2020-01-01'
    END_DATE = '2025-12-31'
    
    # Example tickers (replace with your actual list)
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'JPM', 'V', 'WMT',
        'XOM', 'UNH', 'PG', 'MA', 'HD',
        'CVX', 'BAC', 'ABBV', 'PFE', 'COST'
    ]
    
    print("="*60)
    print("Factor-Neutral Transfer Entropy Pipeline")
    print("="*60)
    
    # Step 1: Download factors
    factors = download_fama_french(START_DATE, END_DATE)
    
    # Step 2: Download stock returns
    returns = download_stock_returns(TICKERS, START_DATE, END_DATE)
    
    # Step 3: Residualize returns
    residuals = residualize_returns(returns, factors)
    
    # Step 4: Compute TE networks
    print("\n" + "="*60)
    print("Computing Raw TE Network...")
    te_raw = compute_transfer_entropy(returns, lag=1, bins=10)
    
    print("\n" + "="*60)
    print("Computing Factor-Neutral TE Network...")
    te_neutral = compute_transfer_entropy(residuals, lag=1, bins=10)
    
    # Step 5: Compare statistics
    print("\n" + "="*60)
    print("Network Statistics Comparison")
    print("="*60)
    
    stats_raw = compute_network_stats(te_raw, threshold=0.01)
    stats_neutral = compute_network_stats(te_neutral, threshold=0.01)
    
    print("\nRaw TE Network:")
    for k, v in stats_raw.items():
        print(f"  {k}: {v:.6f}")
    
    print("\nFactor-Neutral TE Network:")
    for k, v in stats_neutral.items():
        print(f"  {k}: {v:.6f}")
    
    # Step 6: Visualize
    print("\n" + "="*60)
    print("Generating visualizations...")
    plot_te_comparison(te_raw, te_neutral, threshold=0.01)
    
    # Save results
    te_raw.to_csv('te_raw.csv')
    te_neutral.to_csv('te_neutral.csv')
    residuals.to_csv('residuals.csv')
    
    print("\nSaved outputs:")
    print("  - te_raw.csv")
    print("  - te_neutral.csv")
    print("  - residuals.csv")
    print("  - te_comparison.png")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == '__main__':
    main()
