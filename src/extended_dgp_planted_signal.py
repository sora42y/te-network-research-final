"""
extended_dgp_planted_signal.py
DGP with planted NIO premium for power analysis

Modification: Add network information outflow (NIO) premium to expected returns
r_{i,t} = ... + lambda_NIO * NIO_std_i + ...

where NIO_i = (out_flow_i - in_flow_i) / (N-1) is cross-sectional constant
"""

import numpy as np
from extended_dgp import generate_sparse_var_extended


def generate_sparse_var_with_nio_premium(N=50, T=500, density=0.05, seed=42, 
                                          dgp='garch_factor', lambda_NIO=0.01):
    """
    Generate sparse VAR(1) data with planted NIO premium.
    
    Parameters
    ----------
    lambda_NIO : float
        NIO premium coefficient in daily return units.
        Example: 0.01 corresponds to ~1% daily return difference per std NIO,
        or roughly 250 * 0.01 = 2.5% annualized L/S spread (ignoring compounding).
        More precisely: annualized L/S spread ≈ lambda_NIO * sqrt(252) * 2
        (factor of 2 from long-short, sqrt(252) from daily to annual vol scaling)
    
    Returns
    -------
    R : (T, N) array
        Returns with planted premium
    A : (N, N) array
        VAR coefficient matrix
    A_true : (N, N) binary array
        True adjacency matrix
    NIO_true : (N,) array
        True NIO (cross-sectional constant)
    """
    # Generate base VAR data without premium
    R_base, A, A_true = generate_sparse_var_extended(
        N=N, T=T, density=density, seed=seed, dgp=dgp
    )
    
    # Compute oracle NIO from true A (P1 FIX: corrected direction)
    # Matrix convention: A[i,j] = j->i
    # Therefore: column sums = out-degree, row sums = in-degree
    out_flow = A_true.sum(axis=0)  # column sums = out-degree
    in_flow = A_true.sum(axis=1)   # row sums = in-degree
    NIO_true = (out_flow - in_flow) / (N - 1)
    
    # Standardize NIO cross-sectionally (mean=0, std=1)
    NIO_std = (NIO_true - NIO_true.mean()) / (NIO_true.std() + 1e-10)
    
    # Add planted premium to returns
    # Shape: (T, N) + (N,) broadcasts to (T, N)
    premium = lambda_NIO * NIO_std  # (N,) cross-sectional constant
    R = R_base + premium[np.newaxis, :]  # Add to each time period
    
    return R, A, A_true, NIO_std


def test_planted_signal():
    """Quick test: verify premium is embedded correctly"""
    np.random.seed(42)
    
    lambda_NIO = 0.01
    R, A, A_true, NIO_std = generate_sparse_var_with_nio_premium(
        N=50, T=1000, lambda_NIO=lambda_NIO, dgp='gaussian'
    )
    
    # Cross-sectional regression: mean(R_i) ~ NIO_std_i
    mean_returns = R.mean(axis=0)
    
    # OLS regression
    X = np.column_stack([np.ones(50), NIO_std])
    beta = np.linalg.lstsq(X, mean_returns, rcond=None)[0]
    
    print(f"Planted lambda: {lambda_NIO:.4f}")
    print(f"Recovered beta: {beta[1]:.4f}")
    print(f"Ratio: {beta[1] / lambda_NIO:.2f}")
    print("\nExpected: ratio ≈ 1.0 (exact recovery in Gaussian case with large T)")
    
    # t-stat
    residuals = mean_returns - X @ beta
    se = np.sqrt((residuals**2).sum() / (50 - 2)) / np.sqrt((NIO_std**2).sum())
    t_stat = beta[1] / se
    print(f"t-statistic: {t_stat:.2f}")
    

if __name__ == '__main__':
    print("="*80)
    print("Testing Planted NIO Premium DGP")
    print("="*80)
    test_planted_signal()
