"""
Extended Simulation DGP: GARCH(1,1) + Common Factor Structure
Drop-in replacement for generate_sparse_var() in lasso_simulation.py

Three DGPs:
  1. Baseline: i.i.d. Gaussian (original)
  2. GARCH: GARCH(1,1) innovations (fat tails + volatility clustering)
  3. Factor+GARCH: Common factor structure + GARCH innovations (realistic)

Usage:
  Replace generate_sparse_var() calls with generate_sparse_var_extended(dgp='garch_factor')
  Everything else in the simulation pipeline stays the same.
"""

import numpy as np


def generate_sparse_var_extended(N=50, T=500, density=0.05, seed=42, dgp='gaussian'):
    """
    Generate sparse VAR(1) data with realistic DGP options.
    
    Parameters
    ----------
    dgp : str
        'gaussian'     : i.i.d. N(0, σ²I) innovations (original baseline)
        'garch'        : GARCH(1,1) innovations per stock, no cross-sectional correlation
        'garch_factor' : K common factors + GARCH(1,1) idiosyncratic innovations
                         + contemporaneous cross-sectional correlation via factor structure
    
    Returns: R (T×N), A (N×N coefficient matrix), A_true (N×N binary adjacency)
    """
    rng = np.random.RandomState(seed)
    
    # ---- Step 1: Sparse VAR(1) coefficient matrix (same for all DGPs) ----
    A = np.zeros((N, N))
    n_edges = int(N * N * density)
    
    off_diag = [(i, j) for i in range(N) for j in range(N) if i != j]
    edge_idx = rng.choice(len(off_diag), size=n_edges, replace=False)
    
    for idx in edge_idx:
        i, j = off_diag[idx]
        A[i, j] = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
    
    eigvals = np.abs(np.linalg.eigvals(A))
    if eigvals.max() > 0.9:
        A = A * (0.9 / eigvals.max())
    
    A_true = (A != 0).astype(int)
    
    # ---- Step 2: Generate innovations based on DGP ----
    sigma_base = 0.01  # daily return volatility scale
    
    if dgp == 'gaussian':
        # Original: i.i.d. N(0, σ²)
        innovations = rng.normal(0, sigma_base, (T, N))
    
    elif dgp == 'garch':
        # GARCH(1,1) per stock: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
        # Calibrated to typical equity GARCH parameters
        omega = sigma_base**2 * 0.05   # ~5% of unconditional variance from constant
        alpha = 0.08                    # news coefficient
        beta  = 0.90                    # persistence
        # unconditional var = omega / (1 - alpha - beta) = omega / 0.02 ≈ sigma_base^2 * 2.5
        
        innovations = np.zeros((T, N))
        h = np.full(N, sigma_base**2)  # initial conditional variance
        
        for t in range(T):
            z = rng.standard_t(df=5, size=N)  # t(5) for fat tails
            innovations[t] = np.sqrt(h) * z
            if t < T - 1:
                h = omega + alpha * innovations[t]**2 + beta * h
                h = np.maximum(h, 1e-10)  # floor
        
    elif dgp == 'garch_factor':
        # Factor structure: r_i,t = β_i' F_t + u_i,t
        # where F_t are K common factors and u_i,t has GARCH dynamics
        # The VAR acts on TOTAL returns, so factors create contemporaneous correlation
        
        K = 3  # number of common factors (market, size, sector)
        
        # Factor loadings: drawn from N(0, 1), then scaled
        # This creates cross-sectional correlation structure
        B = rng.normal(0, 1, (N, K)) * 0.4  # loadings matrix
        
        # Factor returns: AR(1) with moderate persistence
        F = np.zeros((T, K))
        factor_vol = sigma_base * 1.5  # factors are more volatile than idiosyncratic
        for t in range(1, T):
            F[t] = 0.1 * F[t-1] + rng.normal(0, factor_vol, K)
        
        # Idiosyncratic innovations with GARCH(1,1) + t(5)
        omega = (sigma_base * 0.7)**2 * 0.05
        alpha = 0.08
        beta  = 0.90
        
        u = np.zeros((T, N))
        h = np.full(N, (sigma_base * 0.7)**2)
        
        for t in range(T):
            z = rng.standard_t(df=5, size=N)
            u[t] = np.sqrt(h) * z
            if t < T - 1:
                h = omega + alpha * u[t]**2 + beta * h
                h = np.maximum(h, 1e-10)
        
        # Total innovation = factor component + idiosyncratic
        innovations = F @ B.T + u
        
    else:
        raise ValueError(f"Unknown dgp: {dgp}. Use 'gaussian', 'garch', or 'garch_factor'.")
    
    # ---- Step 3: Generate VAR(1) process ----
    R = np.zeros((T, N))
    R[0] = rng.normal(0, sigma_base, N)
    
    for t in range(1, T):
        R[t] = A @ R[t-1] + innovations[t]
    
    return R, A, A_true


def verify_dgp_properties(N=50, T=500, seed=42):
    """
    Quick diagnostic: compare the three DGPs on key statistical properties.
    Run this once to verify calibration is reasonable.
    """
    print(f"{'DGP':<15} {'Mean|ret|':<12} {'Std(ret)':<12} {'Kurt(ret)':<12} {'Mean xcorr':<12}")
    print("-" * 63)
    
    for dgp in ['gaussian', 'garch', 'garch_factor']:
        R, _, _ = generate_sparse_var_extended(N=N, T=T, seed=seed, dgp=dgp)
        
        # Mean absolute return
        mean_abs = np.mean(np.abs(R[1:]))
        # Std of returns
        std_ret = np.std(R[1:])
        # Excess kurtosis (averaged across stocks)
        from scipy.stats import kurtosis
        kurt = np.mean([kurtosis(R[1:, i]) for i in range(N)])
        # Mean pairwise cross-sectional correlation
        corr_matrix = np.corrcoef(R[1:].T)
        np.fill_diagonal(corr_matrix, np.nan)
        mean_xcorr = np.nanmean(corr_matrix)
        
        print(f"{dgp:<15} {mean_abs:<12.6f} {std_ret:<12.6f} {kurt:<12.2f} {mean_xcorr:<12.4f}")


# ============================================================================
# Integration: how to use in the existing simulation pipeline
# ============================================================================
# 
# In lasso_simulation.py, replace:
#
#   R, A_true_coef, A_true = generate_sparse_var(N=N, T=T, density=density, seed=seed)
#
# with:
#
#   R, A_true_coef, A_true = generate_sparse_var_extended(
#       N=N, T=T, density=density, seed=seed, dgp='garch_factor'
#   )
#
# Then run the same evaluation pipeline. Expected results:
#   - Gaussian:      baseline (what you already have)
#   - GARCH:         similar or slightly worse precision/recall
#   - GARCH+Factor:  notably worse (contemporaneous correlation creates spurious
#                    partial correlations that inflate false positives)
#
# For the paper, report all three DGPs in Table 2 (expanded).
# The key message: "even under i.i.d. Gaussian — the most favorable DGP —
# network recovery fails at T/N < 1. Under realistic financial dynamics,
# it is strictly worse."
# ============================================================================


if __name__ == '__main__':
    print("DGP Verification (N=50, T=500)")
    print("=" * 63)
    verify_dgp_properties(N=50, T=500)
    
    print("\n\nDGP Verification (N=100, T=60) — your real data regime")
    print("=" * 63)
    verify_dgp_properties(N=100, T=60)
