"""
nonparametric_te.py
非参数 Transfer Entropy 估计（KNN-based Shannon entropy）
对比实验：Linear TE (OLS/LASSO) vs Nonparametric TE (KNN)

目标：证明在金融 T/N 比率下，非参数方法的维度灾难比线性方法更严重

Method:
  - KNN-based entropy estimation (Kozachenko-Leonenko estimator)
  - Embedding dimension d=1 (conservative, avoids curse of dimensionality)
  - k=3 nearest neighbors (standard choice)
  
References:
  - Kraskov et al. (2004) "Estimating mutual information"
  - Barnett et al. (2009) "Granger Causality and Transfer Entropy Are Equivalent for Gaussian Variables"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from scipy.special import digamma
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extended_dgp import generate_sparse_var_extended
from te_core import compute_linear_te_matrix

# Use relative path from repo root
REPO_ROOT = Path(__file__).parent.parent
OUTPUT = REPO_ROOT / "results"
SEED_BASE = 42
N_TRIALS = 50  # Reduced from 100 due to computational cost
TOP_K = 5


def knn_entropy(X, k=3):
    """
    KNN-based Shannon entropy estimator (Kozachenko-Leonenko)
    
    H(X) ≈ -ψ(k) + ψ(n) + log(c_d) + (d/n) Σ log(ε_i)
    where ε_i is distance to k-th nearest neighbor
    
    Parameters
    ----------
    X : (n, d) array
        n samples, d dimensions
    k : int
        Number of nearest neighbors
    
    Returns
    -------
    H : float
        Entropy estimate in nats
    """
    n, d = X.shape
    
    if n < k + 1:
        return np.nan  # Not enough samples
    
    tree = KDTree(X)
    # Query k+1 because first neighbor is the point itself
    distances, _ = tree.query(X, k=k+1)
    
    # Take k-th neighbor distance (index k, since 0 is self)
    epsilon = distances[:, k]
    
    # Volume of d-dimensional unit ball: c_d = π^(d/2) / Γ(d/2 + 1)
    if d == 1:
        c_d = 2.0
    elif d == 2:
        c_d = np.pi
    else:
        from scipy.special import gamma
        c_d = (np.pi ** (d/2)) / gamma(d/2 + 1)
    
    H = -digamma(k) + digamma(n) + np.log(c_d) + (d / n) * np.sum(np.log(epsilon + 1e-10))
    return H


def transfer_entropy_knn(X, Y, lag=1, k=3):
    """
    Transfer Entropy: TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    
    Parameters
    ----------
    X, Y : (T,) arrays
        Time series (must be same length)
    lag : int
        Time lag
    k : int
        Number of nearest neighbors for entropy estimation
    
    Returns
    -------
    te : float
        Transfer entropy in nats
    """
    T = len(X)
    if T < lag + 10:
        return np.nan
    
    # Construct embedding vectors
    Y_present = Y[lag:].reshape(-1, 1)          # Y_t
    Y_past    = Y[:-lag].reshape(-1, 1)         # Y_{t-1}
    X_past    = X[:-lag].reshape(-1, 1)         # X_{t-1}
    
    n = len(Y_present)
    
    # H(Y_t | Y_{t-1}) ≈ H(Y_t, Y_{t-1}) - H(Y_{t-1})
    YY = np.hstack([Y_present, Y_past])
    H_YY = knn_entropy(YY, k=k)
    H_Yp = knn_entropy(Y_past, k=k)
    H_Y_given_Yp = H_YY - H_Yp
    
    # H(Y_t | Y_{t-1}, X_{t-1}) ≈ H(Y_t, Y_{t-1}, X_{t-1}) - H(Y_{t-1}, X_{t-1})
    YYX = np.hstack([Y_present, Y_past, X_past])
    XY  = np.hstack([Y_past, X_past])
    H_YYX = knn_entropy(YYX, k=k)
    H_XYp = knn_entropy(XY, k=k)
    H_Y_given_XYp = H_YYX - H_XYp
    
    te = H_Y_given_Yp - H_Y_given_XYp
    return te


def compute_nonparametric_te_matrix(R, k=3, lag=1, threshold_percentile=95):
    """
    Compute N×N transfer entropy matrix using KNN method
    
    Parameters
    ----------
    R : (T, N) array
        Time series data
    k : int
        KNN parameter
    lag : int
        Time lag
    threshold_percentile : float
        Percentile threshold for edge detection (default 95 = top 5%)
    
    Returns
    -------
    TE : (N, N) array
        Transfer entropy matrix (raw values)
    A_pred : (N, N) binary array
        Predicted adjacency matrix (thresholded)
    """
    T, N = R.shape
    TE = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            TE[i, j] = transfer_entropy_knn(R[:, j], R[:, i], lag=lag, k=k)
    
    # Threshold: top (100 - threshold_percentile)% of off-diagonal values
    off_diag = TE[~np.eye(N, dtype=bool)]
    valid = off_diag[~np.isnan(off_diag)]
    
    if len(valid) == 0:
        return TE, np.zeros((N, N), dtype=int)
    
    cutoff = np.percentile(valid, threshold_percentile)
    A_pred = (TE > cutoff).astype(int)
    np.fill_diagonal(A_pred, 0)
    
    return TE, A_pred


def eval_metrics(A_true, A_pred, top_k=5):
    """Compute precision, recall, F1, hub recovery"""
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool).flatten()
    yt = A_true.flatten()[mask]
    yp = A_pred.flatten()[mask]
    
    true_od = A_true.sum(1)
    pred_od = A_pred.sum(1)
    true_h = set(np.argsort(true_od)[-top_k:])
    pred_h = set(np.argsort(pred_od)[-top_k:])
    
    return dict(
        precision    = precision_score(yt, yp, zero_division=0),
        recall       = recall_score(yt, yp, zero_division=0),
        f1           = f1_score(yt, yp, zero_division=0),
        hub_recovery = len(true_h & pred_h) / top_k,
        density      = A_pred.sum() / (N * (N-1))
    )


def run_nonparametric_comparison():
    """
    Compare Linear TE (OLS/LASSO) vs Nonparametric TE (KNN)
    Same DGP (GARCH+Factor), multiple (N, T) configurations
    """
    
    # Focus on critical T/N ratios
    CONFIGS = [
        (20,  12),   # T/N = 0.6
        (20,  50),   # T/N = 2.5
        (20, 100),   # T/N = 5.0
        (50,  30),   # T/N = 0.6
        (50, 125),   # T/N = 2.5
        (50, 250),   # T/N = 5.0
    ]
    
    results = []
    
    for N, T in tqdm(CONFIGS, desc="Config"):
        T_over_N = T / N
        
        for trial in tqdm(range(N_TRIALS), desc=f"  N={N}, T={T}", leave=False):
            seed = SEED_BASE + trial
            
            # Generate data
            R, A, A_true = generate_sparse_var_extended(
                N=N, T=T, density=0.05, seed=seed, dgp='garch_factor'
            )
            
            # --- Linear methods (baseline) ---
            try:
                _, A_ols = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)
                metrics_ols = eval_metrics(A_true, A_ols, top_k=TOP_K)
            except:
                metrics_ols = dict(precision=np.nan, recall=np.nan, f1=np.nan, 
                                   hub_recovery=np.nan, density=np.nan)
            
            try:
                _, A_lasso = compute_linear_te_matrix(R, method='lasso')
                metrics_lasso = eval_metrics(A_true, A_lasso, top_k=TOP_K)
            except:
                metrics_lasso = dict(precision=np.nan, recall=np.nan, f1=np.nan,
                                     hub_recovery=np.nan, density=np.nan)
            
            # --- Nonparametric KNN method ---
            try:
                _, A_knn = compute_nonparametric_te_matrix(R, k=3, lag=1, threshold_percentile=95)
                metrics_knn = eval_metrics(A_true, A_knn, top_k=TOP_K)
            except Exception as e:
                metrics_knn = dict(precision=np.nan, recall=np.nan, f1=np.nan,
                                   hub_recovery=np.nan, density=np.nan)
            
            # Store results
            results.append({
                'N': N, 'T': T, 'T/N': T_over_N, 'trial': trial,
                'method': 'OLS', **metrics_ols
            })
            results.append({
                'N': N, 'T': T, 'T/N': T_over_N, 'trial': trial,
                'method': 'LASSO', **metrics_lasso
            })
            results.append({
                'N': N, 'T': T, 'T/N': T_over_N, 'trial': trial,
                'method': 'KNN', **metrics_knn
            })
    
    # Save raw results
    df = pd.DataFrame(results)
    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / 'nonparametric_te_comparison.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT / 'nonparametric_te_comparison.csv'}")
    
    # Compute summary statistics
    summary = df.groupby(['N', 'T', 'T/N', 'method']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'hub_recovery': ['mean', 'std'],
        'density': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(OUTPUT / 'nonparametric_te_summary.csv', index=False)
    print(f"✓ Saved: {OUTPUT / 'nonparametric_te_summary.csv'}")
    
    # Print key comparisons
    print("\n" + "="*80)
    print("NONPARAMETRIC TE vs LINEAR TE — PRECISION COMPARISON")
    print("="*80)
    
    for (N, T, tn), group in summary.groupby(['N', 'T', 'T/N']):
        print(f"\nN={N}, T={T}, T/N={tn:.1f}")
        print("-" * 60)
        for _, row in group.iterrows():
            method = row['method']
            prec_mean = row['precision_mean']
            prec_std = row['precision_std']
            print(f"  {method:8s}  Precision: {prec_mean:5.1%} ± {prec_std:5.1%}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("If KNN precision ≈ random at low T/N, nonparametric TE is impractical.")
    print("Linear TE (despite being 'wrong' for nonlinear dynamics) is the only")
    print("computationally feasible approach under financial sample constraints.")
    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("NONPARAMETRIC TRANSFER ENTROPY COMPARISON")
    print("Linear (OLS/LASSO) vs Nonparametric (KNN-based Shannon Entropy)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Trials per config: {N_TRIALS}")
    print(f"  - DGP: GARCH+Factor (realistic)")
    print(f"  - KNN k=3, lag=1, embedding_dim=1")
    print(f"  - Edge density: 5%")
    print(f"  - Hub definition: top-{TOP_K} out-degree nodes")
    print(f"\nExpected runtime: ~30-60 min (KNN is slow for large N)")
    print("\nStarting...\n")
    
    run_nonparametric_comparison()
