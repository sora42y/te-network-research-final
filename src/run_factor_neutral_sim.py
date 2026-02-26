"""
FIX 1: Factor-Neutral Simulation Pipeline
Re-run Table 2, 4, 5 with factor-neutral preprocessing

This script implements the THREE preprocessing modes:
  1. raw: Current baseline (no factor adjustment)
  2. oracle_fn: Factor-neutral using TRUE factors from DGP
  3. estimated_fn: Factor-neutral using PCA (RECOMMENDED for paper)

Usage:
  python run_factor_neutral_sim.py --mode estimated_fn --trials 100

Output:
  results/table2_factor_neutral_estimated.csv  (PRIMARY result for paper)
  results/table2_factor_neutral_oracle.csv     (Diagnostic)
  results/table2_raw.csv                       (Appendix: current baseline)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from sklearn.decomposition import PCA
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from extended_dgp import generate_sparse_var_extended
from factor_neutral_preprocessing import preprocess_returns


def estimate_te_network(R, method='ols', penalty_factor=1.0):
    """
    Estimate TE network from return matrix R.
    
    Parameters
    ----------
    R : ndarray (T, N)
        Return matrix
    method : str
        'ols' or 'lasso'
    penalty_factor : float
        For LASSO: multiplier on BIC-selected penalty
    
    Returns
    -------
    A_est : ndarray (N, N)
        Estimated adjacency matrix (binary)
    """
    T, N = R.shape
    A_est = np.zeros((N, N))
    
    if method == 'ols':
        # OLS with 75th percentile threshold (as in paper)
        te_matrix = np.zeros((N, N))
        
        for i in range(N):
            y = R[1:, i]
            
            for j in range(N):
                if i == j:
                    continue
                
                # Restricted model: y_t ~ y_{t-1}
                X_restricted = R[:-1, i].reshape(-1, 1)
                residuals_restricted = y - X_restricted @ np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                sigma2_restricted = np.var(residuals_restricted)
                
                # Unrestricted model: y_t ~ y_{t-1} + x_{j,t-1}
                X_full = np.column_stack([R[:-1, i], R[:-1, j]])
                residuals_full = y - X_full @ np.linalg.lstsq(X_full, y, rcond=None)[0]
                sigma2_full = np.var(residuals_full)
                
                # TE(j → i)
                if sigma2_full > 0 and sigma2_restricted > 0:
                    te_matrix[i, j] = 0.5 * np.log(sigma2_restricted / sigma2_full)
                else:
                    te_matrix[i, j] = 0
        
        # Threshold at 75th percentile
        te_flat = te_matrix[te_matrix > 0]
        if len(te_flat) > 0:
            threshold = np.percentile(te_flat, 75)
            A_est = (te_matrix >= threshold).astype(int)
        
    elif method == 'lasso':
        # LASSO with own lag unpenalized (Frisch-Waugh residualization)
        for i in range(N):
            y = R[1:, i]
            
            # Own lag (unpenalized control)
            own_lag = R[:-1, i].reshape(-1, 1)
            
            # Other stocks' lags (penalized)
            other_idx = [j for j in range(N) if j != i]
            X_others = R[:-1, other_idx]
            
            # Frisch-Waugh: residualize y and X_others on own lag
            # Step 1: regress y on own_lag
            beta_y = np.linalg.lstsq(own_lag, y, rcond=None)[0]
            y_resid = y - own_lag @ beta_y
            
            # Step 2: regress X_others on own_lag
            beta_X = np.linalg.lstsq(own_lag, X_others, rcond=None)[0]
            X_resid = X_others - own_lag @ beta_X
            
            # Fit LASSO on residualized data (cross-lags only)
            try:
                lasso = LassoLarsIC(criterion='bic', max_iter=1000)
                lasso.fit(X_resid, y_resid)
                coef = lasso.coef_  # (N-1,) for other_idx
                
                # Map back to full coefficient vector
                coef_full = np.zeros(N)
                for k, j in enumerate(other_idx):
                    coef_full[j] = coef[k]
                
                A_est[i, :] = (coef_full != 0).astype(int)
            except:
                # If LASSO fails, leave row as zeros
                pass
    
    return A_est


def compute_metrics(A_true, A_est):
    """
    Compute precision, recall, F1, and hub recovery.
    """
    # Flatten to edge lists (excluding diagonal)
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    
    true_edges = A_true[mask]
    pred_edges = A_est[mask]
    
    tp = np.sum((true_edges == 1) & (pred_edges == 1))
    fp = np.sum((true_edges == 0) & (pred_edges == 1))
    fn = np.sum((true_edges == 1) & (pred_edges == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Hub recovery: top-5 out-degree overlap
    true_out = A_true.sum(axis=1)
    pred_out = A_est.sum(axis=1)
    
    true_top5 = set(np.argsort(true_out)[-5:])
    pred_top5 = set(np.argsort(pred_out)[-5:])
    
    hub_overlap = len(true_top5 & pred_top5) / 5.0
    
    # Kendall's tau
    from scipy.stats import kendalltau
    tau, _ = kendalltau(true_out, pred_out)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hub_recovery': hub_overlap,
        'kendall_tau': tau
    }


def run_single_trial(N, T, density, seed, dgp, preprocessing, method):
    """
    Run one Monte Carlo trial.
    
    Returns
    -------
    metrics : dict
    """
    # Generate data
    if dgp == 'garch_factor' and preprocessing in ['oracle_fn', 'estimated_fn']:
        R, A_coef, A_true, F_true = generate_sparse_var_extended(
            N=N, T=T, density=density, seed=seed, dgp=dgp, return_factors=True
        )
    else:
        R, A_coef, A_true = generate_sparse_var_extended(
            N=N, T=T, density=density, seed=seed, dgp=dgp
        )
        F_true = None
    
    # Preprocess
    if preprocessing == 'raw':
        R_proc, _ = preprocess_returns(R, mode='raw')
    elif preprocessing == 'oracle_fn':
        R_proc, _ = preprocess_returns(R, mode='oracle_fn', true_factors=F_true)
    elif preprocessing == 'estimated_fn':
        R_proc, _ = preprocess_returns(R, mode='estimated_fn', n_factors=3)
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}")
    
    # Estimate network
    A_est = estimate_te_network(R_proc, method=method)
    
    # Compute metrics
    metrics = compute_metrics(A_true, A_est)
    
    return metrics


def run_table2_simulation(dgp='garch_factor', preprocessing='estimated_fn', method='ols', n_trials=100):
    """
    Run Table 2 simulation across all (N, T) configurations.
    
    Returns
    -------
    df : DataFrame with columns [N, T, T/N, precision, recall, f1, hub_recovery, kendall_tau]
    """
    # Table 2 configurations
    configs = [
        (30, 60, 100, 250),    # N=30
        (50, 120, 250, 500),   # N=50
        (100, 60, 250, 500, 1000)  # N=100
    ]
    
    results = []
    
    for N_idx, N in enumerate([30, 50, 100]):
        T_values = configs[N_idx]
        
        for T in T_values:
            print(f"\n{'='*60}")
            print(f"Running: N={N}, T={T}, T/N={T/N:.2f}, dgp={dgp}, prep={preprocessing}, method={method}")
            print(f"{'='*60}")
            
            trial_results = []
            
            for trial in range(n_trials):
                if (trial + 1) % 10 == 0:
                    print(f"  Trial {trial+1}/{n_trials}...", end='\r')
                
                seed = 1000 + trial
                metrics = run_single_trial(
                    N=N, T=T, density=0.05, seed=seed, 
                    dgp=dgp, preprocessing=preprocessing, method=method
                )
                trial_results.append(metrics)
            
            # Aggregate
            agg = {
                'N': N,
                'T': T,
                'T_N': T / N,
                'precision_mean': np.mean([r['precision'] for r in trial_results]),
                'precision_std': np.std([r['precision'] for r in trial_results]),
                'recall_mean': np.mean([r['recall'] for r in trial_results]),
                'recall_std': np.std([r['recall'] for r in trial_results]),
                'f1_mean': np.mean([r['f1'] for r in trial_results]),
                'f1_std': np.std([r['f1'] for r in trial_results]),
                'hub_recovery_mean': np.mean([r['hub_recovery'] for r in trial_results]),
                'hub_recovery_std': np.std([r['hub_recovery'] for r in trial_results]),
                'kendall_tau_mean': np.mean([r['kendall_tau'] for r in trial_results]),
                'kendall_tau_std': np.std([r['kendall_tau'] for r in trial_results]),
                'n_trials': n_trials,
                'dgp': dgp,
                'preprocessing': preprocessing,
                'method': method
            }
            results.append(agg)
            
            print(f"\n  -> Precision: {agg['precision_mean']:.3f} +/- {agg['precision_std']:.3f}")
            print(f"  -> Recall: {agg['recall_mean']:.3f} +/- {agg['recall_std']:.3f}")
            print(f"  -> F1: {agg['f1_mean']:.3f}")
            print(f"  -> Hub recovery: {agg['hub_recovery_mean']:.3f}")
            print(f"  -> Kendall's tau: {agg['kendall_tau_mean']:.3f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIX 1: Factor-Neutral Simulation')
    parser.add_argument('--mode', type=str, default='estimated_fn',
                        choices=['raw', 'oracle_fn', 'estimated_fn'],
                        help='Preprocessing mode')
    parser.add_argument('--method', type=str, default='ols',
                        choices=['ols', 'lasso'],
                        help='Estimation method')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--dgp', type=str, default='garch_factor',
                        choices=['gaussian', 'garch', 'garch_factor'],
                        help='DGP type')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"FIX 1: Factor-Neutral Simulation Pipeline")
    print(f"{'='*70}")
    print(f"DGP: {args.dgp}")
    print(f"Preprocessing: {args.mode}")
    print(f"Method: {args.method}")
    print(f"Trials: {args.trials}")
    print(f"{'='*70}\n")
    
    # Run simulation
    df = run_table2_simulation(
        dgp=args.dgp,
        preprocessing=args.mode,
        method=args.method,
        n_trials=args.trials
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'table2_{args.mode}_{args.method}_{args.dgp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("\nSUMMARY TABLE:")
    print(df[['N', 'T', 'T_N', 'precision_mean', 'recall_mean', 'f1_mean', 
              'hub_recovery_mean', 'kendall_tau_mean']].to_string(index=False))
