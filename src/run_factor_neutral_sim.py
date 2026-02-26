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
from sklearn.decomposition import PCA
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Unified imports
from dgp import generate_sparse_var
from factor_neutral_preprocessing import preprocess_returns
from te_core import compute_linear_te_matrix
from evaluation import compute_precision_recall_f1, eval_metrics  # P2 FIX: Added eval_metrics


# P2 FIX: Use unified evaluation functions instead of local duplicate
# compute_metrics functionality now provided by evaluation.eval_metrics


def run_single_trial(N, T, density, seed, dgp, preprocessing, method):
    """
    Run one Monte Carlo trial.
    
    Returns
    -------
    metrics : dict
    """
    # Generate data
    if dgp == 'garch_factor' and preprocessing in ['oracle_fn', 'estimated_fn']:
        R, A_coef, A_true, F_true = generate_sparse_var(
            N=N, T=T, density=density, seed=seed, dgp=dgp, return_factors=True
        )
    else:
        R, A_coef, A_true = generate_sparse_var(
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
    
    # Estimate network (call unified te_core function)
    _, A_est = compute_linear_te_matrix(R_proc, method=method, t_threshold=2.0)
    
    # P2 FIX: Use unified evaluation function
    metrics = eval_metrics(A_true, A_est, top_k=5)
    
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
                
                # P2 FIX: Use centralized seed strategy from simulation_config
                # (For now, keep simple formula for backward compatibility with existing results)
                # TODO: Consider migrating to SimulationConfig.get_seed(trial, N, T) in future
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

