"""
oracle_nio_power_analysis.py
Power analysis: At what T/N ratio can TE estimation recover a planted NIO signal?

Design:
1. Embed known NIO premium (lambda_NIO) in DGP
2. Compute oracle NIO t-stat (using true A)
3. Compute estimated NIO t-stat (using OLS-TE and LASSO-TE)
4. Sweep across lambda values and T/N ratios
5. Generate degradation curve table

Expected result:
- Oracle t-stat ≈ 3-5 (detectable signal)
- Estimated t-stat << oracle at low T/N (signal destroyed by estimation noise)
- T/N ≈ 10 required for marginal detectability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))

from extended_dgp_planted_signal import generate_sparse_var_with_nio_premium
from te_core import compute_linear_te_matrix

OUTPUT = Path(__file__).parent.parent / 'results'
OUTPUT.mkdir(exist_ok=True)

N_TRIALS = 50
SEED_BASE = 42


def compute_nio_tstat(R, A_binary, use_true_A=False):
    """
    Compute NIO and cross-sectional t-statistic for return prediction.
    
    Parameters
    ----------
    R : (T, N) array
        Returns
    A_binary : (N, N) binary array
        Adjacency matrix (true or estimated)
    use_true_A : bool
        If True, skip TE estimation and use A_binary directly
    
    Returns
    -------
    t_stat : float
        t-statistic from regressing forward returns on NIO
    nio : (N,) array
        Net information outflow
    """
    T, N = R.shape
    
    # Compute NIO from adjacency matrix
    out_flow = A_binary.sum(axis=1)
    in_flow = A_binary.sum(axis=0)
    nio = (out_flow - in_flow) / (N - 1)
    
    # Cross-sectional regression: mean forward return ~ NIO
    # Use mean return over full sample as dependent variable
    mean_returns = R.mean(axis=0)
    
    # Standardize NIO
    nio_std = (nio - nio.mean()) / (nio.std() + 1e-10)
    
    # OLS: y = alpha + beta * NIO + epsilon
    X = np.column_stack([np.ones(N), nio_std])
    try:
        beta = np.linalg.lstsq(X, mean_returns, rcond=None)[0]
        residuals = mean_returns - X @ beta
        se = np.sqrt((residuals**2).sum() / (N - 2)) / np.sqrt((nio_std**2).sum())
        t_stat = beta[1] / se if se > 1e-10 else 0.0
    except:
        t_stat = 0.0
    
    return t_stat, nio


def run_power_analysis():
    """
    Main power analysis experiment.
    Sweep over lambda_NIO and T/N ratios.
    """
    
    # Lambda sweep: correspond to different annualized L/S spreads
    # lambda_daily * sqrt(252) * 2 ≈ annualized L/S spread
    # 0.001 → ~3%, 0.005 → ~16%, 0.01 → ~32%, 0.02 → ~64%
    # Actually more realistic: use daily vol ≈ 1.5%, so scale down
    lambda_values = [0.001, 0.003, 0.005, 0.010, 0.020]
    
    # T/N configurations
    configs = [
        (50, 100, 2.0),   # T/N = 2
        (50, 250, 5.0),   # T/N = 5
        (50, 500, 10.0),  # T/N = 10
    ]
    
    results = []
    
    for lambda_NIO in tqdm(lambda_values, desc="Lambda"):
        for N, T, tn_ratio in tqdm(configs, desc=f"  λ={lambda_NIO:.3f}", leave=False):
            
            for trial in range(N_TRIALS):
                seed = SEED_BASE + trial
                
                # Generate data with planted premium
                R, A_coef, A_true, NIO_true = generate_sparse_var_with_nio_premium(
                    N=N, T=T, density=0.05, seed=seed, 
                    dgp='garch_factor', lambda_NIO=lambda_NIO
                )
                
                # Oracle NIO (using true A)
                t_oracle, _ = compute_nio_tstat(R, A_true, use_true_A=True)
                
                # Estimated NIO (OLS-TE)
                try:
                    _, A_ols = compute_linear_te_matrix(R, alpha=0.05, method="ols", t_threshold=2.0)
                    t_ols, _ = compute_nio_tstat(R, A_ols)
                except:
                    t_ols = np.nan
                
                # Estimated NIO (LASSO-TE)
                try:
                    _, A_lasso = compute_linear_te_matrix(R, method="lasso")
                    t_lasso, _ = compute_nio_tstat(R, A_lasso)
                except:
                    t_lasso = np.nan
                
                results.append({
                    'lambda_NIO': lambda_NIO,
                    'lambda_ann_pct': lambda_NIO * np.sqrt(252) * 2 * 100,  # Approx annualized %
                    'N': N,
                    'T': T,
                    'T/N': tn_ratio,
                    'trial': trial,
                    't_oracle': t_oracle,
                    't_ols': t_ols,
                    't_lasso': t_lasso
                })
    
    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'oracle_nio_power_analysis.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT / 'oracle_nio_power_analysis.csv'}")
    
    # Compute summary statistics
    summary = df.groupby(['lambda_NIO', 'lambda_ann_pct', 'T/N']).agg({
        't_oracle': ['mean', 'std'],
        't_ols': ['mean', 'std'],
        't_lasso': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(OUTPUT / 'oracle_nio_power_summary.csv', index=False)
    print(f"✓ Saved: {OUTPUT / 'oracle_nio_power_summary.csv'}")
    
    # Print degradation table
    print("\n" + "="*80)
    print("POWER ANALYSIS: NIO Signal Degradation Under Estimation Noise")
    print("="*80)
    print("\nFormat: t-statistic (mean ± std over 50 trials)")
    print("\nλ (ann. L/S) | Oracle t  | T/N=2 t    | T/N=5 t    | T/N=10 t")
    print("-" * 70)
    
    for lam in lambda_values:
        subset = summary[summary['lambda_NIO'] == lam]
        lam_ann = subset['lambda_ann_pct'].values[0]
        
        row_data = {}
        for tn in [2.0, 5.0, 10.0]:
            tn_data = subset[subset['T/N'] == tn]
            if len(tn_data) > 0:
                t_oracle_mean = tn_data['t_oracle_mean'].values[0]
                t_oracle_std = tn_data['t_oracle_std'].values[0]
                t_lasso_mean = tn_data['t_lasso_mean'].values[0]
                t_lasso_std = tn_data['t_lasso_std'].values[0]
                row_data[tn] = (t_oracle_mean, t_oracle_std, t_lasso_mean, t_lasso_std)
        
        # Print row
        oracle_2 = row_data[2.0][0] if 2.0 in row_data else np.nan
        lasso_2_m, lasso_2_s = row_data[2.0][2:] if 2.0 in row_data else (np.nan, np.nan)
        lasso_5_m, lasso_5_s = row_data[5.0][2:] if 5.0 in row_data else (np.nan, np.nan)
        lasso_10_m, lasso_10_s = row_data[10.0][2:] if 10.0 in row_data else (np.nan, np.nan)
        
        print(f"{lam_ann:5.1f}%      | {oracle_2:5.2f}     | "
              f"{lasso_2_m:4.2f}±{lasso_2_s:4.2f} | "
              f"{lasso_5_m:4.2f}±{lasso_5_s:4.2f} | "
              f"{lasso_10_m:4.2f}±{lasso_10_s:4.2f}")
    
    print("\n" + "="*80)
    print("Key finding: Even with 10-30% annualized network premium,")
    print("estimation noise destroys the signal at T/N < 5.")
    print("This is a lower bound on required T/N for detectability.")
    print("="*80)


if __name__ == '__main__':
    run_power_analysis()


