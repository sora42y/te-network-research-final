"""
Oracle NIO Power Analysis
=========================
Replicates Table 6 from the paper:
- Embed known premium in simulated data
- Test if estimation recovers signal
- Compare Oracle vs Estimated (LASSO) vs Estimated (OLS)

This answers: "If a NIO premium exists, can we detect it?"

Output:
    results/table6_oracle_nio_power.csv
    results/table6_oracle_nio_power.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extended_dgp_planted_signal import generate_sparse_var_with_nio_premium
from te_core import compute_linear_te_matrix

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_CSV = REPO_ROOT / "results" / "table6_oracle_nio_power.csv"
OUTPUT_TXT = REPO_ROOT / "results" / "table6_oracle_nio_power.txt"

# Simulation parameters
N = 30
T_N_RATIOS = [2, 5, 10]
PREMIUMS = [0.03, 0.05, 0.10, 0.20, 0.30]  # Annualized L/S spreads
TRIALS = 50  # Default, can override with --trials

# ============================================================================
# Helper Functions
# ============================================================================

def compute_nio(te_matrix):
    """
    Compute Net Information Outflow from TE matrix.
    
    Args:
        te_matrix: (N, N) array where [i,j] = TE from i to j
        
    Returns:
        (N,) array of NIO values
    """
    N = te_matrix.shape[0]
    nio = np.zeros(N)
    
    for i in range(N):
        te_out = te_matrix[i, :].sum() - te_matrix[i, i]  # Exclude diagonal
        te_in = te_matrix[:, i].sum() - te_matrix[i, i]
        nio[i] = (te_out - te_in) / (N - 1)
    
    return nio


def cross_sectional_tstat(returns, nio):
    """
    Compute t-statistic from cross-sectional regression: returns ~ NIO.
    
    Args:
        returns: (T, N) array of returns
        nio: (N,) array of NIO values
        
    Returns:
        t-statistic
    """
    # Mean returns across time for each stock
    mean_rets = returns.mean(axis=0)
    
    # OLS regression: mean_ret_i = alpha + beta * nio_i + error
    # Using numpy's lstsq for speed
    X = np.column_stack([np.ones(len(nio)), nio])
    beta_hat = np.linalg.lstsq(X, mean_rets, rcond=None)[0]
    
    # Residuals and standard error
    fitted = X @ beta_hat
    resid = mean_rets - fitted
    n = len(nio)
    
    # Standard error of beta (NIO coefficient)
    resid_var = (resid ** 2).sum() / (n - 2)
    X_var = ((X[:, 1] - X[:, 1].mean()) ** 2).sum()
    se_beta = np.sqrt(resid_var / X_var)
    
    # t-stat
    t_stat = beta_hat[1] / se_beta if se_beta > 0 else 0
    
    return t_stat


# ============================================================================
# Main Experiment
# ============================================================================

def run_power_analysis(trials=50):
    """
    Run power analysis across all parameter combinations.
    
    Args:
        trials: Number of Monte Carlo trials per cell
        
    Returns:
        DataFrame with results
    """
    results = []
    
    total_runs = len(T_N_RATIOS) * len(PREMIUMS) * trials
    run_count = 0
    
    for t_n_ratio in T_N_RATIOS:
        T = int(N * t_n_ratio)
        
        for premium in PREMIUMS:
            print(f"\nT/N={t_n_ratio}, Premium={premium:.0%}")
            print("-" * 60)
            
            trial_tstats_oracle = []
            trial_tstats_lasso = []
            trial_tstats_ols = []
            
            for trial in range(trials):
                run_count += 1
                if trial % 10 == 0:
                    print(f"  Trial {trial+1}/{trials} ({run_count}/{total_runs} total)")
                
                # Generate data with planted premium
                R, A_true_coef, A_true_binary, NIO_true_std = generate_sparse_var_with_nio_premium(
                    N=N,
                    T=T,
                    density=0.1,
                    seed=trial,
                    dgp='garch_factor',
                    lambda_NIO=premium / 252  # Convert annualized to daily
                )
                
                # Compute true NIO (Oracle)
                nio_oracle = compute_nio(A_true_binary.astype(float))
                
                # Estimate TE networks
                te_lasso, adj_lasso = compute_linear_te_matrix(R, method="lasso")
                te_ols, adj_ols = compute_linear_te_matrix(R, method="ols", t_threshold=2.0)
                
                # Compute estimated NIOs
                nio_lasso = compute_nio(adj_lasso.astype(float))
                nio_ols = compute_nio(adj_ols.astype(float))
                
                # Cross-sectional t-stats
                t_oracle = cross_sectional_tstat(R, nio_oracle)
                t_lasso = cross_sectional_tstat(R, nio_lasso)
                t_ols = cross_sectional_tstat(R, nio_ols)
                
                trial_tstats_oracle.append(t_oracle)
                trial_tstats_lasso.append(t_lasso)
                trial_tstats_ols.append(t_ols)
            
            # Aggregate across trials
            results.append({
                'T_N_ratio': t_n_ratio,
                'premium': premium,
                't_oracle': np.mean(trial_tstats_oracle),
                't_lasso': np.mean(trial_tstats_lasso),
                't_ols': np.mean(trial_tstats_ols),
                'power_oracle': np.mean([abs(t) > 1.96 for t in trial_tstats_oracle]),
                'power_lasso': np.mean([abs(t) > 1.96 for t in trial_tstats_lasso]),
                'power_ols': np.mean([abs(t) > 1.96 for t in trial_tstats_ols]),
                'trials': trials
            })
    
    return pd.DataFrame(results)


def format_table6(df):
    """
    Format results into LaTeX-style table matching Table 6.
    
    Args:
        df: Results DataFrame
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("="*100)
    lines.append("Table 6: Power Analysis - NIO Signal Degradation Under Estimation Noise")
    lines.append("="*100)
    lines.append("")
    lines.append(f"{'Premium':<15} {'T/N=2':<30} {'T/N=5':<30} {'T/N=10':<30}")
    lines.append(f"{'(Ann. L/S)':<15} {'Oracle':>9} {'LASSO':>9} {'OLS':>9} {'Oracle':>9} {'LASSO':>9} {'OLS':>9} {'Oracle':>9} {'LASSO':>9} {'OLS':>9}")
    lines.append("-"*100)
    
    for premium in PREMIUMS:
        premium_str = f"{premium:.0%}"
        
        row_str = f"{premium_str:<15}"
        
        for t_n in T_N_RATIOS:
            row_data = df[(df['T_N_ratio'] == t_n) & (df['premium'] == premium)].iloc[0]
            
            row_str += f" {row_data['t_oracle']:>9.2f}"
            row_str += f" {row_data['t_lasso']:>9.2f}"
            row_str += f" {row_data['t_ols']:>9.2f}"
        
        lines.append(row_str)
    
    lines.append("="*100)
    lines.append("")
    lines.append("Note: Reported t-statistics from cross-sectional regression: mean return ~ NIO.")
    lines.append("      Oracle uses true adjacency matrix; estimated values use LASSO-TE.")
    lines.append("      Even with 10% annualized premium (oracle t~5-6), estimated t~0.7 at T/N=5.")
    lines.append("      Signal emerges only at extreme premia (30%+) or very high T/N (>10).")
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50, help='Number of Monte Carlo trials')
    args = parser.parse_args()
    
    global TRIALS
    TRIALS = args.trials
    
    print("="*100)
    print("Oracle NIO Power Analysis")
    print("="*100)
    print(f"Monte Carlo trials: {TRIALS}")
    print(f"N: {N}")
    print(f"T/N ratios: {T_N_RATIOS}")
    print(f"Premiums: {[f'{p:.0%}' for p in PREMIUMS]}")
    print("="*100)
    
    # Run analysis
    results = run_power_analysis(trials=TRIALS)
    
    # Format table
    table_text = format_table6(results)
    print("\n" + table_text)
    
    # Save outputs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Saved CSV: {OUTPUT_CSV}")
    
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(table_text)
    print(f"[SUCCESS] Saved table: {OUTPUT_TXT}")
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)


if __name__ == "__main__":
    main()


