"""
Empirical Portfolio Sort Analysis
==================================
Replicates Table 5 from the paper:
- Portfolio sort on factor-neutral OLS-TE NIO
- N ≈ 100 stocks, T = 60 days
- Time period: 2021-2026
- Quintile portfolios + binary split

Input:
    data/empirical/te_features_weekly.csv
    
Output:
    results/table5_portfolio_sort.csv
    results/table5_portfolio_sort.txt (formatted table)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "data" / "empirical" / "te_features_weekly.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "table5_portfolio_sort.csv"
OUTPUT_TXT = REPO_ROOT / "results" / "table5_portfolio_sort.txt"

# Time periods for sub-sample analysis
FULL_START = "2021-01-01"
FULL_END = "2026-12-31"
SUB1_START = "2021-01-01"
SUB1_END = "2023-12-31"
SUB2_START = "2023-01-01"
SUB2_END = "2026-12-31"

# ============================================================================
# Helper Functions
# ============================================================================

def portfolio_sort(data, signal_col, ret_col, n_groups=5):
    """
    Standard portfolio sort: form quintile portfolios based on signal,
    compute equal-weighted average returns.
    
    Args:
        data: DataFrame with formation_date, signal, and returns
        signal_col: Column name for sorting characteristic
        ret_col: Column name for forward returns
        n_groups: Number of portfolios (default 5 for quintiles)
        
    Returns:
        DataFrame with date, quintile, and returns
    """
    results = []
    
    for date, group in data.groupby("formation_date"):
        valid = group.dropna(subset=[signal_col, ret_col])
        
        # Need at least n_groups * 5 stocks
        if len(valid) < n_groups * 5:
            continue
        
        try:
            # Assign quintiles
            valid["quintile"] = pd.qcut(
                valid[signal_col], 
                n_groups, 
                labels=False, 
                duplicates="drop"
            ) + 1
        except ValueError:
            # Skip if can't form quintiles (e.g., too many ties)
            continue
        
        # Compute equal-weighted returns for each quintile
        for q in range(1, n_groups + 1):
            q_stocks = valid[valid["quintile"] == q]
            if len(q_stocks) > 0:
                results.append({
                    "date": date,
                    "quintile": q,
                    "ret": q_stocks[ret_col].mean(),
                    "n_stocks": len(q_stocks)
                })
    
    return pd.DataFrame(results)


def compute_portfolio_stats(port_returns):
    """
    Compute annualized return and t-stat for a portfolio time series.
    
    Args:
        port_returns: Series of weekly returns
        
    Returns:
        dict with ann_ret and t_stat
    """
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    n = len(port_returns)
    
    # Annualize (52 weeks per year)
    ann_ret = mean_ret * 52
    t_stat = (mean_ret / (std_ret / np.sqrt(n))) if std_ret > 0 else 0
    
    return {
        "ann_ret": ann_ret,
        "t_stat": t_stat
    }


def format_table5(results_dict):
    """
    Format results into LaTeX-style table matching Table 5.
    
    Args:
        results_dict: dict with keys 'full', 'sub1', 'sub2'
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("="*80)
    lines.append("Table 5: Portfolio Sort - Factor-Neutral OLS-TE NIO")
    lines.append("(N ~= 100, T = 60)")
    lines.append("="*80)
    lines.append("")
    lines.append(f"{'Quintile':<15} {'Full (2021-2026)':<25} {'Sub 1 (2021-2023)':<25} {'Sub 2 (2023-2026)':<25}")
    lines.append(f"{'':15} {'Ann.Ret.':<12} {'t':>12} {'Ann.Ret.':<12} {'t':>12} {'Ann.Ret.':<12} {'t':>12}")
    lines.append("-"*80)
    
    for q in range(1, 6):
        q_label = f"Q{q} " + ("(Low NIO)" if q == 1 else "(High NIO)" if q == 5 else "")
        
        full = results_dict['full'][results_dict['full']['quintile'] == q].iloc[0]
        sub1 = results_dict['sub1'][results_dict['sub1']['quintile'] == q].iloc[0]
        sub2 = results_dict['sub2'][results_dict['sub2']['quintile'] == q].iloc[0]
        
        lines.append(
            f"{q_label:<15} "
            f"{full['ann_ret']:>10.2%}  {full['t_stat']:>12.2f} "
            f"{sub1['ann_ret']:>10.2%}  {sub1['t_stat']:>12.2f} "
            f"{sub2['ann_ret']:>10.2%}  {sub2['t_stat']:>12.2f}"
        )
    
    lines.append("-"*80)
    
    # Long-short
    full_ls = results_dict['full'][results_dict['full']['quintile'] == 5].iloc[0]['ann_ret'] - \
              results_dict['full'][results_dict['full']['quintile'] == 1].iloc[0]['ann_ret']
    full_ls_t = results_dict['full_ls_t']
    
    sub1_ls = results_dict['sub1'][results_dict['sub1']['quintile'] == 5].iloc[0]['ann_ret'] - \
              results_dict['sub1'][results_dict['sub1']['quintile'] == 1].iloc[0]['ann_ret']
    sub1_ls_t = results_dict['sub1_ls_t']
    
    sub2_ls = results_dict['sub2'][results_dict['sub2']['quintile'] == 5].iloc[0]['ann_ret'] - \
              results_dict['sub2'][results_dict['sub2']['quintile'] == 1].iloc[0]['ann_ret']
    sub2_ls_t = results_dict['sub2_ls_t']
    
    lines.append(
        f"{'L/S (Q5-Q1)':<15} "
        f"{full_ls:>10.2%}  {full_ls_t:>12.2f} "
        f"{sub1_ls:>10.2%}  {sub1_ls_t:>12.2f} "
        f"{sub2_ls:>10.2%}  {sub2_ls_t:>12.2f}"
    )
    
    lines.append("="*80)
    lines.append("")
    lines.append("Note: Equal-weighted quintile portfolios sorted on NIO.")
    lines.append("      5-day forward factor-adjusted returns.")
    lines.append("      No quintile produces significant returns.")
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("Empirical Portfolio Sort Analysis")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["formation_date"])
    
    print(f"Total observations: {len(df):,}")
    print(f"Date range: {df['formation_date'].min()} to {df['formation_date'].max()}")
    print(f"Unique stocks: {df['ticker'].nunique()}")
    
    # Filter to paper sample period
    df_full = df[(df['formation_date'] >= FULL_START) & (df['formation_date'] <= FULL_END)]
    df_sub1 = df[(df['formation_date'] >= SUB1_START) & (df['formation_date'] <= SUB1_END)]
    df_sub2 = df[(df['formation_date'] >= SUB2_START) & (df['formation_date'] <= SUB2_END)]
    
    print(f"\nFull sample: {len(df_full):,} obs, {df_full['formation_date'].nunique()} weeks")
    print(f"Sub-period 1: {len(df_sub1):,} obs, {df_sub1['formation_date'].nunique()} weeks")
    print(f"Sub-period 2: {len(df_sub2):,} obs, {df_sub2['formation_date'].nunique()} weeks")
    
    # Run portfolio sorts
    print("\n" + "="*80)
    print("Running Portfolio Sorts")
    print("="*80)
    
    ps_full = portfolio_sort(df_full, "nio", "next_week_ret", n_groups=5)
    ps_sub1 = portfolio_sort(df_sub1, "nio", "next_week_ret", n_groups=5)
    ps_sub2 = portfolio_sort(df_sub2, "nio", "next_week_ret", n_groups=5)
    
    # Compute statistics for each quintile
    results_full = []
    results_sub1 = []
    results_sub2 = []
    
    for q in range(1, 6):
        # Full sample
        q_rets = ps_full[ps_full['quintile'] == q]['ret']
        stats_full = compute_portfolio_stats(q_rets)
        results_full.append({
            'quintile': q,
            'ann_ret': stats_full['ann_ret'],
            't_stat': stats_full['t_stat'],
            'n_weeks': len(q_rets)
        })
        
        # Sub-period 1
        q_rets = ps_sub1[ps_sub1['quintile'] == q]['ret']
        stats_sub1 = compute_portfolio_stats(q_rets)
        results_sub1.append({
            'quintile': q,
            'ann_ret': stats_sub1['ann_ret'],
            't_stat': stats_sub1['t_stat'],
            'n_weeks': len(q_rets)
        })
        
        # Sub-period 2
        q_rets = ps_sub2[ps_sub2['quintile'] == q]['ret']
        stats_sub2 = compute_portfolio_stats(q_rets)
        results_sub2.append({
            'quintile': q,
            'ann_ret': stats_sub2['ann_ret'],
            't_stat': stats_sub2['t_stat'],
            'n_weeks': len(q_rets)
        })
    
    results_full = pd.DataFrame(results_full)
    results_sub1 = pd.DataFrame(results_sub1)
    results_sub2 = pd.DataFrame(results_sub2)
    
    # Compute long-short t-stats
    q5_rets_full = ps_full[ps_full['quintile'] == 5]['ret'].values
    q1_rets_full = ps_full[ps_full['quintile'] == 1]['ret'].values
    ls_full = q5_rets_full - q1_rets_full
    ls_full_t = ls_full.mean() / (ls_full.std() / np.sqrt(len(ls_full)))
    
    q5_rets_sub1 = ps_sub1[ps_sub1['quintile'] == 5]['ret'].values
    q1_rets_sub1 = ps_sub1[ps_sub1['quintile'] == 1]['ret'].values
    ls_sub1 = q5_rets_sub1 - q1_rets_sub1
    ls_sub1_t = ls_sub1.mean() / (ls_sub1.std() / np.sqrt(len(ls_sub1)))
    
    q5_rets_sub2 = ps_sub2[ps_sub2['quintile'] == 5]['ret'].values
    q1_rets_sub2 = ps_sub2[ps_sub2['quintile'] == 1]['ret'].values
    ls_sub2 = q5_rets_sub2 - q1_rets_sub2
    ls_sub2_t = ls_sub2.mean() / (ls_sub2.std() / np.sqrt(len(ls_sub2)))
    
    # Package results
    results_dict = {
        'full': results_full,
        'sub1': results_sub1,
        'sub2': results_sub2,
        'full_ls_t': ls_full_t,
        'sub1_ls_t': ls_sub1_t,
        'sub2_ls_t': ls_sub2_t
    }
    
    # Format and save
    table_text = format_table5(results_dict)
    print("\n" + table_text)
    
    # Save to files
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    all_results = pd.concat([
        results_full.assign(period='full'),
        results_sub1.assign(period='sub1'),
        results_sub2.assign(period='sub2')
    ])
    all_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Saved CSV: {OUTPUT_CSV}")
    
    # Save formatted table
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(table_text)
    print(f"[SUCCESS] Saved table: {OUTPUT_TXT}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
