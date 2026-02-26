"""
Compare Results Across Runs
Implements clareLab's suggestion #3: Benchmark comparison
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_run_metadata(run_dir):
    """Load metadata for a run"""
    meta_file = Path(run_dir) / 'run_metadata.json'
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)
    return None

def compute_stability_metrics(values):
    """
    Compute stability metrics across runs.
    
    Returns:
        dict with mean, std, cv, min, max
    """
    values = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(values) == 0:
        return {'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'min': np.nan, 'max': np.nan}
    
    return {
        'mean': values.mean(),
        'std': values.std(),
        'cv': values.std() / values.mean() if values.mean() != 0 else np.nan,
        'min': values.min(),
        'max': values.max(),
    }

def compare_table2(run_dirs):
    """Compare Table 2 results across runs with stability analysis"""
    comparison = []
    all_precisions = []
    
    for run_dir in run_dirs:
        run_id = Path(run_dir).name
        
        # Load metadata
        meta = load_run_metadata(run_dir)
        seed_base = meta.get('params', {}).get('seed_base', 'unknown') if meta else 'unknown'
        
        # Try to find table2 files
        table2_files = list(Path(run_dir).glob('table2_*.csv'))
        
        if table2_files:
            # Load the main one (estimated FN LASSO)
            target_file = None
            for f in table2_files:
                if 'estimated_fn_lasso' in f.name:
                    target_file = f
                    break
            
            if not target_file:
                target_file = table2_files[0]
            
            df = pd.read_csv(target_file)
            
            if 'precision_mean' in df.columns:
                avg_precision = df['precision_mean'].mean()
                all_precisions.append(avg_precision)
                avg_recall = df['recall_mean'].mean() if 'recall_mean' in df.columns else np.nan
                avg_f1 = df['f1_mean'].mean() if 'f1_mean' in df.columns else np.nan
            else:
                avg_precision = np.nan
                avg_recall = np.nan
                avg_f1 = np.nan
            
            comparison.append({
                'Run ID': run_id,
                'Seed Base': seed_base,
                'Precision': f"{avg_precision:.4f}" if not np.isnan(avg_precision) else 'N/A',
                'Recall': f"{avg_recall:.4f}" if not np.isnan(avg_recall) else 'N/A',
                'F1': f"{avg_f1:.4f}" if not np.isnan(avg_f1) else 'N/A',
            })
    
    # Compute stability
    stability = compute_stability_metrics(all_precisions)
    
    return pd.DataFrame(comparison), stability

def compare_metadata(run_dirs):
    """Compare metadata across runs"""
    comparison = []
    
    for run_dir in run_dirs:
        meta = load_run_metadata(run_dir)
        if meta:
            comparison.append({
                'Run ID': Path(run_dir).name,
                'Timestamp': meta.get('timestamp', 'unknown')[:19],
                'Git Commit': meta.get('git_commit', 'unknown')[:8],
                'Seed Base': meta.get('params', {}).get('seed_base', 'unknown'),
                'N Trials': meta.get('params', {}).get('n_trials', 'unknown'),
            })
    
    return pd.DataFrame(comparison)

def main():
    parser = argparse.ArgumentParser(description='Compare experiment runs with stability analysis')
    parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
    parser.add_argument('--base-dir', default='results', help='Base results directory')
    parser.add_argument('--table', default='table2', choices=['table2', 'table4', 'table5', 'table6', 'meta'],
                       help='Which table to compare')
    args = parser.parse_args()
    
    # Find run directories
    base_dir = Path(args.base_dir)
    run_dirs = []
    
    for run_id in args.run_ids:
        run_dir = base_dir / run_id
        if run_dir.exists():
            run_dirs.append(run_dir)
        else:
            print(f"Warning: Run '{run_id}' not found")
    
    if not run_dirs:
        print("No valid runs found.")
        return
    
    print("\n" + "="*80)
    print(f"Comparing {len(run_dirs)} runs: {args.table.upper()}")
    print("="*80 + "\n")
    
    # Compare based on table selection
    if args.table == 'meta':
        df = compare_metadata(run_dirs)
        stability = None
    elif args.table == 'table2':
        # First show metadata
        print("--- Metadata ---")
        meta_df = compare_metadata(run_dirs)
        print(meta_df.to_string(index=False))
        print("\n--- Table 2 Results ---")
        df, stability = compare_table2(run_dirs)
    else:
        print(f"Comparison for {args.table} not implemented yet.")
        return
    
    # Print comparison
    print(df.to_string(index=False))
    
    # Print stability metrics
    if stability and not np.isnan(stability['cv']):
        print("\n" + "="*80)
        print("STABILITY ANALYSIS (Precision)")
        print("="*80)
        print(f"Mean:  {stability['mean']:.4f}")
        print(f"Std:   {stability['std']:.6f}")
        print(f"CV:    {stability['cv']*100:.2f}%")
        print(f"Range: [{stability['min']:.4f}, {stability['max']:.4f}]")
        print()
        
        if stability['cv'] < 0.05:
            print("✓ STABLE: CV < 5% - results are reproducible")
        else:
            print("⚠ UNSTABLE: CV >= 5% - results vary significantly across seeds")
    print("\n" + "="*80)
    
    # Calculate differences (if numeric)
    if len(df) >= 2 and 'Avg Precision' in df.columns:
        try:
            precisions = [float(x) for x in df['Avg Precision'] if x != 'N/A']
            if len(precisions) >= 2:
                max_diff = max(precisions) - min(precisions)
                cv = (pd.Series(precisions).std() / pd.Series(precisions).mean()) * 100
                print(f"\nVariability:")
                print(f"  Max difference: {max_diff:.4f}")
                print(f"  Coefficient of Variation: {cv:.2f}%")
                
                if cv < 5:
                    print("  ✓ Results are STABLE (CV < 5%)")
                elif cv < 10:
                    print("  ⚠ Results are MODERATELY stable (CV < 10%)")
                else:
                    print("  ✗ Results are UNSTABLE (CV > 10%)")
        except:
            pass

if __name__ == '__main__':
    main()
