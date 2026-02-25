"""
Compare Results Across Runs
Implements clareLab's suggestion #4: Horizontal comparison
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate

def load_run_metadata(run_dir):
    """Load metadata for a run"""
    meta_file = Path(run_dir) / 'run_metadata.json'
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)
    return None

def compare_table2(run_dirs):
    """Compare Table 2 results across runs"""
    comparison = []
    
    for run_dir in run_dirs:
        run_id = Path(run_dir).name
        
        # Try to find table2 files
        table2_files = list(Path(run_dir).glob('table2_*.csv'))
        
        if table2_files:
            # Load the main one (OLS)
            df = pd.read_csv(table2_files[0])
            
            if 'precision_mean' in df.columns:
                avg_precision = df['precision_mean'].mean()
                avg_recall = df['recall_mean'].mean() if 'recall_mean' in df.columns else 'N/A'
                avg_f1 = df['f1_mean'].mean() if 'f1_mean' in df.columns else 'N/A'
            else:
                avg_precision = 'N/A'
                avg_recall = 'N/A'
                avg_f1 = 'N/A'
            
            comparison.append({
                'Run ID': run_id,
                'Avg Precision': f"{avg_precision:.4f}" if isinstance(avg_precision, float) else avg_precision,
                'Avg Recall': f"{avg_recall:.4f}" if isinstance(avg_recall, float) else avg_recall,
                'Avg F1': f"{avg_f1:.4f}" if isinstance(avg_f1, float) else avg_f1,
            })
    
    return pd.DataFrame(comparison)

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
    parser = argparse.ArgumentParser(description='Compare experiment runs')
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
    elif args.table == 'table2':
        # First show metadata
        print("--- Metadata ---")
        meta_df = compare_metadata(run_dirs)
        print(tabulate(meta_df, headers='keys', tablefmt='grid', showindex=False))
        print("\n--- Table 2 Results ---")
        df = compare_table2(run_dirs)
    else:
        print(f"Comparison for {args.table} not implemented yet.")
        return
    
    # Print comparison
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
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
