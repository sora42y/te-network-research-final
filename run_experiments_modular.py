"""
Modular Experiment Runner
Implements clareLab's suggestion #3: Break into small processes

Each table can be run independently for easier auditing.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from results_manager import ResultsManager

def run_table2(mgr, args):
    """Run Table 2: Main simulation results"""
    print("\n" + "="*60)
    print("Running Table 2: Main Simulation Results (GARCH+Factor DGP)")
    print("="*60)
    
    from run_factor_neutral_sim import main as run_sim
    
    # Run experiment
    run_sim(trials=args.trials, seed_base=args.seed_base)
    
    # Copy results to versioned directory
    result_files = [
        'results/table2_raw_ols_garch_factor.csv',
        'results/table2_raw_lasso_garch_factor.csv',
        'results/table2_estimated_fn_ols_garch_factor.csv',
        'results/table2_estimated_fn_lasso_garch_factor.csv',
    ]
    
    for f in result_files:
        if Path(f).exists():
            mgr.save_table(Path(f).stem, f, description='Table 2 results')

def run_table4(mgr, args):
    """Run Table 4: Oracle vs Estimated"""
    print("\n" + "="*60)
    print("Running Table 4: Oracle vs Estimated Factor-Neutral")
    print("="*60)
    
    from all_experiments_v2 import run_oracle_vs_estimated
    
    run_oracle_vs_estimated()
    
    result_file = 'results/oracle_vs_estimated.csv'
    if Path(result_file).exists():
        mgr.save_table('table4', result_file, description='Oracle vs Estimated')

def run_table5(mgr, args):
    """Run Table 5: Empirical portfolio sort"""
    print("\n" + "="*60)
    print("Running Table 5: Empirical Portfolio Sort (NIO)")
    print("="*60)
    
    from empirical_portfolio_sort import main as run_sort
    
    run_sort()
    
    result_files = [
        'results/table5_portfolio_sort.csv',
        'results/table5_portfolio_sort.txt'
    ]
    
    for f in result_files:
        if Path(f).exists():
            mgr.save_table(Path(f).stem, f, description='Table 5 empirical results')

def run_table6(mgr, args):
    """Run Table 6: Oracle NIO power analysis"""
    print("\n" + "="*60)
    print("Running Table 6: Oracle NIO Power Analysis")
    print("="*60)
    
    from oracle_nio_power import main as run_power
    
    run_power(trials=args.trials)
    
    result_files = [
        'results/table6_oracle_nio_power.csv',
        'results/table6_oracle_nio_power.txt'
    ]
    
    for f in result_files:
        if Path(f).exists():
            mgr.save_table(Path(f).stem, f, description='Table 6 power analysis')

def main():
    parser = argparse.ArgumentParser(description='Modular Experiment Runner')
    
    # Run selection
    parser.add_argument('--tables', nargs='+', default=['all'],
                       choices=['all', 'table2', 'table4', 'table5', 'table6'],
                       help='Which tables to run')
    
    # Run identification
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID for versioning (auto-generated if not specified)')
    
    # Experiment parameters
    parser.add_argument('--seed-base', type=int, default=42,
                       help='Base seed for reproducibility')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (10 trials only)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.trials = 10
        print("Quick mode: 10 trials only")
    
    # Create results manager
    mgr = ResultsManager(run_id=args.run_id)
    
    # Save run metadata
    params = {
        'seed_base': args.seed_base,
        'n_trials': args.trials,
        'tables': args.tables,
        'quick_mode': args.quick
    }
    mgr.save_run_metadata(params)
    
    # Run selected tables
    table_funcs = {
        'table2': run_table2,
        'table4': run_table4,
        'table5': run_table5,
        'table6': run_table6
    }
    
    if 'all' in args.tables:
        tables_to_run = ['table2', 'table4', 'table5', 'table6']
    else:
        tables_to_run = args.tables
    
    for table in tables_to_run:
        if table in table_funcs:
            try:
                table_funcs[table](mgr, args)
            except Exception as e:
                print(f"ERROR running {table}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"All results saved to: {mgr.run_dir}")
    print("="*60)
    
    # Print summary
    print("\nRun Summary:")
    print(f"  Run ID: {mgr.run_id}")
    print(f"  Seed base: {args.seed_base}")
    print(f"  Trials: {args.trials}")
    print(f"  Tables: {', '.join(tables_to_run)}")
    
    return mgr.run_dir

if __name__ == '__main__':
    main()
