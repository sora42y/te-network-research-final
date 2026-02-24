"""
One-Click Experiment Runner
============================
Runs ALL experiments for the paper in correct order:
1. Simulation experiments (Table 2, Table 4)
2. Empirical analysis (Table 5, Table 7)
3. Generates all figures

Usage:
    python run_all_experiments.py [--quick]
    
    --quick: Run with reduced Monte Carlo trials (10 instead of 100) for fast testing
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)

# Check if quick mode
QUICK_MODE = "--quick" in sys.argv
TRIALS = 10 if QUICK_MODE else 100

print("=" * 80)
print("TE Network Research - One-Click Experiment Runner")
print("=" * 80)
print(f"Mode: {'QUICK (10 trials)' if QUICK_MODE else 'FULL (100 trials)'}")
print(f"Repository: {REPO_ROOT}")
print(f"Results will be saved to: {RESULTS_DIR}")
print("=" * 80)

# ============================================================================
# Helper Functions
# ============================================================================

def run_script(script_path, description, args=None):
    """Run a Python script and report status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=SRC_DIR,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False

# ============================================================================
# Experiment Pipeline
# ============================================================================

def main():
    start_time = time.time()
    results = {}
    
    # ------------------------------------------------------------------------
    # Part 1: Simulation Experiments
    # ------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("PART 1: SIMULATION EXPERIMENTS")
    print("="*80)
    
    # Experiment 1: Main Results (Table 2 - GARCH+Factor DGP)
    # This runs all T/N ratios, all methods (OLS, LASSO, Raw, Factor-neutral)
    print("\n📊 Experiment 1: Main Results (Table 2)")
    print("   - GARCH(1,1) + t(5) innovations")
    print("   - Common factor structure (K=3)")
    print(f"   - Monte Carlo trials: {TRIALS}")
    print("   - T/N ratios: 2, 2.5, 5, 10")
    print("   - Methods: OLS-TE (Raw), LASSO-TE (Raw), Factor-neutral")
    
    results['main'] = run_script(
        SRC_DIR / "run_factor_neutral_sim.py",
        "Main Results (Table 2)",
        ["--trials", str(TRIALS)]
    )
    
    # Experiment 2: Oracle vs Estimated (Table 4 - Factor-Neutral)
    # Tests whether factor-neutral preprocessing helps
    print("\n📊 Experiment 2: Oracle vs Estimated Factor-Neutral (Table 4)")
    print("   - Oracle: True factors known")
    print("   - Estimated: PCA-estimated factors")
    print(f"   - Monte Carlo trials: {TRIALS}")
    
    results['oracle'] = run_script(
        SRC_DIR / "all_experiments_v2.py",
        "Oracle vs Estimated (Table 4)",
        ["--trials", str(TRIALS), "--experiments", "3"]  # Exp #3 only
    )
    
    # ------------------------------------------------------------------------
    # Part 2: Empirical Analysis
    # ------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("PART 2: EMPIRICAL ANALYSIS")
    print("="*80)
    
    # Check if empirical data exists
    empirical_data = DATA_DIR / "empirical" / "te_features_weekly.csv"
    if not empirical_data.exists():
        print(f"\n⚠️  WARNING: Empirical data not found at {empirical_data}")
        print("   Skipping empirical analysis.")
        print("   To run empirical tests, place data at:")
        print(f"   {empirical_data}")
        results['empirical'] = False
    else:
        # Experiment 3: Portfolio Sort (Table 5)
        print("\n📊 Experiment 3: Portfolio Sort on NIO (Table 5)")
        print("   - Data: S&P 500 stocks, 2021-2026")
        print("   - Method: Factor-neutral OLS-TE")
        print("   - Quintile portfolios + binary split")
        
        results['portfolio'] = run_script(
            SRC_DIR / "empirical_portfolio_sort.py",
            "Portfolio Sort (Table 5)"
        )
        
        # Experiment 4: Power Analysis (Table 6)
        print("\n📊 Experiment 4: Oracle NIO Power Analysis (Table 6)")
        print("   - Embed known premium in simulated data")
        print("   - Test if estimation recovers signal")
        print(f"   - Monte Carlo trials: {TRIALS}")
        
        results['power'] = run_script(
            SRC_DIR / "oracle_nio_power.py",
            "Oracle NIO Power Analysis (Table 6)",
            ["--trials", str(TRIALS)]
        )
    
    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{exp_name:20s} {status}")
    
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    
    if all(results.values()):
        print("\n🎉 All experiments completed successfully!")
        print(f"\nResults saved to: {RESULTS_DIR}")
        print("\nNext steps:")
        print("  1. Check results/*.csv for numerical output")
        print("  2. Check paper_assets/*.png for figures")
        print("  3. Compile paper: cd paper && pdflatex main.tex")
    else:
        print("\n⚠️  Some experiments failed. Check logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
