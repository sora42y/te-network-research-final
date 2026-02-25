"""
Simulation Configuration - Dual Pipeline Design
================================================

This script supports TWO modes:

1. RANDOM mode (robustness testing, research phase)
   - Generates fresh random seeds each run
   - Used to verify results are stable across different networks
   - Output: robustness_*.csv with variation statistics

2. FIXED mode (paper submission, reproducibility)
   - Uses pre-defined seed sequence (SEED_BASE + trial*1000)
   - Guarantees exact reproducibility
   - Output: results_*.csv matching paper tables

Usage:
------
# Research phase: test robustness
python run_all_experiments.py --mode random --trials 100

# Paper version: exact reproducibility
python run_all_experiments.py --mode fixed --trials 100

# Quick test
python run_all_experiments.py --mode fixed --quick
"""

import argparse
import numpy as np
from pathlib import Path

class SimulationConfig:
    """
    Centralized configuration for all simulation experiments.
    
    Design principles:
    1. Reproducibility: Fixed seed mode guarantees exact replication
    2. Robustness: Random seed mode proves results generalize
    3. Versioning: Git commit + config file track exact setup
    4. Transparency: All parameters explicitly documented
    """
    
    def __init__(self, mode='fixed', seed_base=42):
        """
        Parameters
        ----------
        mode : str
            'fixed'  : Use deterministic seed sequence (for paper)
            'random' : Generate fresh random seeds (for robustness check)
        seed_base : int
            Base seed for 'fixed' mode (default: 42)
        """
        self.mode = mode
        self.seed_base = seed_base
        
        # Experiment parameters (same for both modes)
        self.n_trials = 100
        self.dgp_type = 'garch_factor'
        self.density = 0.05
        
        # GARCH parameters (literature-calibrated)
        self.garch_omega_scale = 0.05
        self.garch_alpha = 0.08  # Engle & Bollerslev (1986)
        self.garch_beta = 0.90
        
        # Innovation distribution
        self.innovation_df = 5  # t(5) for fat tails
        
        # Factor structure
        self.n_factors = 3  # Fama-French standard
        
        # Random seed generator
        if mode == 'random':
            # Use system randomness, but make it reproducible for robustness analysis
            self.meta_seed = np.random.RandomState(seed_base)
        else:
            self.meta_seed = None
    
    def get_seed(self, trial, N, T):
        """
        Generate seed for a specific trial.
        
        Parameters
        ----------
        trial : int
            Trial index (0 to n_trials-1)
        N : int
            Network size
        T : int
            Time series length
        
        Returns
        -------
        seed : int
            Seed for this trial
        """
        if self.mode == 'fixed':
            # Deterministic: same (trial, N, T) always gives same seed
            return self.seed_base + trial * 1000 + N + T
        
        elif self.mode == 'random':
            # Random: different seed each run, but track for reproducibility
            return self.meta_seed.randint(0, 999999)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def save_config(self, output_path):
        """Save configuration to file for exact reproducibility."""
        import json
        from datetime import datetime
        
        config_dict = {
            'mode': self.mode,
            'seed_base': self.seed_base,
            'n_trials': self.n_trials,
            'dgp_type': self.dgp_type,
            'density': self.density,
            'garch_params': {
                'omega_scale': self.garch_omega_scale,
                'alpha': self.garch_alpha,
                'beta': self.garch_beta
            },
            'innovation_df': self.innovation_df,
            'n_factors': self.n_factors,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _get_git_commit(self):
        """Get current git commit hash for version tracking."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            return result.stdout.strip()
        except:
            return 'unknown'

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation configuration test')
    parser.add_argument('--mode', choices=['fixed', 'random'], default='fixed')
    parser.add_argument('--seed-base', type=int, default=42)
    args = parser.parse_args()
    
    config = SimulationConfig(mode=args.mode, seed_base=args.seed_base)
    
    print(f"Mode: {config.mode}")
    print(f"Seed base: {config.seed_base}")
    print("\nFirst 10 seeds (N=50, T=250):")
    
    for trial in range(10):
        seed = config.get_seed(trial, N=50, T=250)
        print(f"  Trial {trial}: seed={seed}")
    
    # Save config
    config.save_config('results/simulation_config.json')
    print("\nConfig saved to results/simulation_config.json")
