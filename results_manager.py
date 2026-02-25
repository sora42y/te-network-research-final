"""
Results Version Control System
Implements clareLab's suggestion #3 & #4:
- Separate small processes (each table can run independently)
- Version & benchmark results (no overwriting, enable comparison)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

class ResultsManager:
    """
    Manage versioned experiment results.
    
    Design:
    - Each run gets a unique ID (timestamp + short hash)
    - Results stored in results/<run_id>/
    - Old results preserved (no overwriting)
    - Metadata tracked for each run
    - Easy horizontal comparison
    """
    
    def __init__(self, base_dir='results', run_id=None):
        """
        Parameters
        ----------
        base_dir : str
            Base directory for all results
        run_id : str, optional
            Manual run ID (e.g., 'paper_final', 'robustness_check')
            If None, auto-generate timestamp-based ID
        """
        self.base_dir = Path(base_dir)
        
        if run_id is None:
            # Auto-generate: YYYYMMDD_HHMMSS_<git_short_hash>
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            git_hash = self._get_git_hash()[:8]
            self.run_id = f"{timestamp}_{git_hash}"
        else:
            self.run_id = run_id
        
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.run_dir}")
        print(f"Run ID: {self.run_id}")
    
    def save_table(self, table_name, df, description=''):
        """
        Save a single table with metadata.
        
        Parameters
        ----------
        table_name : str
            e.g., 'table2', 'table4', 'table5'
        df : pd.DataFrame or str (path)
            Data to save
        description : str
            Optional description
        """
        import pandas as pd
        
        # Save data
        output_path = self.run_dir / f"{table_name}.csv"
        if isinstance(df, pd.DataFrame):
            df.to_csv(output_path, index=False)
        elif isinstance(df, (str, Path)):
            shutil.copy(df, output_path)
        
        # Save metadata
        meta_path = self.run_dir / f"{table_name}_meta.json"
        metadata = {
            'table': table_name,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'rows': len(df) if isinstance(df, pd.DataFrame) else 'unknown',
            'columns': list(df.columns) if isinstance(df, pd.DataFrame) else 'unknown'
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved: {table_name}.csv")
    
    def save_run_metadata(self, params):
        """
        Save overall run metadata.
        
        Parameters
        ----------
        params : dict
            Experiment parameters (seed_base, n_trials, etc.)
        """
        metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_hash(),
            'git_branch': self._get_git_branch(),
            'params': params,
            'python_version': self._get_python_version(),
        }
        
        meta_path = self.run_dir / 'run_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a human-readable README
        readme_path = self.run_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"Experiment Run: {self.run_id}\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Git commit: {metadata['git_commit']}\n")
            f.write(f"Git branch: {metadata['git_branch']}\n\n")
            f.write("Parameters:\n")
            for key, val in params.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"\nResults saved in: {self.run_dir}\n")
        
        print(f"  ✓ Metadata saved: run_metadata.json")
    
    def _get_git_hash(self):
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            return result.stdout.strip()
        except:
            return 'unknown'
    
    def _get_git_branch(self):
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            return result.stdout.strip()
        except:
            return 'unknown'
    
    def _get_python_version(self):
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    @classmethod
    def list_runs(cls, base_dir='results'):
        """List all available runs"""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            print("No results directory found.")
            return []
        
        runs = []
        for run_dir in sorted(base_dir.iterdir()):
            if run_dir.is_dir():
                meta_file = run_dir / 'run_metadata.json'
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    runs.append({
                        'run_id': run_dir.name,
                        'timestamp': meta.get('timestamp', 'unknown'),
                        'git_commit': meta.get('git_commit', 'unknown')[:8],
                        'params': meta.get('params', {})
                    })
        
        return runs
    
    @classmethod
    def compare_runs(cls, run_ids, base_dir='results', metric='precision_mean'):
        """
        Compare results across multiple runs.
        
        Parameters
        ----------
        run_ids : list of str
            Run IDs to compare
        metric : str
            Metric to compare (e.g., 'precision_mean')
        """
        import pandas as pd
        
        base_dir = Path(base_dir)
        comparison = {}
        
        for run_id in run_ids:
            run_dir = base_dir / run_id
            
            # Load Table 2 results
            table2_path = run_dir / 'table2.csv'
            if table2_path.exists():
                df = pd.read_csv(table2_path)
                if metric in df.columns:
                    comparison[run_id] = df[metric].mean()
                else:
                    comparison[run_id] = 'N/A'
        
        # Print comparison
        print("\n" + "=" * 60)
        print(f"Comparison: {metric}")
        print("=" * 60)
        for run_id, value in comparison.items():
            print(f"{run_id:30s}: {value}")
        print("=" * 60)
        
        return comparison


# Example usage
if __name__ == '__main__':
    # Run 1: Paper baseline
    mgr = ResultsManager(run_id='paper_baseline')
    
    params = {
        'seed_base': 42,
        'n_trials': 100,
        'garch_alpha': 0.08,
        'garch_beta': 0.90
    }
    
    mgr.save_run_metadata(params)
    
    # Simulate saving tables
    import pandas as pd
    fake_data = pd.DataFrame({
        'N': [30, 50, 100],
        'T': [60, 120, 250],
        'precision_mean': [0.281, 0.273, 0.285]
    })
    
    mgr.save_table('table2', fake_data, description='Main simulation results')
    
    print("\n--- List all runs ---")
    runs = ResultsManager.list_runs()
    for run in runs:
        print(f"{run['run_id']}: {run['timestamp']} (commit {run['git_commit']})")
    
    print("\n--- Compare runs ---")
    ResultsManager.compare_runs(['paper_baseline', 'paper_baseline'])
