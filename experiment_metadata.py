"""
Simulation Experiment Metadata - Lineage Tracking
Inspired by production data pipeline design
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
import subprocess

class ExperimentMetadata:
    """
    Track complete lineage of simulation experiments.
    
    Design principles (borrowed from production pipelines):
    1. Every run has unique fingerprint
    2. All inputs (params, code, env) are hashed
    3. Dependency chain is recursively tracked
    4. Results can be traced back to exact code version
    """
    
    def __init__(self, script_path, params, output_dir):
        self.script_path = Path(script_path)
        self.params = params
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.utcnow().isoformat() + '+00:00'
        
        # Compute SHA256 hashes
        self.sha = {
            'script': self._hash_file(script_path),
            'env': self._hash_env(),
            'params': self._hash_dict(params),
            'src': self._hash_dir('src/'),  # all source code
        }
        
        # Generate fingerprint (short hash for human readability)
        combined = ''.join(self.sha.values())
        self.fingerprint = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        # Lineage (dependencies)
        self.lineage = []
    
    def add_dependency(self, dep_metadata):
        """Add upstream dependency (e.g., DGP generation step)"""
        self.lineage.append(dep_metadata)
    
    def _hash_file(self, filepath):
        """SHA256 of a single file"""
        filepath = Path(filepath)
        if not filepath.exists():
            return 'missing'
        return hashlib.sha256(filepath.read_bytes()).hexdigest()
    
    def _hash_dir(self, dirpath):
        """SHA256 of all .py files in directory"""
        dirpath = Path(dirpath)
        if not dirpath.exists():
            return 'missing'
        
        all_files = sorted(dirpath.rglob('*.py'))
        combined = b''
        for f in all_files:
            combined += f.read_bytes()
        return hashlib.sha256(combined).hexdigest()
    
    def _hash_env(self):
        """SHA256 of requirements.txt"""
        req_file = Path('requirements.txt')
        if req_file.exists():
            return self._hash_file(req_file)
        return 'none'
    
    def _hash_dict(self, d):
        """SHA256 of dictionary (parameters)"""
        json_str = json.dumps(d, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_git_commit(self):
        """Get git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=self.script_path.parent
            )
            return result.stdout.strip()
        except:
            return 'unknown'
    
    def to_dict(self):
        """Export as dictionary (for JSON serialization)"""
        return {
            'fingerprint': self.fingerprint,
            'timestamp': self.timestamp,
            'script': str(self.script_path),
            'git_commit': self._get_git_commit(),
            'params': self.params,
            'sha': self.sha,
            'lineage': [dep.to_dict() if hasattr(dep, 'to_dict') else dep 
                       for dep in self.lineage],
            'output_dir': str(self.output_dir)
        }
    
    def save(self, filename='experiment_metadata.json'):
        """Save metadata to JSON file"""
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Metadata saved: {output_path}")
        print(f"Fingerprint: {self.fingerprint}")
        return output_path


# Example usage
if __name__ == '__main__':
    # Step 1: DGP generation
    dgp_params = {
        'N': 50,
        'T': 250,
        'seed_base': 42,
        'dgp_type': 'garch_factor',
        'garch_alpha': 0.08,
        'garch_beta': 0.90,
        'innovation_df': 5,
        'n_factors': 3
    }
    
    dgp_meta = ExperimentMetadata(
        script_path='src/extended_dgp.py',
        params=dgp_params,
        output_dir='results/dgp_20260226'
    )
    
    # Step 2: TE estimation
    te_params = {
        'method': 'lasso',
        'input_fingerprint': dgp_meta.fingerprint,  # link to upstream
        'n_trials': 100
    }
    
    te_meta = ExperimentMetadata(
        script_path='src/lasso_simulation.py',
        params=te_params,
        output_dir='results/te_20260226'
    )
    te_meta.add_dependency(dgp_meta.to_dict())  # record lineage
    
    # Save both
    dgp_meta.save('dgp_metadata.json')
    te_meta.save('te_metadata.json')
    
    print("\n--- Lineage Chain ---")
    print(f"DGP:    {dgp_meta.fingerprint}")
    print(f"TE Est: {te_meta.fingerprint}")
    print(f"  └─ Depends on: {dgp_meta.fingerprint}")
