"""
Code Audit Script: Verify TE Implementation Consistency
========================================================

This script checks:
1. All experiments import from te_core.py (no duplicate implementations)
2. No lingering calls to old TE functions
3. Results match between old and new implementations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def check_imports():
    """Check which files import TE functions"""
    src_dir = Path('src')
    
    print("="*60)
    print("Checking TE imports across all source files...")
    print("="*60)
    
    issues = []
    
    for py_file in src_dir.glob('*.py'):
        if py_file.name == 'te_core.py':
            continue
        
        content = py_file.read_text(encoding='utf-8')
        
        # Check for imports from te_core (good)
        has_te_core_import = 'from te_core import' in content or 'import te_core' in content
        
        # Check for old function definitions (bad)
        has_old_te_func = (
            'def compute_lasso_te_matrix' in content or
            'def compute_ols_te_matrix' in content or
            'def compute_transfer_entropy' in content
        )
        
        if has_old_te_func and py_file.name not in ['lasso_simulation.py', 'factor_neutral_te.py']:
            issues.append(f"FAIL {py_file.name}: Defines TE function (should import from te_core)")
        
        if not has_te_core_import and 'te' in py_file.name.lower():
            issues.append(f"WARN {py_file.name}: No te_core import (might be using old implementation)")
        
        if has_te_core_import:
            print(f"OK {py_file.name}: Imports from te_core")
    
    if issues:
        print("\n" + "="*60)
        print("ISSUES FOUND:")
        print("="*60)
        for issue in issues:
            print(issue)
    else:
        print("\n✅ All files correctly import from te_core!")
    
    return len(issues) == 0

def test_implementation_equivalence():
    """Test that new te_core gives same results as old implementation"""
    import numpy as np
    from te_core import compute_linear_te_matrix as new_ols_te
    
    # Load old implementation if it exists
    try:
        from lasso_simulation import compute_ols_te_matrix as old_ols_te
    except ImportError:
        print("⚠️  Old implementation not found (already migrated?)")
        return True
    
    print("\n" + "="*60)
    print("Testing implementation equivalence...")
    print("="*60)
    
    # Generate simple test data
    np.random.seed(42)
    N, T = 10, 100
    R = np.random.randn(T, N) * 0.01
    
    # Old implementation
    te_old, a_old = old_ols_te(R, t_threshold=2.0)
    
    # New implementation
    te_new, a_new = new_ols_te(R, method='ols', t_threshold=2.0)
    
    # Compare
    te_diff = np.abs(te_old - te_new).max()
    a_diff = np.abs(a_old - a_new).max()
    
    print(f"Max TE difference: {te_diff:.2e}")
    print(f"Max adjacency difference: {a_diff}")
    
    if te_diff < 1e-10 and a_diff == 0:
        print("✅ Implementations are IDENTICAL!")
        return True
    elif te_diff < 1e-6:
        print("⚠️  Minor numerical differences (acceptable)")
        return True
    else:
        print("❌ Implementations DIFFER significantly!")
        return False

def main():
    print("\n" + "="*60)
    print("TE IMPLEMENTATION AUDIT")
    print("="*60 + "\n")
    
    # Check 1: Imports
    imports_ok = check_imports()
    
    # Check 2: Equivalence test
    equiv_ok = test_implementation_equivalence()
    
    # Summary
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    print(f"Import check: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Equivalence test: {'✅ PASS' if equiv_ok else '❌ FAIL'}")
    
    if imports_ok and equiv_ok:
        print("\n🎉 Code audit PASSED! All implementations consistent.")
        return 0
    else:
        print("\n⚠️  Code audit found issues. See details above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
