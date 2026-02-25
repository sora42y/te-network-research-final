# Code Consolidation Plan

## Problem
Multiple implementations of the same TE methods exist across different files:
- `factor_neutral_te.py::compute_transfer_entropy()` (Linear TE)
- `lasso_simulation.py::compute_ols_te_matrix()` (Linear TE)
- Potential subtle differences in implementation

## Solution
Create a unified `te_core.py` module with canonical implementations.

## Consolidation Plan

### Step 1: Create `src/te_core.py`
```python
"""
Core Transfer Entropy Implementations
All TE calculations should import from this module.
"""

def compute_linear_te_matrix(R, method='ols', t_threshold=2.0):
    """
    Unified linear TE implementation.
    
    Parameters
    ----------
    R : ndarray (T, N)
        Return matrix
    method : str
        'ols' : pairwise OLS with t-statistic thresholding
        'lasso' : LASSO with BIC selection
    t_threshold : float
        t-statistic threshold for OLS method
    
    Returns
    -------
    TE_matrix : ndarray (N, N)
        Transfer entropy matrix
    A_binary : ndarray (N, N)
        Binary adjacency matrix
    """
    if method == 'ols':
        return _compute_ols_te(R, t_threshold)
    elif method == 'lasso':
        return _compute_lasso_te(R)
    else:
        raise ValueError(f"Unknown method: {method}")

def _compute_ols_te(R, t_threshold):
    """OLS-based TE (internal use only)"""
    # Move implementation from lasso_simulation.py here
    pass

def _compute_lasso_te(R):
    """LASSO-based TE (internal use only)"""
    # Move implementation from lasso_simulation.py here
    pass

def compute_nio(te_matrix):
    """
    Net Information Outflow from TE matrix.
    Canonical implementation (used by all experiments).
    """
    # Move from oracle_nio_power.py
    pass
```

### Step 2: Deprecate old functions
```python
# In factor_neutral_te.py
def compute_transfer_entropy(returns, lag=1, bins=10):
    """
    DEPRECATED: Use te_core.compute_linear_te_matrix() instead.
    """
    import warnings
    warnings.warn("Use te_core.compute_linear_te_matrix()", DeprecationWarning)
    from .te_core import compute_linear_te_matrix
    return compute_linear_te_matrix(returns.values, method='ols')
```

### Step 3: Update all imports
```python
# Before
from lasso_simulation import compute_ols_te_matrix
from factor_neutral_te import compute_transfer_entropy

# After
from te_core import compute_linear_te_matrix
```

### Step 4: Add unit tests
```python
# tests/test_te_core.py
def test_ols_lasso_consistency():
    """Verify OLS and LASSO produce same results on simple DGP"""
    R = generate_test_data()
    te_ols, _ = compute_linear_te_matrix(R, method='ols')
    te_lasso, _ = compute_linear_te_matrix(R, method='lasso')
    # Check correlation > 0.9
    assert np.corrcoef(te_ols.flatten(), te_lasso.flatten())[0,1] > 0.9
```

## Benefits
1. Single source of truth for TE calculations
2. No risk of subtle differences between implementations
3. Easier to verify correctness
4. Clear API for reviewers

## Migration Checklist
- [ ] Create src/te_core.py
- [ ] Move compute_ols_te_matrix() → te_core._compute_ols_te()
- [ ] Move compute_lasso_te_matrix() → te_core._compute_lasso_te()
- [ ] Move compute_nio() → te_core.compute_nio()
- [ ] Deprecate old functions (with warnings)
- [ ] Update all imports in experiments
- [ ] Add unit tests
- [ ] Verify all tables still match
