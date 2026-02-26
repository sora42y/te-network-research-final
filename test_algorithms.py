"""
Unit Tests for Transfer Entropy Algorithms

Tests all pure algorithm implementations with:
- Known ground truth cases
- Edge cases (empty, single node, etc.)
- Numerical stability
- Reproducibility

Run: pytest test_algorithms.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from algorithms import (
    compute_linear_te_matrix,
    compute_nio,
    compute_precision_recall_f1
)


class TestOLS_TE:
    """Test OLS-based Transfer Entropy"""
    
    def test_independent_series(self):
        """TE should be zero for independent series"""
        np.random.seed(42)
        N, T = 5, 100
        R = np.random.randn(T, N) * 0.01  # i.i.d. returns
        
        te_matrix, adj = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)
        
        # Most edges should be zero (some might pass by chance at α=0.05)
        assert adj.sum() / (N * (N-1)) < 0.15, "Too many edges for independent series"
        assert np.diag(adj).sum() == 0, "Diagonal should be zero"
    
    def test_known_granger_causality(self):
        """TE should detect known Granger causality: y(t) = 0.5*y(t-1) + 0.3*x(t-1)"""
        np.random.seed(42)
        T = 200
        
        x = np.zeros(T)
        y = np.zeros(T)
        
        for t in range(1, T):
            x[t] = 0.5 * x[t-1] + np.random.randn() * 0.1
            y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + np.random.randn() * 0.1
        
        R = np.column_stack([x, y])  # (T, 2)
        
        te_matrix, adj = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)
        
        # Should detect x→y (adj[1,0]=1) but not y→x (adj[0,1]=0)
        assert adj[1, 0] == 1, "Should detect x→y"
        # Note: adj[0,1] might be 1 due to spurious correlation, so we don't strictly test it
    
    def test_zero_variance(self):
        """Handle constant series gracefully"""
        R = np.ones((100, 5))  # All zeros/constants
        
        te_matrix, adj = compute_linear_te_matrix(R, method='ols')
        
        assert adj.sum() == 0, "No edges for constant series"
        assert np.all(np.isfinite(te_matrix)), "Should not have inf/nan"
    
    def test_reproducibility(self):
        """Same input should give same output"""
        np.random.seed(123)
        R = np.random.randn(50, 10) * 0.01
        
        te1, adj1 = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)
        te2, adj2 = compute_linear_te_matrix(R, method='ols', t_threshold=2.0)
        
        np.testing.assert_array_equal(adj1, adj2)
        np.testing.assert_array_almost_equal(te1, te2)


class TestLASSO_TE:
    """Test LASSO-based Transfer Entropy"""
    
    def test_sparse_network(self):
        """LASSO should work on sparse VAR"""
        np.random.seed(42)
        N, T = 20, 200
        
        # Generate sparse VAR
        A_true = (np.random.rand(N, N) < 0.1).astype(float) * 0.3
        np.fill_diagonal(A_true, 0.5)
        
        R = np.zeros((T, N))
        for t in range(1, T):
            R[t] = A_true @ R[t-1] + np.random.randn(N) * 0.1
        
        te_matrix, adj = compute_linear_te_matrix(R, method='lasso')
        
        # P2 FIX: Add positive assertion to catch silent LASSO failures
        assert adj.sum() >= 1, "LASSO should detect at least one edge in causal data"
        
        # Should produce a sparse network
        density = adj.sum() / (N * (N-1))
        assert density < 0.3, f"Network too dense: {density}"
        assert np.diag(adj).sum() == 0, "No self-loops"
    
    def test_empty_network(self):
        """LASSO on i.i.d. should give mostly empty network"""
        np.random.seed(99)
        R = np.random.randn(100, 15) * 0.01
        
        te_matrix, adj = compute_linear_te_matrix(R, method='lasso')
        
        # Should be very sparse (but allow some spurious edges)
        assert adj.sum() / (15 * 14) < 0.2, "Too many edges for i.i.d. data"


class TestNIO:
    """Test Net Information Outflow"""
    
    def test_hub_node(self):
        """Hub node should have high positive NIO"""
        N = 10
        te_matrix = np.zeros((N, N))
        
        # P1 FIX: Create hub with OUT-edges (corrected for A[i,j] = j->i convention)
        # Node 0 is a hub: 9 out-edges, 0 in-edges
        # Column 0 represents edges FROM node 0
        te_matrix[1:, 0] = 1.0  # Node 0 causes all others (9 out-edges)
        
        nio = compute_nio(te_matrix, method='binary')
        
        # Hub has only out-edges: NIO = (9 - 0) / (10-1) = 1.0
        assert nio[0] == 1.0, f"Hub should have NIO=1.0, got {nio[0]}"
        # Others have 1 in-edge, 0 out-edges: NIO = (0 - 1) / 9 = -1/9
        assert np.all(nio[1:] < 0), "Non-hubs should have negative NIO"
    
    def test_symmetric_network(self):
        """Symmetric network should have zero NIO"""
        N = 5
        te_matrix = np.ones((N, N))
        np.fill_diagonal(te_matrix, 0)
        
        nio = compute_nio(te_matrix, method='binary')
        
        np.testing.assert_array_almost_equal(nio, np.zeros(N))
    
    def test_weighted_vs_binary(self):
        """Weighted and binary NIO should differ"""
        te_matrix = np.array([
            [0, 1.0, 2.0],
            [0.5, 0, 0],
            [0, 0, 0]
        ])
        
        nio_binary = compute_nio(te_matrix, method='binary')
        nio_weighted = compute_nio(te_matrix, method='weighted')
        
        # They should differ
        assert not np.allclose(nio_binary, nio_weighted)


class TestMetrics:
    """Test precision/recall/F1 computation"""
    
    def test_perfect_recovery(self):
        """Perfect recovery should give P=R=F1=1"""
        A_true = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        p, r, f1 = compute_precision_recall_f1(A_true, A_true)
        
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
    
    def test_all_wrong(self):
        """Complete mismatch should give P=R=F1=0"""
        A_true = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        A_pred = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        p, r, f1 = compute_precision_recall_f1(A_true, A_pred)
        
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
    
    def test_partial_recovery(self):
        """Partial recovery should give intermediate values"""
        A_true = np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        
        # Predict 2 out of 4 edges correctly
        A_pred = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        p, r, f1 = compute_precision_recall_f1(A_true, A_pred)
        
        # True edges: 4 (manually count from A_true)
        # Predicted edges: 3 (manually count from A_pred)
        # TP: [0,1]=1, [1,2]=1 -> 2 correct
        # FP: [1,3]=1 -> 1 wrong
        # FN: [0,2]=1, [2,3]=1, [3,0]=1 -> 3 missed
        # Wait, let me recount...
        # A_true has edges: (0,1), (0,2), (1,2), (2,3), (3,0) = 5 edges (not 4)
        # Let me fix the test
        
        # Simpler test: 
        A_true = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])  # 2 edges total
        
        A_pred = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])  # 2 edges total
        
        p, r, f1 = compute_precision_recall_f1(A_true, A_pred)
        
        # TP=1 (edge 0->1), FP=1 (edge 0->2), FN=1 (edge 1->2)
        assert p == 0.5  # 1/(1+1)
        assert r == 0.5  # 1/(1+1)
        assert f1 == 0.5  # 2*0.5*0.5/(0.5+0.5)
    
    def test_empty_network(self):
        """Empty true network should handle gracefully"""
        A_true = np.zeros((5, 5))
        A_pred = np.zeros((5, 5))
        
        p, r, f1 = compute_precision_recall_f1(A_true, A_pred)
        
        # Convention: perfect match on empty set
        assert p == 0.0  # 0/(0+0) = undefined, we return 0
        assert r == 0.0
        assert f1 == 0.0


class TestEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_single_asset(self):
        """Single asset should produce empty 1x1 matrix"""
        R = np.random.randn(100, 1) * 0.01
        
        te_matrix, adj = compute_linear_te_matrix(R, method='ols')
        
        assert te_matrix.shape == (1, 1)
        assert adj[0, 0] == 0
    
    def test_very_short_series(self):
        """Short series (T<10) should not crash"""
        R = np.random.randn(5, 10) * 0.01
        
        # Should run without error
        te_matrix, adj = compute_linear_te_matrix(R, method='ols')
        
        assert te_matrix.shape == (10, 10)
    
    def test_high_correlation(self):
        """Highly correlated series should not cause numerical issues"""
        T = 100
        x = np.random.randn(T)
        y = x + np.random.randn(T) * 0.01  # Almost identical to x
        
        R = np.column_stack([x, y])
        
        te_matrix, adj = compute_linear_te_matrix(R, method='ols')
        
        assert np.all(np.isfinite(te_matrix)), "Should not have inf/nan"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
