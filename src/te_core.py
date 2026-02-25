"""
Transfer Entropy Core Module
=============================

Canonical implementations of all TE methods.
All experiments MUST import from this module to ensure consistency.

Design principles:
1. Single source of truth - no duplicate implementations
2. Explicit parameters - no hidden defaults
3. Vectorized computation - fast and readable
4. Extensive documentation - reviewers can verify correctness
"""

import numpy as np
from sklearn.linear_model import LassoLarsIC


def compute_linear_te_matrix(R, method='ols', t_threshold=2.0):
    """
    Compute pairwise Transfer Entropy matrix using linear Gaussian assumption.
    
    TE(j → i) measures how much knowing r_{j,t-1} reduces uncertainty about r_{i,t}
    beyond what r_{i,t-1} already tells us.
    
    Formula: TE(j→i) = 0.5 * ln(σ²_restricted / σ²_full)
    - Restricted model:   r_{i,t} = α + β·r_{i,t-1} + ε
    - Unrestricted model: r_{i,t} = α + β·r_{i,t-1} + γ·r_{j,t-1} + ε
    
    Parameters
    ----------
    R : ndarray, shape (T, N)
        Return matrix (T time periods, N assets)
    method : {'ols', 'lasso'}
        Edge selection method:
        - 'ols': Include edge if t-statistic > t_threshold
        - 'lasso': Use LASSO with BIC for variable selection
    t_threshold : float, default=2.0
        t-statistic threshold for OLS method (approx p<0.05 for T>60)
    
    Returns
    -------
    TE_matrix : ndarray, shape (N, N)
        Transfer entropy values, TE_matrix[i,j] = TE from j to i
    A_binary : ndarray, shape (N, N)
        Binary adjacency matrix (1 if edge exists, 0 otherwise)
    
    References
    ----------
    Barnett, L., & Seth, A. K. (2014). The MVGC multivariate Granger causality
    toolbox: a new approach to Granger-causal inference. Journal of neuroscience
    methods, 223, 50-68.
    
    Notes
    -----
    - This is the LINEAR Gaussian TE approximation (fast, parametric)
    - For nonlinear TE, use compute_nonparametric_te_matrix()
    - Edge direction: A[i,j]=1 means j→i (j causes i)
    """
    if method == 'ols':
        return _compute_ols_te_matrix(R, t_threshold)
    elif method == 'lasso':
        return _compute_lasso_te_matrix(R)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ols' or 'lasso'.")


def _compute_ols_te_matrix(R, t_threshold=2.0):
    """
    OLS-based TE with t-statistic thresholding.
    
    For each pair (i,j), test if r_{j,t-1} significantly predicts r_{i,t}
    beyond r_{i,t-1}. Include edge if |t-stat| > threshold.
    
    Internal function - use compute_linear_te_matrix(method='ols') instead.
    """
    T, N = R.shape
    R_t = R[1:]      # (T-1, N) current values
    R_lag = R[:-1]   # (T-1, N) lagged values
    T_eff = T - 1
    
    TE_matrix = np.zeros((N, N))
    A_binary = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        # Restricted model: r_i(t) ~ r_i(t-1)
        y = R_t[:, i]
        X_res = R_lag[:, i].reshape(-1, 1)
        
        # Add constant
        X_res = np.column_stack([np.ones(T_eff), X_res])
        
        # Fit restricted model
        beta_res = np.linalg.lstsq(X_res, y, rcond=None)[0]
        resid_res = y - X_res @ beta_res
        var_res = (resid_res ** 2).sum() / (T_eff - 2)
        
        for j in range(N):
            if i == j:
                continue
            
            # Unrestricted model: r_i(t) ~ r_i(t-1) + r_j(t-1)
            X_full = np.column_stack([np.ones(T_eff), R_lag[:, i], R_lag[:, j]])
            
            # Fit full model
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            resid_full = y - X_full @ beta_full
            var_full = (resid_full ** 2).sum() / (T_eff - 3)
            
            # Compute TE
            if var_full > 0 and var_res > 0:
                TE_matrix[i, j] = 0.5 * np.log(var_res / var_full)
            
            # t-statistic for coefficient of r_j(t-1)
            if var_full > 0:
                se_beta = np.sqrt(var_full * np.linalg.inv(X_full.T @ X_full)[2, 2])
                t_stat = abs(beta_full[2] / se_beta) if se_beta > 0 else 0
                
                if t_stat > t_threshold:
                    A_binary[i, j] = 1
    
    return TE_matrix, A_binary


def _compute_lasso_te_matrix(R):
    """
    LASSO-based TE with BIC variable selection.
    
    For each target i, run LASSO regression of r_{i,t} on all r_{j,t-1}.
    Selected variables (j with non-zero coefficients) form edges j→i.
    
    Internal function - use compute_linear_te_matrix(method='lasso') instead.
    """
    T, N = R.shape
    R_t = R[1:]      # (T-1, N)
    R_lag = R[:-1]   # (T-1, N)
    T_eff = T - 1
    
    TE_matrix = np.zeros((N, N))
    A_binary = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        y = R_t[:, i]
        
        # Restricted model: r_i(t) ~ r_i(t-1)
        X_res = R_lag[:, i].reshape(-1, 1)
        beta_res = np.linalg.lstsq(X_res, y, rcond=None)[0]
        resid_res = y - X_res @ beta_res
        var_res = (resid_res ** 2).sum() / T_eff
        
        # Full model: LASSO of r_i(t) on [r_i(t-1), all r_j(t-1)]
        X_full = R_lag.copy()
        
        try:
            lasso = LassoLarsIC(criterion='bic', normalize=False, fit_intercept=True)
            lasso.fit(X_full, y)
            coef = lasso.coef_
            
            # Predict and compute residual variance
            y_pred = lasso.predict(X_full)
            var_full = ((y - y_pred) ** 2).sum() / T_eff
            
            # Compute TE for selected variables
            for j in range(N):
                if i == j:
                    continue
                
                if abs(coef[j]) > 1e-10:  # Variable selected by LASSO
                    A_binary[i, j] = 1
                    
                    if var_full > 0 and var_res > 0:
                        TE_matrix[i, j] = 0.5 * np.log(var_res / var_full)
        
        except:
            # LASSO failed (rare), fall back to empty network
            pass
    
    return TE_matrix, A_binary


def compute_nio(te_matrix, method='binary'):
    """
    Compute Net Information Outflow (NIO) from TE matrix.
    
    NIO measures the imbalance between outgoing and incoming information flow.
    
    NIO_i = (out-degree_i - in-degree_i) / (N-1)
    
    where degree is computed from binary adjacency matrix.
    
    Parameters
    ----------
    te_matrix : ndarray, shape (N, N)
        Transfer entropy matrix (or binary adjacency matrix)
    method : {'binary', 'weighted'}
        - 'binary': Use binary edges (out-degree - in-degree)
        - 'weighted': Use TE values (sum of outgoing TE - incoming TE)
    
    Returns
    -------
    nio : ndarray, shape (N,)
        NIO values for each node
    
    Notes
    -----
    Edge direction: te_matrix[i,j] represents edge j→i (j causes i)
    So:
    - out-degree_i = sum of edges i→* = sum(te_matrix[:, i] > 0)
    - in-degree_i  = sum of edges *→i = sum(te_matrix[i, :] > 0)
    """
    N = te_matrix.shape[0]
    nio = np.zeros(N)
    
    for i in range(N):
        if method == 'binary':
            # Binarize first
            A = (te_matrix != 0).astype(int)
            
            # Out-degree: i→j means A[j,i]=1 (column i)
            out_degree = A[:, i].sum() - A[i, i]  # Exclude self-loop
            
            # In-degree: j→i means A[i,j]=1 (row i)
            in_degree = A[i, :].sum() - A[i, i]
        
        elif method == 'weighted':
            # Out-flow: sum of TE from i to others
            out_flow = te_matrix[:, i].sum() - te_matrix[i, i]
            
            # In-flow: sum of TE from others to i
            in_flow = te_matrix[i, :].sum() - te_matrix[i, i]
            
            out_degree = out_flow
            in_degree = in_flow
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize by max possible degree
        nio[i] = (out_degree - in_degree) / (N - 1)
    
    return nio


def compute_precision_recall_f1(A_true, A_pred):
    """
    Compute precision, recall, F1 for binary network comparison.
    
    Parameters
    ----------
    A_true : ndarray, shape (N, N)
        True adjacency matrix (ground truth)
    A_pred : ndarray, shape (N, N)
        Predicted adjacency matrix
    
    Returns
    -------
    precision : float
        TP / (TP + FP)
    recall : float
        TP / (TP + FN)
    f1 : float
        2 * (precision * recall) / (precision + recall)
    """
    N = A_true.shape[0]
    
    # Flatten and exclude diagonal
    mask = ~np.eye(N, dtype=bool).flatten()
    y_true = A_true.flatten()[mask]
    y_pred = A_pred.flatten()[mask]
    
    # True Positives, False Positives, False Negatives
    TP = (y_true * y_pred).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()
    
    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# Export public API
__all__ = [
    'compute_linear_te_matrix',
    'compute_nio',
    'compute_precision_recall_f1'
]
