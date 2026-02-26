"""
Core Transfer Entropy Algorithms (Pure Implementations)

This module contains ONLY pure algorithmic implementations with NO:
- External file I/O
- Hardcoded paths
- Side effects

All functions are deterministic and unit-testable.
"""

import numpy as np
from sklearn.linear_model import LassoLarsIC


def compute_linear_te_matrix(R, method='ols', t_threshold=2.0):
    """
    Compute linear Transfer Entropy matrix.
    
    Parameters
    ----------
    R : ndarray (T, N)
        Return matrix (T time steps, N assets)
    method : str
        'ols' - OLS-TE with t-statistic thresholding
        'lasso' - LASSO-TE with BIC regularization
    t_threshold : float
        t-statistic threshold for edge inclusion (OLS only)
    
    Returns
    -------
    TE_matrix : ndarray (N, N)
        Transfer entropy values [i,j] = TE from j to i
    A_binary : ndarray (N, N)
        Binary adjacency matrix (1 = edge present)
    
    Notes
    -----
    Edge direction: A[i,j]=1 means j→i (j Granger-causes i)
    Diagonal is always zero (no self-loops)
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
    
    Algorithm:
    For each pair (i,j):
      1. Restricted model: r_i(t) ~ r_i(t-1) + intercept
      2. Unrestricted: r_i(t) ~ r_i(t-1) + r_j(t-1) + intercept
      3. TE(j→i) = 0.5 * ln(σ²_restricted / σ²_unrestricted)
      4. Include edge only if |t-stat(β_j)| > threshold
    """
    T, N = R.shape
    R_t = R[1:]      # (T-1, N) current values
    R_lag = R[:-1]   # (T-1, N) lagged values
    T_eff = T - 1
    
    TE_matrix = np.zeros((N, N))
    A_binary = np.zeros((N, N), dtype=int)
    
    ones = np.ones((T_eff, 1))
    
    for i in range(N):
        y = R_t[:, i]
        
        # Restricted model: r_i(t) ~ r_i(t-1) + constant
        X_res = np.column_stack([ones, R_lag[:, i]])
        beta_res = np.linalg.lstsq(X_res, y, rcond=None)[0]
        resid_res = y - X_res @ beta_res
        sigma2_res = np.mean(resid_res ** 2)
        
        for j in range(N):
            if i == j:
                continue
            
            # Unrestricted model: r_i(t) ~ r_i(t-1) + r_j(t-1) + constant
            X_full = np.column_stack([ones, R_lag[:, i], R_lag[:, j]])
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            resid_full = y - X_full @ beta_full
            sigma2_full = np.mean(resid_full ** 2)
            
            # Skip if variance estimate invalid
            if sigma2_full < 1e-12 or sigma2_res < sigma2_full:
                continue
            
            # t-statistic for coefficient of r_j(t-1)
            dof = T_eff - 3
            s2 = np.sum(resid_full ** 2) / dof
            try:
                cov = s2 * np.linalg.inv(X_full.T @ X_full)
                se_j = np.sqrt(cov[2, 2])
                t_stat = abs(beta_full[2] / (se_j + 1e-12))
            except np.linalg.LinAlgError:
                # P1 FIX: Singular matrix (perfect collinearity)
                t_stat = 0
            
            # Include edge only if t-stat exceeds threshold
            if t_stat > t_threshold:
                TE_matrix[i, j] = 0.5 * np.log(sigma2_res / sigma2_full)
                A_binary[i, j] = 1
    
    return TE_matrix, A_binary


def _compute_lasso_te_matrix(R):
    """
    LASSO-based TE with BIC regularization.
    
    Algorithm:
    For each asset i:
      1. Frisch-Waugh: Residualize y and X on own lag
      2. Fit LASSO on residualized cross-lags (BIC criterion)
      3. Include edge j→i if LASSO coefficient ≠ 0
    """
    T, N = R.shape
    R_t = R[1:]
    R_lag = R[:-1]
    T_eff = T - 1
    
    TE_matrix = np.zeros((N, N))
    A_binary = np.zeros((N, N), dtype=int)
    
    ones = np.ones((T_eff, 1))
    
    for i in range(N):
        y = R_t[:, i]
        own_lag = R_lag[:, i].reshape(-1, 1)
        
        # Other assets' lags
        other_idx = [j for j in range(N) if j != i]
        X_others = R_lag[:, other_idx]
        
        # Frisch-Waugh residualization
        X_res_with_const = np.column_stack([ones, own_lag])
        
        # Residualize y
        beta_y = np.linalg.lstsq(X_res_with_const, y, rcond=None)[0]
        y_resid = y - X_res_with_const @ beta_y
        
        # Residualize X_others
        beta_X = np.linalg.lstsq(X_res_with_const, X_others, rcond=None)[0]
        X_resid = X_others - X_res_with_const @ beta_X
        
        # Fit LASSO (P0 FIX: removed normalize=False for sklearn >= 1.2)
        try:
            lasso = LassoLarsIC(criterion='bic', max_iter=1000)
            lasso.fit(X_resid, y_resid)
            coef = lasso.coef_
            
            # Map back to full index
            for k, j in enumerate(other_idx):
                if coef[k] != 0:
                    A_binary[i, j] = 1
                    # Compute TE for selected edges
                    X_full = np.column_stack([ones, own_lag, R_lag[:, j]])
                    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
                    resid_full = y - X_full @ beta_full
                    
                    X_restricted = np.column_stack([ones, own_lag])
                    beta_res = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                    resid_res = y - X_restricted @ beta_res
                    
                    sigma2_res = np.mean(resid_res ** 2)
                    sigma2_full = np.mean(resid_full ** 2)
                    
                    if sigma2_full > 0 and sigma2_res > sigma2_full:
                        TE_matrix[i, j] = 0.5 * np.log(sigma2_res / sigma2_full)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            # P1 FIX: Catch specific exceptions instead of bare except
            import warnings
            warnings.warn(f"LASSO fitting failed for edge {i}->{j}: {e}")
    
    return TE_matrix, A_binary


def compute_nio(te_matrix, method='binary'):
    """
    Compute Net Information Outflow from TE matrix.
    
    Parameters
    ----------
    te_matrix : ndarray (N, N)
        Transfer entropy matrix [i,j] = TE from j to i
    method : str
        'binary' - Use binary adjacency (count edges)
        'weighted' - Use TE values
    
    Returns
    -------
    nio : ndarray (N,)
        Net information outflow for each node
    
    Notes
    -----
    NIO_i = (out_flow_i - in_flow_i) / (N-1)
    Normalized by N-1 to be comparable across network sizes
    
    Matrix convention: TE[i,j] represents j->i causality
    Therefore: column sums give out-degree, row sums give in-degree
    """
    N = te_matrix.shape[0]
    nio = np.zeros(N)
    
    for i in range(N):
        if method == 'binary':
            # Binary: count edges (P1 FIX: corrected direction)
            # Column sum (TE[:,i]) = edges FROM i TO others = out-degree
            # Row sum (TE[i,:]) = edges FROM others TO i = in-degree
            out_flow = (te_matrix[:, i] > 0).sum() - (te_matrix[i, i] > 0)
            in_flow = (te_matrix[i, :] > 0).sum() - (te_matrix[i, i] > 0)
        elif method == 'weighted':
            # Weighted: sum TE values (P1 FIX: corrected direction)
            out_flow = te_matrix[:, i].sum() - te_matrix[i, i]
            in_flow = te_matrix[i, :].sum() - te_matrix[i, i]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        nio[i] = (out_flow - in_flow) / (N - 1)
    
    return nio


def compute_precision_recall_f1(A_true, A_pred):
    """
    Compute precision, recall, and F1 score for binary adjacency matrices.
    
    Parameters
    ----------
    A_true : ndarray (N, N)
        True adjacency matrix
    A_pred : ndarray (N, N)
        Predicted adjacency matrix
    
    Returns
    -------
    precision : float
        TP / (TP + FP)
    recall : float
        TP / (TP + FN)
    f1 : float
        2 * precision * recall / (precision + recall)
    """
    N = A_true.shape[0]
    
    # Flatten and exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    y_true = A_true[mask].flatten()
    y_pred = A_pred[mask].flatten()
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
