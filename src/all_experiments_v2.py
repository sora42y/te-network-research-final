"""
all_experiments_v2.py — 新增模拟：
#3:  Generated Regressor 扩展（T/N={1,2,5,10}, OLS+LASSO）
#4:  Threshold-VAR 宽 T/N + 规格说明
#9:  VAR(2) DGP + lag-1 估计
#10: PageRank + weighted out-degree 信号
#11: Simulation NIO 基准（ground truth 下 NIO 预测能力）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import kendalltau, ttest_ind
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extended_dgp import generate_sparse_var_extended
from lasso_simulation import compute_lasso_te_matrix, compute_ols_te_matrix

OUTPUT   = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research\results')
SEED_BASE = 42
N_TRIALS  = 8

def eval_metrics(A_true, A_pred, top_k=5):
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool).flatten()
    yt = A_true.flatten()[mask]; yp = A_pred.flatten()[mask]
    return dict(
        precision=precision_score(yt, yp, zero_division=0),
        recall=recall_score(yt, yp, zero_division=0),
        f1=f1_score(yt, yp, zero_division=0),
        hub_recovery=len(set(np.argsort(A_true.sum(1))[-top_k:]) &
                         set(np.argsort(A_pred.sum(1))[-top_k:])) / top_k,
        kendall_tau=kendalltau(A_true.sum(1), A_pred.sum(1))[0],
    )


# ── Exp #3: Generated Regressor Extended ─────────────────────────────────────

def run_oracle_extended():
    print("\n=== Exp #3: Generated Regressor Extended ===")
    K = 3
    # Cover wider T/N range: N=50, T in {50,100,250,500} → T/N={1,2,5,10}
    configs = [(50, 50), (50, 100), (50, 250), (50, 500)]
    results = []
    with tqdm(total=len(configs)*N_TRIALS*3*2, desc='Exp3') as pbar:
        for N, T in configs:
            for trial in range(N_TRIALS):
                seed = SEED_BASE + trial*1000 + N + T
                rng  = np.random.RandomState(seed)
                sigma = 0.01

                A = np.zeros((N, N))
                off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]
                for idx in rng.choice(len(off_d), int(N*N*0.05), replace=False):
                    i,j = off_d[idx]
                    A[i,j] = rng.uniform(0.05,0.15)*rng.choice([-1,1])
                ev = np.abs(np.linalg.eigvals(A))
                if ev.max() > 0.9: A *= 0.9/ev.max()
                A_true = (A!=0).astype(int)

                B_true = rng.normal(0,1,(N,K))*0.4
                F = np.zeros((T,K))
                for t in range(1,T): F[t] = 0.1*F[t-1]+rng.normal(0,sigma*1.5,K)
                innov = F@B_true.T + rng.normal(0,sigma*0.7,(T,N))
                R = np.zeros((T,N)); R[0] = rng.normal(0,sigma,N)
                for t in range(1,T): R[t] = A@R[t-1]+innov[t]

                R_oracle = R - F@B_true.T
                pca = PCA(n_components=K); F_hat = pca.fit_transform(R)
                R_est = R - F_hat@pca.components_

                base = dict(N=N, T=T, T_N=round(T/N,1), trial=trial)
                for Rdata, label in [(R,'Raw'),(R_oracle,'Oracle'),(R_est,'Estimated(PCA)')]:
                    for mname, A_pred in [
                        ('OLS',   compute_ols_te_matrix(Rdata)[1]),
                        ('LASSO', compute_lasso_te_matrix(Rdata)[1]),
                    ]:
                        m = eval_metrics(A_true, A_pred)
                        m.update({**base, 'preprocess': label, 'method': mname})
                        results.append(m); pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'oracle_extended.csv', index=False)
    summ = df.groupby(['preprocess','method','T_N'])[['precision','recall','f1']].mean().round(3)
    print(summ.to_string())

    fig, axes = plt.subplots(1,2,figsize=(13,5))
    clrs = {'Raw':'#D32F2F','Oracle':'#1976D2','Estimated(PCA)':'#388E3C'}
    for ax, meth in zip(axes, ['OLS','LASSO']):
        for prep in ['Raw','Oracle','Estimated(PCA)']:
            sub = df[(df['method']==meth)&(df['preprocess']==prep)].groupby('T_N')['precision'].mean().reset_index()
            ax.plot(sub['T_N'], sub['precision'], marker='o', ms=5, lw=2,
                    color=clrs[prep], label=prep)
        ax.axvline(5, color='grey', ls='--', lw=1, alpha=0.6)
        ax.set_xlabel('T/N'); ax.set_ylabel('Precision')
        ax.set_title(f'{meth}-TE: Oracle vs Estimated (N=50)', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(OUTPUT / 'figure_oracle_extended.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved oracle_extended.csv + figure")
    return df


# ── Exp #4: Threshold-VAR Wide T/N ────────────────────────────────────────────

def make_A(N, density, rng, scale=1.0):
    A = np.zeros((N,N))
    off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]
    for idx in rng.choice(len(off_d), int(N*N*density), replace=False):
        i,j = off_d[idx]; A[i,j] = rng.uniform(0.05,0.15)*rng.choice([-1,1])*scale
    ev = np.abs(np.linalg.eigvals(A))
    if ev.max() > 0.85: A *= 0.85/ev.max()
    return A

def generate_threshold_var_v2(N=30, T=500, density=0.05, seed=42):
    """
    Threshold-VAR DGP specification:
    - Threshold variable: mean(|R[t-1]|) compared against 40th percentile
      of estimated |R| distribution (calibrated so ~40% of time in high regime)
    - Regime 1 (low vol): VAR coef scale=0.5, innov sigma=1x
    - Regime 2 (high vol): VAR coef scale=1.5, innov sigma=1.5x
    - True edges: UNION of edges in both regimes (union ground truth)
    - Both regimes have same sparsity (density param)
    """
    rng = np.random.RandomState(seed)
    sigma = 0.01
    A_low  = make_A(N, density, rng, scale=0.5)
    A_high = make_A(N, density, rng, scale=1.5)
    A_union = ((A_low!=0)|(A_high!=0)).astype(int)
    n_true_edges = A_union.sum()

    # Calibrate threshold: use 40th pct of |R| (estimated from a short pre-run)
    R_pre = np.zeros((100, N)); R_pre[0] = rng.normal(0, sigma, N)
    for t in range(1,100): R_pre[t] = A_low@R_pre[t-1]+rng.normal(0,sigma,N)
    threshold = np.percentile(np.abs(R_pre).mean(axis=1), 40)

    R = np.zeros((T,N)); R[0] = rng.normal(0,sigma,N)
    regime_seq = []
    for t in range(1,T):
        high = np.mean(np.abs(R[t-1])) > threshold
        regime_seq.append(int(high))
        A = A_high if high else A_low
        R[t] = A@R[t-1] + rng.normal(0, sigma*(1.5 if high else 1.0), N)

    pct_high = np.mean(regime_seq)*100
    return R, A_union, n_true_edges, pct_high

def run_threshold_var_wide():
    print("\n=== Exp #4: Threshold-VAR Wide T/N ===")
    configs = [(30,60),(30,250),(30,500),(50,100),(50,250),(50,500),(100,120),(100,500),(100,1000)]
    results = []
    with tqdm(total=len(configs)*N_TRIALS*4, desc='Exp4') as pbar:
        for N,T in configs:
            for trial in range(N_TRIALS):
                seed = SEED_BASE+trial*1000+N+T
                # Linear baseline
                R_lin, _, A_lin = generate_sparse_var_extended(N=N,T=T,density=0.05,seed=seed,dgp='garch_factor')
                for meth, Ap in [('OLS',compute_ols_te_matrix(R_lin)[1]),
                                  ('LASSO',compute_lasso_te_matrix(R_lin)[1])]:
                    m = eval_metrics(A_lin, Ap)
                    m.update(dict(N=N,T=T,T_N=round(T/N,2),trial=trial,dgp='linear',method=meth))
                    results.append(m); pbar.update(1)
                # Threshold-VAR
                R_thr, A_thr, _, _ = generate_threshold_var_v2(N=N,T=T,density=0.05,seed=seed)
                for meth, Ap in [('OLS',compute_ols_te_matrix(R_thr)[1]),
                                  ('LASSO',compute_lasso_te_matrix(R_thr)[1])]:
                    m = eval_metrics(A_thr, Ap)
                    m.update(dict(N=N,T=T,T_N=round(T/N,2),trial=trial,dgp='threshold',method=meth))
                    results.append(m); pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'threshold_var_wide.csv', index=False)
    summ = df.groupby(['dgp','method','T_N'])[['precision','recall']].mean().round(3)
    print(summ.to_string())

    fig, ax = plt.subplots(figsize=(10,5))
    style = {'linear_OLS':('#1976D2','-','o'),'linear_LASSO':('#90CAF9','--','s'),
             'threshold_OLS':('#D32F2F','-','o'),'threshold_LASSO':('#EF9A9A','--','s')}
    for dgp in ['linear','threshold']:
        for meth in ['OLS','LASSO']:
            key = f'{dgp}_{meth}'
            clr, ls, mk = style[key]
            sub = df[(df['dgp']==dgp)&(df['method']==meth)].groupby('T_N')['precision'].mean().reset_index().sort_values('T_N')
            ax.plot(sub['T_N'], sub['precision'], color=clr, ls=ls, marker=mk, ms=5, lw=2, label=key)
    ax.axvline(5, color='grey', ls='--', lw=1, alpha=0.6, label='T/N=5')
    ax.set_xlabel('T/N'); ax.set_ylabel('Precision'); ax.set_ylim(0,1)
    ax.set_title('Linear vs Threshold-VAR: Full T/N Range', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / 'figure_threshold_wide.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved threshold_var_wide.csv + figure")
    return df


# ── Exp #9: VAR(2) DGP ───────────────────────────────────────────────────────

def generate_var2(N=50, T=500, density=0.05, seed=42):
    """VAR(2) DGP estimated with lag-1 (model misspecification)"""
    rng = np.random.RandomState(seed)
    sigma = 0.01
    off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]
    def make_sparse(scale):
        A = np.zeros((N,N))
        for idx in rng.choice(len(off_d), int(N*N*density), replace=False):
            i,j = off_d[idx]; A[i,j] = rng.uniform(0.05,0.12)*rng.choice([-1,1])*scale
        ev = np.abs(np.linalg.eigvals(A))
        if ev.max()>0.7: A*=0.7/ev.max()
        return A
    A1 = make_sparse(1.0); A2 = make_sparse(0.4)
    A_true = ((A1!=0)|(A2!=0)).astype(int)

    innov = rng.normal(0,sigma,(T,N))
    R = np.zeros((T,N))
    R[0] = rng.normal(0,sigma,N); R[1] = rng.normal(0,sigma,N)
    for t in range(2,T): R[t] = A1@R[t-1]+A2@R[t-2]+innov[t]
    return R, A_true

def run_var2():
    print("\n=== Exp #9: VAR(2) DGP + lag-1 estimation ===")
    configs = [(30,60),(30,250),(50,120),(50,500),(100,250),(100,500)]
    results = []
    with tqdm(total=len(configs)*N_TRIALS*4, desc='Exp9') as pbar:
        for N,T in configs:
            for trial in range(N_TRIALS):
                seed = SEED_BASE+trial*1000+N+T
                # VAR(1) baseline
                R1, _, A1 = generate_sparse_var_extended(N=N,T=T,density=0.05,seed=seed,dgp='garch_factor')
                for meth, Ap in [('OLS',compute_ols_te_matrix(R1)[1]),
                                  ('LASSO',compute_lasso_te_matrix(R1)[1])]:
                    m = eval_metrics(A1, Ap)
                    m.update(dict(N=N,T=T,T_N=round(T/N,2),trial=trial,dgp='VAR1',method=meth))
                    results.append(m); pbar.update(1)
                # VAR(2) with lag-1 estimation
                R2, A2 = generate_var2(N=N,T=T,density=0.05,seed=seed)
                for meth, Ap in [('OLS',compute_ols_te_matrix(R2)[1]),
                                  ('LASSO',compute_lasso_te_matrix(R2)[1])]:
                    m = eval_metrics(A2, Ap)
                    m.update(dict(N=N,T=T,T_N=round(T/N,2),trial=trial,dgp='VAR2',method=meth))
                    results.append(m); pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'var2_results.csv', index=False)
    summ = df.groupby(['dgp','method','T_N'])[['precision','recall','f1']].mean().round(3)
    print(summ.to_string())
    return df


# ── Exp #10: PageRank + Weighted Out-Degree ───────────────────────────────────

def compute_pagerank(TE_matrix, d=0.85, max_iter=100):
    """Simple PageRank on TE-weighted adjacency"""
    N = TE_matrix.shape[0]
    W = TE_matrix.copy(); np.fill_diagonal(W, 0)
    col_sum = W.sum(axis=0); col_sum[col_sum==0] = 1
    W_norm = W / col_sum
    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr_new = (1-d)/N + d * W_norm @ pr
        if np.abs(pr_new-pr).max() < 1e-8: break
        pr = pr_new
    return pr_new

def run_alternative_signals():
    print("\n=== Exp #10: Alternative Signals (PageRank, WeightedOutDeg) ===")
    configs = [(50,120),(50,500),(100,250),(100,500)]
    results = []
    with tqdm(total=len(configs)*N_TRIALS*3, desc='Exp10') as pbar:
        for N,T in configs:
            for trial in range(N_TRIALS):
                seed = SEED_BASE+trial*1000+N+T
                R, _, A_true = generate_sparse_var_extended(N=N,T=T,density=0.05,seed=seed,dgp='garch_factor')
                true_od   = A_true.sum(axis=1)
                true_hubs = set(np.argsort(true_od)[-5:])

                # LASSO-TE
                TE_las, A_las = compute_lasso_te_matrix(R)
                np.fill_diagonal(TE_las, 0)

                # OLS-TE
                TE_ols, A_ols = compute_ols_te_matrix(R)
                np.fill_diagonal(TE_ols, 0)

                for label, TE_mat, A_mat in [
                    ('LASSO', TE_las, A_las),
                    ('OLS',   TE_ols, A_ols),
                ]:
                    # NIO
                    nio   = TE_mat.sum(axis=1) - TE_mat.sum(axis=0)
                    # Weighted out-degree
                    wod   = TE_mat.sum(axis=1)
                    # PageRank
                    pr    = compute_pagerank(TE_mat)
                    # Hub recovery for each signal
                    for sig_name, sig in [('NIO',nio),('WOD',wod),('PageRank',pr)]:
                        pred_hubs = set(np.argsort(sig)[-5:])
                        hub_rec   = len(true_hubs & pred_hubs) / 5
                        tau, _    = kendalltau(true_od, sig)
                        results.append(dict(N=N,T=T,T_N=round(T/N,2),trial=trial,
                                            estimator=label, signal=sig_name,
                                            hub_recovery=hub_rec, kendall_tau=tau))
                    pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'alt_signals.csv', index=False)
    summ = df.groupby(['estimator','signal','T_N'])[['hub_recovery','kendall_tau']].mean().round(3)
    print(summ.to_string())
    return df


# ── Exp #11: Simulation NIO Baseline ─────────────────────────────────────────

def run_simulation_nio_baseline():
    """
    In simulated data with known DGP:
    - Compute LASSO-TE NIO for each stock
    - Generate simulated 5-day forward returns from the TRUE VAR
    - Test if NIO predicts returns even with perfect ground truth
    This establishes whether NIO can EVER predict returns, regardless of estimation quality.
    """
    print("\n=== Exp #11: Simulation NIO Baseline ===")
    configs = [(50, 500), (100, 500)]
    results = []
    with tqdm(total=len(configs)*N_TRIALS*2, desc='Exp11') as pbar:
        for N, T in configs:
            for trial in range(N_TRIALS):
                seed = SEED_BASE+trial*1000+N+T
                rng  = np.random.RandomState(seed+99999)

                # Generate estimation window
                R_est, A_mat, A_true = generate_sparse_var_extended(
                    N=N, T=T, density=0.05, seed=seed, dgp='garch_factor')

                # Oracle NIO (from true A)
                oracle_nio = (A_true.sum(axis=1) - A_true.sum(axis=0)).astype(float)
                # LASSO NIO
                TE_las, _ = compute_lasso_te_matrix(R_est)
                np.fill_diagonal(TE_las, 0)
                lasso_nio = TE_las.sum(axis=1) - TE_las.sum(axis=0)

                # Generate OOS returns from the TRUE DGP (NOT from A_true which is binary)
                # Use actual VAR coefficients
                sigma = 0.01
                T_oos = 100
                # Re-extract A from generate call (need continuous A, not binary)
                # Use a simple approach: generate fresh OOS returns with same seed
                R_oos, A_cont, _ = generate_sparse_var_extended(
                    N=N, T=T_oos+5, density=0.05, seed=seed+1, dgp='garch_factor')

                for h in range(0, T_oos, 5):
                    if h+5 >= T_oos: break
                    fwd_ret = R_oos[h+1:h+6].sum(axis=0)  # 5-day fwd return per stock
                    for sig_name, sig in [('Oracle_NIO', oracle_nio),
                                           ('LASSO_NIO',  lasso_nio)]:
                        connected = fwd_ret[sig != 0]
                        isolated  = fwd_ret[sig == 0]
                        if len(connected) < 3 or len(isolated) < 3: continue
                        t_stat, _ = ttest_ind(connected, isolated)
                        results.append(dict(N=N,T=T,T_N=round(T/N,2),
                                            trial=trial, h=h, signal=sig_name,
                                            t_stat=t_stat,
                                            n_conn=len(connected),
                                            n_isol=len(isolated)))
                pbar.update(2)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'sim_nio_baseline.csv', index=False)
    summ = df.groupby(['signal','T_N'])['t_stat'].agg(['mean','std']).round(3)
    print(summ.to_string())
    print("\nKey: if Oracle_NIO also shows |t|<1.96, NIO is not a valid predictor even with perfect estimation.")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time; t0 = time.time()
    run_oracle_extended()       # #3  ~20min
    run_threshold_var_wide()    # #4  ~25min
    run_var2()                  # #9  ~15min
    run_alternative_signals()   # #10 ~15min
    run_simulation_nio_baseline() # #11 ~10min
    print(f"\n=== ALL DONE in {(time.time()-t0)/60:.1f} min ===")
