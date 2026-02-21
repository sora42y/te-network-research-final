"""
run_main_sim_100.py
重跑主模拟 Table 2，100 trials per cell，报告均值 + 95% CI。
覆盖 3 个 DGP × 全部 (N, T) 配置 × OLS/LASSO。
预计运行时间：~2-3 小时（取决于机器）。
进度实时写入 results/main_sim_100_progress.csv。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import t as tdist
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extended_dgp import generate_sparse_var_extended
from lasso_simulation import compute_lasso_te_matrix, compute_ols_te_matrix

OUTPUT    = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research\results')
SEED_BASE = 42
N_TRIALS  = 100
TOP_K     = 5
DENSITY   = 0.05

# ── Same (N,T) grid as Table 2 ────────────────────────────────────────────────
CONFIGS = [
    (20,  12), (20,  20), (20, 50), (20, 100), (20, 200),
    (50,  30), (50,  60), (50,125), (50, 250), (50, 500),
    (100, 60), (100,120), (100,250),(100,500), (100,1000),
]
DGPS = ['gaussian', 'garch', 'garch_factor']
METHODS = ['OLS', 'LASSO']

def eval_cell(A_true, A_pred):
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool).flatten()
    yt = A_true.flatten()[mask]
    yp = A_pred.flatten()[mask]
    true_od = A_true.sum(1); pred_od = A_pred.sum(1)
    true_h  = set(np.argsort(true_od)[-TOP_K:])
    pred_h  = set(np.argsort(pred_od)[-TOP_K:])
    return dict(
        precision    = precision_score(yt, yp, zero_division=0),
        recall       = recall_score(yt, yp, zero_division=0),
        f1           = f1_score(yt, yp, zero_division=0),
        hub_recovery = len(true_h & pred_h) / TOP_K,
        net_density  = yp.mean(),
    )

def ci95(arr):
    """95% CI half-width via t-distribution."""
    n = len(arr)
    if n < 2: return float('nan')
    se = np.std(arr, ddof=1) / np.sqrt(n)
    return tdist.ppf(0.975, df=n-1) * se

# ── Progress file ─────────────────────────────────────────────────────────────
prog_path = OUTPUT / 'main_sim_100_progress.csv'
done_keys = set()
if prog_path.exists():
    done_df = pd.read_csv(prog_path)
    for _, row in done_df.iterrows():
        done_keys.add((row['dgp'], row['method'], int(row['N']), int(row['T']), int(row['trial'])))
    print(f"Resuming: {len(done_df)} rows already done.")
else:
    done_df = pd.DataFrame()
    print("Starting fresh.")

# ── Main loop ─────────────────────────────────────────────────────────────────
total = len(DGPS) * len(METHODS) * len(CONFIGS) * N_TRIALS
done_n = len(done_keys)
results = []

with tqdm(total=total, initial=done_n, desc='Main 100-trial sim') as pbar:
    for dgp in DGPS:
        for N, T in CONFIGS:
            for trial in range(N_TRIALS):
                seed = SEED_BASE + trial * 10000 + N * 100 + T
                # Generate once, use for both methods
                key_ols   = (dgp, 'OLS',   N, T, trial)
                key_lasso = (dgp, 'LASSO', N, T, trial)
                if key_ols in done_keys and key_lasso in done_keys:
                    pbar.update(2); continue

                try:
                    R, _, A_true = generate_sparse_var_extended(
                        N=N, T=T, density=DENSITY, seed=seed, dgp=dgp)
                except Exception as e:
                    print(f"DGP error {dgp} N={N} T={T} trial={trial}: {e}")
                    pbar.update(2); continue

                base = dict(dgp=dgp, N=N, T=T, T_N=round(T/N, 3), trial=trial)

                for meth, key, fn in [
                    ('OLS',   key_ols,   compute_ols_te_matrix),
                    ('LASSO', key_lasso, compute_lasso_te_matrix),
                ]:
                    if key in done_keys:
                        pbar.update(1); continue
                    try:
                        _, A_pred = fn(R)
                        m = eval_cell(A_true, A_pred)
                    except Exception as e:
                        print(f"  {meth} error {dgp} N={N} T={T} trial={trial}: {e}")
                        m = dict(precision=np.nan, recall=np.nan, f1=np.nan,
                                 hub_recovery=np.nan, net_density=np.nan)
                    row = {**base, 'method': meth, **m}
                    results.append(row)
                    done_keys.add(key)
                    pbar.update(1)

                # Flush to disk every 20 new rows
                if len(results) >= 20:
                    chunk = pd.DataFrame(results)
                    chunk.to_csv(prog_path, mode='a',
                                 header=not prog_path.exists(), index=False)
                    results = []

# Final flush
if results:
    chunk = pd.DataFrame(results)
    chunk.to_csv(prog_path, mode='a',
                 header=not prog_path.exists(), index=False)

# ── Aggregate and produce summary table ───────────────────────────────────────
print("\nAggregating results...")
full = pd.read_csv(prog_path).dropna(subset=['precision'])

summ_rows = []
for dgp in DGPS:
    for meth in METHODS:
        for N, T in CONFIGS:
            sub = full[(full['dgp']==dgp)&(full['method']==meth)&
                       (full['N']==N)&(full['T']==T)]
            if len(sub) < 5: continue
            row = dict(dgp=dgp, method=meth, N=N, T=T, T_N=round(T/N,2),
                       n_trials=len(sub))
            for col in ['precision','recall','f1','hub_recovery','net_density']:
                vals = sub[col].dropna().values
                row[col]         = round(np.mean(vals), 4)
                row[f'{col}_ci'] = round(ci95(vals), 4)
            summ_rows.append(row)

summ = pd.DataFrame(summ_rows)
out_path = OUTPUT / 'main_sim_100_summary.csv'
summ.to_csv(out_path, index=False)
print(f"Summary saved: {out_path}")
print(f"Total trials used: {len(full)}")
print()

# Print Table 2 equivalent (GARCH+Factor, key T/N values)
key_tn = [0.6, 1.2, 2.5, 5.0, 10.0]
print("=== GARCH+Factor DGP — Key T/N values ===")
sub = summ[(summ['dgp']=='garch_factor')].copy()
sub['T_N_r'] = sub['T_N'].round(1)
for meth in ['OLS','LASSO']:
    print(f"\n  {meth}:")
    print(f"  {'T/N':>6} {'N':>4} {'T':>5} {'Precision':>12} {'Recall':>12} {'F1':>10} {'HubRec':>10}")
    ms = sub[sub['method']==meth].sort_values('T_N')
    for _, r in ms.iterrows():
        print(f"  {r['T_N']:>6.1f} {r['N']:>4} {r['T']:>5} "
              f"  {r['precision']:.3f}±{r['precision_ci']:.3f}"
              f"  {r['recall']:.3f}±{r['recall_ci']:.3f}"
              f"  {r['f1']:.3f}±{r['f1_ci']:.3f}"
              f"  {r['hub_recovery']:.3f}±{r['hub_recovery_ci']:.3f}")

print("\nDone.")
