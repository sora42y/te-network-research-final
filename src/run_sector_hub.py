"""run_sector_hub.py — Design C: Sector Hub Identity Test (Billio-style), 100 trials"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau, ttest_ind
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import sys

BASE = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research')
sys.path.insert(0, str(BASE))
from te_core import compute_linear_te_matrix

OUTPUT    = BASE / 'results'
SEED_BASE = 42
N_TRIALS  = 100
N_HUB     = 5
N_REST    = 45
N         = N_HUB + N_REST
T         = 500

def make_A_hub(rng):
    A = np.zeros((N, N))
    off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]
    # background edges among non-hub (density 5%)
    normal_pairs = [(i,j) for i,j in off_d if i >= N_HUB and j >= N_HUB]
    for k in rng.choice(len(normal_pairs), int(len(normal_pairs)*0.05), replace=False):
        i,j = normal_pairs[k]; A[i,j] = rng.uniform(0.05,0.10)*rng.choice([-1,1])
    # hub out-edges (density 30% hub→rest)
    hub_to_rest = [(i,j) for i in range(N_HUB) for j in range(N_HUB,N)]
    for k in rng.choice(len(hub_to_rest), int(len(hub_to_rest)*0.30), replace=False):
        i,j = hub_to_rest[k]; A[i,j] = rng.uniform(0.08,0.12)*rng.choice([-1,1])
    ev = np.abs(np.linalg.eigvals(A))
    if ev.max() > 0.85: A *= 0.85/ev.max()
    return A

results = []
sigma = 0.01

with tqdm(total=N_TRIALS*2, desc='SectorHub') as pbar:
    for trial in range(N_TRIALS):
        seed = SEED_BASE + trial*3571
        rng  = np.random.RandomState(seed)
        A = make_A_hub(rng)
        A_true = (A != 0).astype(int)
        true_out = A_true.sum(axis=1)

        R = np.zeros((T,N)); R[0] = rng.normal(0,sigma,N)
        for t in range(1,T):
            R[t] = A@R[t-1] + rng.normal(0,sigma,N)

        for meth, fn in [('OLS', compute_ols_te_matrix),
                          ('LASSO', compute_lasso_te_matrix)]:
            try:
                TE_mat, A_pred = fn(R)
                np.fill_diagonal(TE_mat, 0)
                est_out_count = A_pred.sum(axis=1)
                est_out_te    = TE_mat.sum(axis=1)
                hub_above_median_count = np.mean(est_out_count[:N_HUB] > np.median(est_out_count[N_HUB:]))
                hub_above_median_te    = np.mean(est_out_te[:N_HUB]    > np.median(est_out_te[N_HUB:]))
                tau_count, _ = kendalltau(true_out, est_out_count)
                tau_te,    _ = kendalltau(true_out, est_out_te)
                nio = est_out_te - TE_mat.sum(axis=0)
                t_stat, _ = ttest_ind(nio[:N_HUB], nio[N_HUB:])
                results.append(dict(trial=trial, method=meth,
                    hub_above_median_count=hub_above_median_count,
                    hub_above_median_te=hub_above_median_te,
                    kendall_tau_count=tau_count,
                    kendall_tau_te=tau_te,
                    hub_nio_tstat=t_stat))
            except Exception as e:
                print(f"  {meth} trial {trial}: {e}")
            pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(OUTPUT / 'sector_hub_test.csv', index=False)

print(f"\nN_hub={N_HUB}, N_rest={N_REST}, T={T}")
print(f"Hub out-edge density: 30% vs background 5%")
for meth in ['OLS','LASSO']:
    sub = df[df['method']==meth]
    print(f"\n{meth}:")
    print(f"  Hub > median rest (binary edges): {sub['hub_above_median_count'].mean():.3f}")
    print(f"  Hub > median rest (TE weight):    {sub['hub_above_median_te'].mean():.3f}")
    print(f"  Kendall tau (out-degree):         {sub['kendall_tau_count'].mean():.3f}")
    print(f"  Kendall tau (TE weight):          {sub['kendall_tau_te'].mean():.3f}")
    print(f"  Hub NIO t-stat vs rest:           {sub['hub_nio_tstat'].mean():.3f}  "
          f"(|t|>1.96: {(sub['hub_nio_tstat'].abs()>1.96).mean():.0%})")

print("\nDone.")


