"""
run_aggregate_recovery.py
Design A: Two-regime clean split (main table)
Design B: Rolling window (robustness)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent))
from te_core import compute_linear_te_matrix

OUTPUT = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research\results')
SEED_BASE = 42
N_TRIALS  = 100

# ── DGP helpers ───────────────────────────────────────────────────────────────

def make_A(N, density, scale, rng):
    A = np.zeros((N, N))
    off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]
    n_edges = int(N * N * density)
    idx = rng.choice(len(off_d), n_edges, replace=False)
    for k in idx:
        i, j = off_d[k]
        A[i,j] = rng.uniform(0.05, scale) * rng.choice([-1,1])
    ev = np.abs(np.linalg.eigvals(A))
    if ev.max() > 0.85: A *= 0.85 / ev.max()
    return A

def generate_two_regime(N, T, d1, d2, s1, s2, seed):
    """Abrupt switch at T//2"""
    rng = np.random.RandomState(seed)
    sigma = 0.01
    A1 = make_A(N, d1, s1, rng)
    A2 = make_A(N, d2, s2, rng)
    true_d1 = (A1 != 0).sum() / (N*(N-1))
    true_d2 = (A2 != 0).sum() / (N*(N-1))

    R = np.zeros((T, N)); R[0] = rng.normal(0, sigma, N)
    half = T // 2
    for t in range(1, T):
        A = A1 if t < half else A2
        R[t] = A @ R[t-1] + rng.normal(0, sigma, N)
    return R, A1, A2, true_d1, true_d2

def estimated_density(A_pred):
    N = A_pred.shape[0]
    return A_pred.sum() / (N*(N-1))


# ════════════════════════════════════════════════════════════════════════════
# DESIGN A: Two-regime clean split
# ════════════════════════════════════════════════════════════════════════════

def run_design_A():
    print("\n=== Design A: Two-regime clean split ===")
    # Parameters as specified
    N   = 50
    T   = 500
    d1, d2 = 0.03, 0.12          # regime 1: sparse; regime 2: dense
    s1, s2 = 0.08, 0.12          # coefficient scale

    true_delta_d = d2 - d1       # = 0.09, ground truth change

    results = []
    with tqdm(total=N_TRIALS * 2, desc='DesignA') as pbar:
        for trial in range(N_TRIALS):
            seed = SEED_BASE + trial * 7919
            R, A1, A2, td1, td2 = generate_two_regime(N, T, d1, d2, s1, s2, seed)

            half = T // 2
            R1, R2 = R[:half], R[half:]

            for meth, fn in [('OLS', compute_ols_te_matrix),
                              ('LASSO', compute_lasso_te_matrix)]:
                try:
                    _, Ap1 = fn(R1)
                    _, Ap2 = fn(R2)
                    ed1 = estimated_density(Ap1)
                    ed2 = estimated_density(Ap2)
                    delta_est = ed2 - ed1
                    correct_direction = int(delta_est > 0)   # d2 > d1 always
                    results.append(dict(
                        trial=trial, method=meth,
                        true_d1=td1, true_d2=td2,
                        est_d1=ed1,  est_d2=ed2,
                        true_delta=td2-td1,
                        est_delta=delta_est,
                        correct_direction=correct_direction,
                    ))
                except Exception as e:
                    print(f"  {meth} trial {trial}: {e}")
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'aggregate_recovery_A.csv', index=False)

    print(f"\n  True density: regime1={d1:.0%}, regime2={d2:.0%}, delta={d2-d1:.0%}")
    for meth in ['OLS','LASSO']:
        sub = df[df['method']==meth]
        det_rate = sub['correct_direction'].mean()
        mean_ed1 = sub['est_d1'].mean()
        mean_ed2 = sub['est_d2'].mean()
        mean_delta = sub['est_delta'].mean()
        corr, pval = pearsonr(sub['true_delta'], sub['est_delta'])
        print(f"\n  {meth}:")
        print(f"    Detection rate (Pr[d̂2 > d̂1]): {det_rate:.3f}")
        print(f"    Est density:  regime1={mean_ed1:.4f}, regime2={mean_ed2:.4f}")
        print(f"    Est delta: {mean_delta:.4f}  (true delta: {d2-d1:.4f})")
        print(f"    Corr(true_delta, est_delta): r={corr:.3f}, p={pval:.4f}")

    # Figure A: scatter est_delta vs true_delta + detection rate bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for meth, clr in [('OLS','#1976D2'), ('LASSO','#D32F2F')]:
        sub = df[df['method']==meth]
        ax.scatter(sub['true_delta'], sub['est_delta'],
                   alpha=0.4, s=20, color=clr, label=meth)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(d2-d1, color='grey', lw=0.8, ls='--', label='True delta')
    ax.set_xlabel('True $\\Delta d$ (regime2 - regime1)')
    ax.set_ylabel('Estimated $\\Delta d$')
    ax.set_title('Aggregate Density Change: Estimated vs True', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[1]
    for i, (meth, clr) in enumerate([('OLS','#1976D2'), ('LASSO','#D32F2F')]):
        sub = df[df['method']==meth]
        det = sub['correct_direction'].mean()
        ax.bar(i, det, color=clr, alpha=0.8, label=meth)
        ax.text(i, det+0.01, f'{det:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='grey', ls='--', lw=1, label='Random (0.5)')
    ax.set_xticks([0,1]); ax.set_xticklabels(['OLS','LASSO'])
    ax.set_ylabel('Detection Rate Pr[$\\hat{d}_2 > \\hat{d}_1$]')
    ax.set_ylim(0, 1.05)
    ax.set_title('Crisis Connectivity Detection Rate', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'figure_aggregate_A.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved aggregate_recovery_A.csv + figure_aggregate_A.png")
    return df


# ════════════════════════════════════════════════════════════════════════════
# DESIGN B: Rolling window (robustness)
# ════════════════════════════════════════════════════════════════════════════

def run_design_B():
    print("\n=== Design B: Rolling window robustness ===")
    N     = 50
    T     = 1000
    T_w   = 60    # window length (matches empirical Section 5)
    step  = 5
    # Crisis regime: t=400-600 (high connectivity)
    crisis_start, crisis_end = 400, 600
    d_normal, d_crisis = 0.03, 0.12
    s_normal, s_crisis = 0.08, 0.12

    # One representative trial for the figure
    rng = np.random.RandomState(SEED_BASE)
    A_normal = make_A(N, d_normal, s_normal, rng)
    A_crisis = make_A(N, d_crisis, s_crisis, rng)

    sigma = 0.01
    R = np.zeros((T, N)); R[0] = rng.normal(0, sigma, N)
    for t in range(1, T):
        A = A_crisis if crisis_start <= t < crisis_end else A_normal
        R[t] = A @ R[t-1] + rng.normal(0, sigma, N)

    # True density per time step
    true_d_ts = np.where(
        (np.arange(T) >= crisis_start) & (np.arange(T) < crisis_end),
        (A_crisis != 0).sum() / (N*(N-1)),
        (A_normal != 0).sum() / (N*(N-1))
    )

    # Rolling estimation
    windows = list(range(0, T - T_w, step))
    ols_dens, las_dens, true_dens, t_mids = [], [], [], []

    for t0 in tqdm(windows, desc='DesignB rolling'):
        t1 = t0 + T_w
        chunk = R[t0:t1]
        try:
            _, Ap_ols  = compute_linear_te_matrix(chunk, method="ols", t_threshold=2.0)
            _, Ap_las  = compute_linear_te_matrix(chunk, method="lasso")
            ols_dens.append(estimated_density(Ap_ols))
            las_dens.append(estimated_density(Ap_las))
        except:
            ols_dens.append(np.nan)
            las_dens.append(np.nan)
        true_dens.append(true_d_ts[t0 + T_w//2])
        t_mids.append(t0 + T_w//2)

    roll_df = pd.DataFrame(dict(t=t_mids, true_d=true_dens,
                                 ols_d=ols_dens, las_d=las_dens))
    roll_df.to_csv(OUTPUT / 'aggregate_recovery_B.csv', index=False)

    # MC: correlation across 30 trials
    mc_results = []
    for trial in range(30):
        seed = SEED_BASE + trial * 1337
        rng2 = np.random.RandomState(seed)
        A_n = make_A(N, d_normal, s_normal, rng2)
        A_c = make_A(N, d_crisis, s_crisis, rng2)
        R2 = np.zeros((T,N)); R2[0] = rng2.normal(0,sigma,N)
        for t in range(1,T):
            A = A_c if crisis_start<=t<crisis_end else A_n
            R2[t] = A@R2[t-1] + rng2.normal(0,sigma,N)
        od, ld, td = [], [], []
        for t0 in windows:
            chunk = R2[t0:t0+T_w]
            try:
                _, Ao = compute_linear_te_matrix(chunk, method="ols", t_threshold=2.0)
                _, Al = compute_linear_te_matrix(chunk, method="lasso")
                od.append(estimated_density(Ao))
                ld.append(estimated_density(Al))
            except:
                od.append(np.nan); ld.append(np.nan)
            td.append(true_d_ts[t0+T_w//2])
        td = np.array(td); od = np.array(od); ld = np.array(ld)
        mask = ~(np.isnan(od)|np.isnan(ld))
        if mask.sum() > 5:
            mc_results.append(dict(
                trial=trial,
                corr_ols=pearsonr(td[mask], od[mask])[0],
                corr_las=pearsonr(td[mask], ld[mask])[0],
            ))
    mc_df = pd.DataFrame(mc_results)
    print(f"\n  Rolling window MC (30 trials):")
    print(f"    OLS-density corr with true: {mc_df['corr_ols'].mean():.3f} +/- {mc_df['corr_ols'].std():.3f}")
    print(f"    LASSO-density corr with true: {mc_df['corr_las'].mean():.3f} +/- {mc_df['corr_las'].std():.3f}")

    # Figure B
    fig, ax = plt.subplots(figsize=(12, 5))
    t_arr = np.array(t_mids)
    ax.plot(t_arr, true_dens, 'k-', lw=2, label='True density', zorder=5)
    ax.plot(t_arr, ols_dens,  color='#1976D2', lw=1.5, alpha=0.85, label='OLS-TE estimated')
    ax.plot(t_arr, las_dens,  color='#D32F2F', lw=1.5, alpha=0.85, label='LASSO-TE estimated')
    ax.axvspan(crisis_start, crisis_end, alpha=0.12, color='orange', label='Crisis regime')
    ax.set_xlabel('Time (trading days)')
    ax.set_ylabel('Network density')
    ax.set_title('Rolling Aggregate Density: True vs Estimated (N=50, $T_w$=60)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / 'figure_aggregate_B.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved aggregate_recovery_B.csv + figure_aggregate_B.png")
    return roll_df, mc_df


# ── Sector Hub Test (Design C, Billio-style) ──────────────────────────────────

def run_sector_hub():
    print("\n=== Design C: Sector Hub Identity Test (Billio-style) ===")
    N_hub  = 5    # hub sector size
    N_rest = 45   # rest
    N      = N_hub + N_rest
    T      = 500
    n_trials = N_TRIALS

    results = []
    with tqdm(total=n_trials * 2, desc='SectorHub') as pbar:
        for trial in range(n_trials):
            seed = SEED_BASE + trial * 3571
            rng  = np.random.RandomState(seed)
            sigma = 0.01

            # Hub sector: 3x more out-edges to rest
            A = np.zeros((N, N))
            off_d = [(i,j) for i in range(N) for j in range(N) if i!=j]

            # Normal background edges (density 5% among non-hub pairs)
            normal_pairs = [(i,j) for i,j in off_d if i >= N_hub and j >= N_hub]
            for k in rng.choice(len(normal_pairs), int(len(normal_pairs)*0.05), replace=False):
                i,j = normal_pairs[k]
                A[i,j] = rng.uniform(0.05,0.10)*rng.choice([-1,1])

            # Hub out-edges: hub → rest, 3x density
            hub_to_rest = [(i,j) for i in range(N_hub) for j in range(N_hub,N)]
            for k in rng.choice(len(hub_to_rest), int(len(hub_to_rest)*0.30), replace=False):
                i,j = hub_to_rest[k]
                A[i,j] = rng.uniform(0.08,0.12)*rng.choice([-1,1])

            ev = np.abs(np.linalg.eigvals(A))
            if ev.max() > 0.85: A *= 0.85/ev.max()

            A_true = (A != 0).astype(int)
            true_out = A_true.sum(axis=1)
            true_hub_nodes = set(range(N_hub))

            # Returns
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

                    # Metric 1: are hub nodes ranked higher in estimated out-degree?
                    hub_rank_count = np.mean([
                        np.mean(est_out_count[:N_hub] > np.median(est_out_count[N_hub:]))
                    ])
                    hub_rank_te = np.mean([
                        np.mean(est_out_te[:N_hub] > np.median(est_out_te[N_hub:]))
                    ])

                    # Metric 2: Kendall tau between true and estimated out-degree
                    from scipy.stats import kendalltau
                    tau_count, _ = kendalltau(true_out, est_out_count)
                    tau_te,    _ = kendalltau(true_out, est_out_te)

                    # Metric 3: hub sector NIO vs rest t-test
                    from scipy.stats import ttest_ind
                    nio = est_out_te - TE_mat.sum(axis=0)
                    t_stat, _ = ttest_ind(nio[:N_hub], nio[N_hub:])

                    results.append(dict(
                        trial=trial, method=meth,
                        hub_rank_binary=hub_rank_count,
                        hub_rank_te=hub_rank_te,
                        kendall_tau_count=tau_count,
                        kendall_tau_te=tau_te,
                        hub_nio_tstat=t_stat,
                    ))
                except Exception as e:
                    print(f"  {meth} trial {trial}: {e}")
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / 'sector_hub_test.csv', index=False)

    print(f"\n  N_hub={N_hub}, N_rest={N_rest}, T={T}")
    print(f"  Hub out-edge density: 30% vs background 5%")
    for meth in ['OLS','LASSO']:
        sub = df[df['method']==meth]
        print(f"\n  {meth}:")
        print(f"    Hub > median rest (binary): {sub['hub_rank_binary'].mean():.3f}")
        print(f"    Hub > median rest (TE weight): {sub['hub_rank_te'].mean():.3f}")
        print(f"    Kendall tau (out-degree): {sub['kendall_tau_count'].mean():.3f}")
        print(f"    Kendall tau (TE weight):  {sub['kendall_tau_te'].mean():.3f}")
        print(f"    Hub NIO t-stat vs rest:   {sub['hub_nio_tstat'].mean():.3f} "
              f"(% |t|>1.96: {(sub['hub_nio_tstat'].abs()>1.96).mean():.3f})")

    print("  Saved sector_hub_test.csv")
    return df


if __name__ == '__main__':
    import time; t0 = time.time()
    dfA           = run_design_A()      # ~15 min (100 trials)
    dfB, mc_df    = run_design_B()      # ~10 min (30 trials)
    dfC           = run_sector_hub()    # ~15 min (100 trials)
    print(f"\n=== ALL DONE in {(time.time()-t0)/60:.1f} min ===")


