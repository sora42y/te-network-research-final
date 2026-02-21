"""
LASSO vs Rolling OLS - Simulation Study
Paper Part 1: 在已知稀疏 VAR ground truth 上对比两种方法的网络恢复能力

设定：
- N 只股票，T 个时间步
- Ground truth：稀疏 VAR(1)，真实连接密度 ~5%
- 比较指标：F1 score, Precision, Recall, Edge Recovery Rate
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Extended DGP (GARCH + Factor structure)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extended_dgp import generate_sparse_var_extended

OUTPUT_DIR = Path(r"C:\Users\soray\.openclaw\workspace\te_network_research\results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Step 1: Generate Sparse VAR(1) Data
# ============================================================================

def generate_sparse_var(N=50, T=500, density=0.05, seed=42):
    """
    生成稀疏 VAR(1) 数据
    
    r_t = A * r_{t-1} + ε_t
    
    A: N×N 系数矩阵，只有 density% 的元素非零
    确保平稳性（最大特征值 < 1）
    """
    rng = np.random.RandomState(seed)
    
    # 生成稀疏系数矩阵
    A = np.zeros((N, N))
    n_edges = int(N * N * density)
    
    # 随机选择非零位置（不含对角线）
    off_diag = [(i, j) for i in range(N) for j in range(N) if i != j]
    edge_idx = rng.choice(len(off_diag), size=n_edges, replace=False)
    
    for idx in edge_idx:
        i, j = off_diag[idx]
        A[i, j] = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
    
    # 确保平稳性：缩放使最大特征值 < 0.9
    eigvals = np.abs(np.linalg.eigvals(A))
    if eigvals.max() > 0.9:
        A = A * (0.9 / eigvals.max())
    
    # 生成时间序列
    R = np.zeros((T, N))
    R[0] = rng.normal(0, 0.01, N)
    
    sigma = 0.01  # 日收益率波动率量级
    for t in range(1, T):
        R[t] = A @ R[t-1] + rng.normal(0, sigma, N)
    
    # Ground truth 邻接矩阵（二值）
    A_true = (A != 0).astype(int)
    
    return R, A, A_true

# ============================================================================
# Step 2: LASSO-TE Network Estimation
# ============================================================================

def compute_lasso_te_matrix(R):
    """
    LASSO-TE: σ² 比值路线
    
    对每只股票 i：
      restricted:   OLS of r_{i,t} on r_{i,t-1} only  → σ²_res
      unrestricted: LASSO (BIC) of r_{i,t} on [r_{i,t-1}, all r_{j≠i,t-1}] → σ²_full
    
    TE(j→i):
      - 若 j 的系数非零 → TE = 0.5 * ln(σ²_res / σ²_full)
      - 否则 → TE = 0
    
    返回 TE 矩阵 (N×N) 和 邻接矩阵 (0/1)
    """
    T, N = R.shape
    R_t   = R[1:]    # (T-1, N)
    R_lag = R[:-1]   # (T-1, N)
    T_eff = T - 1

    TE = np.zeros((N, N))
    A  = np.zeros((N, N), dtype=int)

    scaler = StandardScaler()

    for i in range(N):
        y = R_t[:, i]

        # --- Restricted: OLS on own lag only (fit_intercept=True，与 LASSO 一致) ---
        x_own = R_lag[:, i].reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(x_own, y)
        sigma2_res = np.mean((y - reg.predict(x_own)) ** 2)

        # --- Unrestricted: LASSO (BIC) on own lag + all others ---
        # 列顺序：[own_lag, other_lags...]
        other_idx = [j for j in range(N) if j != i]
        X_full_raw = np.column_stack([R_lag[:, i], R_lag[:, other_idx]])
        X_full = scaler.fit_transform(X_full_raw)

        lasso = LassoLarsIC(criterion='bic', max_iter=500)
        try:
            lasso.fit(X_full, y)
        except ValueError:
            try:
                lasso = LassoLarsIC(criterion='bic', max_iter=500,
                                    noise_variance=sigma2_res)
                lasso.fit(X_full, y)
            except Exception:
                continue
        except Exception:
            continue

        coef_scaled = lasso.coef_          # [own, j0, j1, ...]
        nonzero_mask = (coef_scaled != 0)

        y_hat_full = lasso.predict(X_full)
        sigma2_full = np.mean((y - y_hat_full) ** 2)

        # LASSO regularization bias：sigma2_full > sigma2_res 时跳过
        if sigma2_full < 1e-12 or sigma2_res <= sigma2_full:
            continue

        # --- CRITICAL FIX: marginal contribution per j (leave-j-out) ---
        # selected: 所有被 LASSO 选中的列 index（含 own lag col 0）
        selected = np.where(nonzero_mask)[0]

        for k, j in enumerate(other_idx):
            if not nonzero_mask[k + 1]:
                continue

            # leave-j-out: 用 selected 中去掉 j 对应列的子集重新拟合
            cols_without_j = [c for c in selected if c != k + 1]
            if len(cols_without_j) == 0:
                # j 是唯一被选中的 predictor → restricted 就是 own-lag-only
                sigma2_drop_j = sigma2_res
            else:
                X_drop_j = X_full[:, cols_without_j]
                reg_drop = LinearRegression(fit_intercept=True)
                reg_drop.fit(X_drop_j, y)
                sigma2_drop_j = np.mean((y - reg_drop.predict(X_drop_j)) ** 2)

            if sigma2_drop_j <= sigma2_full:
                continue  # j 没有边际贡献

            TE[i, j] = 0.5 * np.log(sigma2_drop_j / sigma2_full)
            A[i, j] = 1

    np.fill_diagonal(TE, 0)
    np.fill_diagonal(A, 0)
    return TE, A

# ============================================================================
# Step 3: Rolling OLS-TE Network Estimation (σ² ratio, pairwise)
# ============================================================================

def compute_ols_te_matrix(R, t_threshold=2.0):
    """
    OLS-TE: 对每对 (i, j)，pairwise σ² 比值
    
    restricted:   OLS of r_{i,t} on r_{i,t-1} only
    unrestricted: OLS of r_{i,t} on [r_{i,t-1}, r_{j,t-1}]
    
    TE(j→i) = 0.5 * ln(σ²_res / σ²_full)，仅当 t-stat(γ_j) > threshold
    """
    T, N = R.shape
    R_t   = R[1:]
    R_lag = R[:-1]

    TE = np.zeros((N, N))
    A  = np.zeros((N, N), dtype=int)

    ones = np.ones((T - 1, 1))

    for i in range(N):
        y = R_t[:, i]

        # Restricted: own lag only
        X_res = np.column_stack([ones, R_lag[:, i]])
        b_res = np.linalg.lstsq(X_res, y, rcond=None)[0]
        sigma2_res = np.mean((y - X_res @ b_res) ** 2)

        for j in range(N):
            if i == j:
                continue

            # Unrestricted: own lag + j's lag
            X_full = np.column_stack([ones, R_lag[:, i], R_lag[:, j]])
            b_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            resid_full = y - X_full @ b_full
            sigma2_full = np.mean(resid_full ** 2)

            if sigma2_full < 1e-12 or sigma2_res < sigma2_full:
                continue

            # t-stat for γ_j (coefficient on j's lag)
            dof = T - 1 - 3
            s2 = np.sum(resid_full ** 2) / dof
            try:
                cov = s2 * np.linalg.inv(X_full.T @ X_full)
                se_j = np.sqrt(cov[2, 2])
                t_stat = abs(b_full[2] / (se_j + 1e-12))
            except:
                t_stat = 0

            if t_stat > t_threshold:
                TE[i, j] = 0.5 * np.log(sigma2_res / sigma2_full)
                A[i, j] = 1

    np.fill_diagonal(TE, 0)
    np.fill_diagonal(A, 0)
    return TE, A

# ============================================================================
# Step 4: Evaluation Metrics
# ============================================================================

def evaluate(A_true, A_pred, name=""):
    """计算 F1, Precision, Recall"""
    y_true = A_true.flatten()
    y_pred = A_pred.flatten()
    
    # 去掉对角线
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool).flatten()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    density_true = y_true.mean()
    density_pred = y_pred.mean()
    
    return {
        'method': name,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'density_true': density_true,
        'density_pred': density_pred
    }

# ============================================================================
# Step 5: Monte Carlo Simulation
# ============================================================================

def run_simulation(N_list=[30, 50], T_list=[120, 250, 500],
                   density=0.05, n_trials=10, dgp='gaussian'):
    """
    Monte Carlo: 不同 N, T 组合下 LASSO-TE vs OLS-TE 的表现
    dgp: 'gaussian' | 'garch' | 'garch_factor'
    """
    results = []

    configs = [(N, T) for N in N_list for T in T_list]
    total = len(configs) * n_trials

    print(f"Running {total} trials ({len(configs)} configs × {n_trials} trials, dgp={dgp})...")

    with tqdm(total=total) as pbar:
        for N, T in configs:
            for trial in range(n_trials):
                seed = trial * 100 + N + T

                R, A_true_coef, A_true = generate_sparse_var_extended(
                    N=N, T=T, density=density, seed=seed, dgp=dgp
                )

                # LASSO-TE
                _, A_lasso = compute_lasso_te_matrix(R)
                res_lasso = evaluate(A_true, A_lasso, "LASSO-TE")
                res_lasso.update({'N': N, 'T': T, 'trial': trial,
                                  'T_N_ratio': T/N, 'dgp': dgp})
                results.append(res_lasso)

                # OLS-TE
                _, A_ols = compute_ols_te_matrix(R)
                res_ols = evaluate(A_true, A_ols, "OLS-TE")
                res_ols.update({'N': N, 'T': T, 'trial': trial,
                                'T_N_ratio': T/N, 'dgp': dgp})
                results.append(res_ols)

                pbar.update(1)

    return pd.DataFrame(results)

# ============================================================================
# Step 6: Visualization
# ============================================================================

def _lowess_smooth(x, y, frac=0.5):
    """Simple LOWESS smoother using numpy (no statsmodels dependency)."""
    from scipy.ndimage import uniform_filter1d
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # weighted moving average as lightweight smoother
    window = max(3, int(len(xs) * frac))
    if window % 2 == 0:
        window += 1
    ys_smooth = uniform_filter1d(ys.astype(float), size=window, mode='nearest')
    return xs, ys_smooth


def plot_results(df, output_path):
    """Figure: LASSO vs OLS — scatter + smoothed trend, clean layout."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    colors = {'LASSO-TE': '#2196F3', 'OLS-TE': '#FF5722'}
    markers = {'LASSO-TE': 'o', 'OLS-TE': 's'}

    # Per-(N,T,method) means across trials
    summary = (df.groupby(['N', 'T', 'method', 'T_N_ratio'])
                 [['f1', 'precision', 'recall']]
                 .mean().reset_index())

    # ── Panel A: F1 vs T/N  (scatter + trend, N=100 only for clarity) ──
    ax_a = fig.add_subplot(gs[0, :2])
    for method in ['LASSO-TE', 'OLS-TE']:
        sub = summary[(summary['method'] == method) & (summary['N'] == 100)]
        ax_a.scatter(sub['T_N_ratio'], sub['f1'],
                     color=colors[method], marker=markers[method],
                     s=55, alpha=0.85, zorder=3, label=f'{method} (N=100)')
        if len(sub) >= 3:
            xs, ys = _lowess_smooth(sub['T_N_ratio'].values, sub['f1'].values)
            ax_a.plot(xs, ys, color=colors[method], lw=2, alpha=0.7)
    ax_a.axvline(5, color='grey', ls='--', lw=1.2, alpha=0.6, label='T/N = 5 threshold')
    ax_a.set_xlabel('T/N Ratio', fontsize=11)
    ax_a.set_ylabel('F1 Score', fontsize=11)
    ax_a.set_title('A. Network Recovery F1 vs T/N  (N = 100)', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.25)
    ax_a.set_ylim(0, 1)

    # ── Panel B: Precision-Recall scatter (all N) ──
    ax_b = fig.add_subplot(gs[0, 2])
    for method in ['LASSO-TE', 'OLS-TE']:
        sub = summary[summary['method'] == method]
        ax_b.scatter(sub['recall'], sub['precision'],
                     c=colors[method], marker=markers[method],
                     label=method, s=45, alpha=0.65,
                     edgecolors='white', linewidth=0.4)
    ax_b.set_xlabel('Recall', fontsize=11)
    ax_b.set_ylabel('Precision', fontsize=11)
    ax_b.set_title('B. Precision–Recall', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.25)
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)

    # ── Panels C/D/E: scatter + smooth for F1, Precision, Recall vs T/N ──
    for idx, (metric, title) in enumerate([
            ('f1', 'F1 Score'),
            ('precision', 'Precision'),
            ('recall', 'Recall')]):
        ax = fig.add_subplot(gs[1, idx])
        for method in ['LASSO-TE', 'OLS-TE']:
            sub = summary[summary['method'] == method]
            # scatter: one point per (N,T) cell
            ax.scatter(sub['T_N_ratio'], sub[metric],
                       color=colors[method], marker=markers[method],
                       s=30, alpha=0.55, zorder=3)
            # smooth trend across all N
            if len(sub) >= 3:
                xs, ys = _lowess_smooth(sub['T_N_ratio'].values,
                                        sub[metric].values, frac=0.55)
                ax.plot(xs, ys, color=colors[method], lw=2,
                        label=method, alpha=0.85)
        ax.axvline(5, color='grey', ls='--', lw=1.1, alpha=0.55)
        ax.set_xlabel('T/N Ratio', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{"CDE"[idx]}. {title} vs T/N', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1)

    plt.suptitle(
        'LASSO-TE vs OLS-TE: Sparse VAR Network Recovery\n'
        '(True density = 5%, 10 Monte Carlo trials per cell | dashed line: T/N = 5)',
        fontsize=12, fontweight='bold', y=1.01)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LASSO vs OLS Simulation Study — Extended DGP")
    print("=" * 60)

    # Quick sanity check with garch_factor
    print("\n[Quick sanity check — garch_factor DGP]")
    R_test, A_true_coef, A_true = generate_sparse_var_extended(
        N=30, T=250, density=0.05, seed=0, dgp='garch_factor'
    )
    print(f"Data shape: {R_test.shape}")
    print(f"True edges: {A_true.sum()} / {30*29} possible = {A_true.mean():.1%} density")

    # Full Monte Carlo: three DGPs
    all_results = []
    for dgp in ['gaussian', 'garch', 'garch_factor']:
        print(f"\n[Monte Carlo — dgp={dgp}]")
        df_dgp = run_simulation(
            N_list=[30, 50, 100],
            T_list=[60, 120, 250, 500],
            density=0.05,
            n_trials=10,
            dgp=dgp
        )
        all_results.append(df_dgp)

    df_results = pd.concat(all_results, ignore_index=True)

    # Save combined results
    out_csv = OUTPUT_DIR / 'simulation_results_extended.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ---- Table 2 (extended): DGP × (N,T) × Method ----
    print("\n" + "=" * 70)
    print("TABLE 2 — Extended (gaussian / garch / garch_factor)")
    print("=" * 70)
    summary = (
        df_results
        .groupby(['dgp', 'N', 'T', 'method'])[['f1', 'precision', 'recall']]
        .mean()
        .round(3)
    )
    print(summary.to_string())

    # Save Table 2 as CSV and LaTeX
    summary_flat = summary.reset_index()
    summary_flat['T_N'] = (summary_flat['T'] / summary_flat['N']).round(1)
    summary_flat.to_csv(OUTPUT_DIR / 'table2_extended.csv', index=False)

    # LaTeX snippet
    tex_lines = []
    tex_lines.append(r"\begin{tabular}{llrrrrccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"DGP & Method & $N$ & $T$ & $T/N$ & & F1 & Prec & Rec \\")
    tex_lines.append(r"\midrule")
    prev_dgp = None
    for _, row in summary_flat.sort_values(['dgp','N','T','method']).iterrows():
        dgp_label = row['dgp'] if row['dgp'] != prev_dgp else ''
        prev_dgp = row['dgp']
        tex_lines.append(
            f"{dgp_label} & {row['method']} & {int(row['N'])} & {int(row['T'])} "
            f"& {row['T_N']} & & {row['f1']:.3f} & {row['precision']:.3f} & {row['recall']:.3f} \\\\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    with open(OUTPUT_DIR / 'table2_extended.tex', 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f"\nSaved: {OUTPUT_DIR / 'table2_extended.tex'}")

    # ---- Quick comparison: gaussian vs garch_factor at key T/N ----
    print("\n" + "=" * 70)
    print("KEY COMPARISON: gaussian vs garch_factor (OLS-TE precision)")
    print("=" * 70)
    comp = (
        df_results[df_results['method'] == 'OLS-TE']
        .groupby(['dgp', 'T_N_ratio'])['precision']
        .mean()
        .unstack('dgp')
        .round(3)
    )
    print(comp.to_string())

    # Plot (extended, one panel per DGP)
    plot_results(df_results[df_results['dgp'] == 'gaussian'],
                 OUTPUT_DIR / 'simulation_figure_gaussian.png')
    plot_results(df_results[df_results['dgp'] == 'garch_factor'],
                 OUTPUT_DIR / 'simulation_figure_garch_factor.png')

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
