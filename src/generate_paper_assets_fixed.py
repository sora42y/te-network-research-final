"""
Generate all paper figures and tables from existing data
Figure 2: TE value distribution
Figure 4: OLS vs LASSO density comparison
Figure 6: t-stat vs T/N ratio (THE KEY FIGURE)
Table 2: Simulation results
Table 3: LASSO network stats by window
Table 4: Window x N sweep t-stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(r"C:\Users\soray\.openclaw\workspace\te_network_research\results")
PAPER_DIR  = Path(r"C:\Users\soray\.openclaw\workspace\te_network_research\paper_assets")
PAPER_DIR.mkdir(exist_ok=True)

# ============================================================================
# Figure 2: TE value distribution (showing 1e-6 scale)
# ============================================================================
print("[Figure 2] TE value distribution...")

df_raw = pd.read_csv(OUTPUT_DIR / 'te_raw_timeseries.csv')
df_neu = pd.read_csv(OUTPUT_DIR / 'te_neutral_timeseries.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, df, label, color in [
    (axes[0], df_raw, 'Raw TE', '#1565c0'),
    (axes[1], df_neu, 'Factor-Neutral TE', '#e65100')
]:
    vals = df['mean_te'].dropna().values * 1e6
    ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(vals.mean(), color='black', ls='--', lw=1.5,
               label=f'Mean = {vals.mean():.2f}e-6')
    ax.set_xlabel('Mean TE Value (×10⁻⁶)', fontsize=11)
    ax.set_ylabel('Count (windows)', fontsize=11)
    ax.set_title(f'{label} Distribution\n(N=100, T=60, 339 windows)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution of Mean TE Values\n'
             'Near-zero mass confirms estimation noise dominance',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PAPER_DIR / 'figure2_te_distribution.pdf', bbox_inches='tight')
plt.savefig(PAPER_DIR / 'figure2_te_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2")

# ============================================================================
# Figure 4: OLS vs LASSO density comparison
# ============================================================================
print("[Figure 4] OLS vs LASSO density...")

# OLS density (from te_raw timeseries, density computed as percentile-based ~25%)
df_raw['date'] = pd.to_datetime(df_raw['formation_date'])
df_neu['date'] = pd.to_datetime(df_neu['formation_date'])

# LASSO density from window_sweep results (stored in sweep logs)
# Use per-window density from te_network_research results
# Approximate from lasso_nio_T60: fraction of nonzero NIO per date
df_lasso60 = pd.read_csv(OUTPUT_DIR / 'lasso_nio_T60.csv')
df_lasso60['date'] = pd.to_datetime(df_lasso60['formation_date'])
lasso_density_ts = df_lasso60.groupby('date')['nio'].apply(lambda x: (x != 0).mean())

fig, ax = plt.subplots(figsize=(12, 5))

# OLS: approximate density using 75th percentile (always ~25%)
ols_density = pd.Series(0.25, index=df_raw['date'])
ax.plot(df_raw['date'], ols_density, color='#e53935', lw=2,
        label='OLS-TE (75th pct threshold, mechanical 25%)', ls='--')

ax.plot(lasso_density_ts.index, lasso_density_ts.values,
        color='#1565c0', lw=1.5, alpha=0.8,
        label=f'LASSO-TE (data-driven, mean={lasso_density_ts.mean():.3f})')

ax.set_ylabel('Network Density', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_title('Network Density — OLS vs LASSO\n'
             'LASSO collapses density from 25% to 0.3%',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.35)

plt.tight_layout()
plt.savefig(PAPER_DIR / 'figure4_density_comparison.pdf', bbox_inches='tight')
plt.savefig(PAPER_DIR / 'figure4_density_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure4")

# ============================================================================
# Figure 5: Raw vs Factor-neutral TE time series
# ============================================================================
print("[Figure 5] Raw vs Factor-neutral TE...")

import shutil
shutil.copy(OUTPUT_DIR / 'te_timeseries.png', PAPER_DIR / 'figure5_factor_neutral.png')
print("  Copied figure5 from te_timeseries.png")

# ============================================================================
# Figure 6: KEY FIGURE — t-stat vs T/N ratio
# ============================================================================
print("[Figure 6] t-stat vs T/N ratio...")

# Collect all (N, T, BS_t, PS_t) from sweep logs
sweep_data = []

# N=30 (from sweep logs parsed manually)
n30_data = [
    (30, 60,  0.80,  None),
    (30, 120, 1.00,  None),
    (30, 180, 1.29,  None),
    (30, 252, 1.96,  None),
]
# N=50
n50_data = [
    (50, 60,  -1.01, 0.58),
    (50, 120, -1.06, None),
    (50, 180, -0.55, None),
    (50, 252, -0.33, None),
]
# N=70
n70_data = [
    (70, 60,  -1.45, None),
    (70, 120, -1.93, 0.91),
    (70, 180,  0.92, 2.08),
    (70, 252, -0.44, None),
]
# N=90
n90_data = [
    (90, 60,  -1.25, None),
    (90, 120, -1.63, -0.02),
    (90, 180, -0.18,  1.09),
    (90, 252, -0.75,  0.39),
]
# N=100
n100_data = [
    (100, 60,  -1.02, None),
    (100, 120, -2.09, None),
    (100, 180,  0.35, None),
    (100, 252, -0.65, 1.90),
]

for dataset in [n30_data, n50_data, n70_data, n90_data, n100_data]:
    for N, T, bs_t, ps_t in dataset:
        sweep_data.append({
            'N': N, 'T': T,
            'TN_ratio': T / N,
            'bs_t': bs_t,
            'ps_t': ps_t,
        })

df_sweep = pd.DataFrame(sweep_data)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_n = {30: '#e53935', 50: '#fb8c00', 70: '#43a047', 90: '#00838f', 100: '#1565c0'}
markers_n = {30: 'o', 50: 's', 70: '^', 90: 'D', 100: 'P'}

for ax, metric, title in [
    (axes[0], 'bs_t', 'Binary Split: Connected vs Isolated'),
    (axes[1], 'ps_t', 'Portfolio Sort: Long-Short Q5−Q1'),
]:
    for N in [30, 50, 70, 90, 100]:
        sub = df_sweep[(df_sweep['N'] == N) & df_sweep[metric].notna()].sort_values('TN_ratio')
        if len(sub) == 0:
            continue
        ax.scatter(sub['TN_ratio'], sub[metric],
                   color=colors_n[N], marker=markers_n[N],
                   s=100, zorder=4, label=f'N={N}')
        ax.plot(sub['TN_ratio'], sub[metric],
                color=colors_n[N], lw=1.2, alpha=0.5)

    ax.axhline(0,     color='black', lw=0.8, ls='--')
    ax.axhline(1.96,  color='#e53935', lw=1.2, ls=':', alpha=0.8, label='t=±1.96')
    ax.axhline(-1.96, color='#e53935', lw=1.2, ls=':', alpha=0.8)
    ax.axhline(1.65,  color='#fb8c00', lw=1.0, ls=':', alpha=0.6, label='t=±1.65')
    ax.axhline(-1.65, color='#fb8c00', lw=1.0, ls=':', alpha=0.6)

    # Shade the "reliable estimation zone"
    ax.axvspan(8, ax.get_xlim()[1] if ax.get_xlim()[1] > 8 else 10,
               alpha=0.08, color='green', label='T/N>8 zone')

    ax.set_xlabel('T/N Ratio', fontsize=12)
    ax.set_ylabel('t-statistic', fontsize=12)
    ax.set_title(f'{title}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

# fix xlim for shade
for ax in axes:
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], max(xlim[1], 9))
    ax.axvspan(8, max(xlim[1], 9), alpha=0.08, color='green')

plt.suptitle('The T/N Barrier — t-statistic vs Estimation Quality\n'
             'No consistent signal emerges below T/N ≈ 8',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PAPER_DIR / 'figure6_TN_barrier.pdf', bbox_inches='tight')
plt.savefig(PAPER_DIR / 'figure6_TN_barrier.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure6")

# ============================================================================
# Table 2: Simulation results
# ============================================================================
print("[Table 2] Simulation results...")

sim = pd.read_csv(OUTPUT_DIR / 'simulation_results.csv')
tbl2 = sim.groupby(['N', 'T', 'method'])[['f1', 'precision', 'recall']].mean().round(3)
tbl2['T_N'] = (tbl2.index.get_level_values('T') /
               tbl2.index.get_level_values('N')).round(2)
tbl2 = tbl2.reset_index()
tbl2.to_csv(PAPER_DIR / 'table2_simulation.csv', index=False)

# LaTeX
latex_t2 = tbl2.to_latex(index=False, float_format='%.3f',
    caption='Monte Carlo Simulation: Network Recovery Metrics by N, T, and Method',
    label='tab:simulation',
    column_format='rrrrrrrr')
with open(PAPER_DIR / 'table2_simulation.tex', 'w') as f:
    f.write(latex_t2)
print("  Saved table2")

# ============================================================================
# Table 3: LASSO network stats by window (N=100)
# ============================================================================
print("[Table 3] LASSO network stats...")

rows3 = []
for T in [60, 120, 180, 252]:
    csv = OUTPUT_DIR / f'lasso_nio_T{T}.csv'
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    conn_pct = (df['nio'] != 0).mean() * 100
    rows3.append({
        'Window T': T,
        'T/N': round(T/100, 2),
        'Connected (%)': round(conn_pct, 1),
        'Mean Density (NIO!=0)': round((df['nio'] != 0).mean(), 4),
        'Observations': len(df),
    })

tbl3 = pd.DataFrame(rows3)
tbl3.to_csv(PAPER_DIR / 'table3_lasso_stats.csv', index=False)
latex_t3 = tbl3.to_latex(index=False,
    caption='LASSO-TE Network Statistics on Real Data (N=100)',
    label='tab:lasso_stats',
    column_format='rrrrrr')
with open(PAPER_DIR / 'table3_lasso_stats.tex', 'w') as f:
    f.write(latex_t3)
print("  Saved table3")

# ============================================================================
# Table 4: Window x N t-stat grid
# ============================================================================
print("[Table 4] Window x N sweep grid...")

# Binary split t-stats
grid_bs = {
    30:  {60: 0.80, 120: 1.00, 180: 1.29, 252: 1.96},
    50:  {60: -1.01, 120: -1.06, 180: -0.55, 252: -0.33},
    70:  {60: -1.45, 120: -1.93, 180: 0.92, 252: -0.44},
    90:  {60: -1.25, 120: -1.63, 180: -0.18, 252: -0.75},
    100: {60: -1.02, 120: -2.09, 180: 0.35, 252: -0.65},
}

rows4 = []
for N in [30, 50, 70, 90, 100]:
    row = {'N': N}
    for T in [60, 120, 180, 252]:
        tn = round(T/N, 1)
        t  = grid_bs[N].get(T, np.nan)
        row[f'T={T} (T/N={tn})'] = f'{t:.2f}' if not np.isnan(t) else '--'
    rows4.append(row)

tbl4 = pd.DataFrame(rows4)
tbl4.to_csv(PAPER_DIR / 'table4_sweep_grid.csv', index=False)
latex_t4 = tbl4.to_latex(index=False,
    caption='Binary Split t-statistics: Connected vs. Isolated Stocks (N x T grid)',
    label='tab:sweep_grid',
    column_format='r' + 'c'*4)
with open(PAPER_DIR / 'table4_sweep_grid.tex', 'w') as f:
    f.write(latex_t4)
print("  Saved table4")

# ============================================================================
# Copy remaining figures
# ============================================================================
import shutil
shutil.copy(OUTPUT_DIR / 'simulation_figure.png', PAPER_DIR / 'figure3_simulation.png')
shutil.copy(OUTPUT_DIR / 'window_sweep_figure.png', PAPER_DIR / 'figure_window_sweep.png')
shutil.copy(OUTPUT_DIR / 'binary_split_figure.png', PAPER_DIR / 'figure_binary_split.png')

print("\n" + "="*50)
print("PAPER ASSETS SUMMARY")
print("="*50)
for f in sorted(PAPER_DIR.iterdir()):
    print(f"  {f.name}")
print("\nDone. Now generating LaTeX...")
