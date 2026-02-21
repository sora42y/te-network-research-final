"""
generate_nonparametric_figure.py
为论文生成非参数 TE 对比图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research\results')
PAPER_ASSETS = Path(r'C:\Users\soray\.openclaw\workspace\te_network_research\paper_assets')
PAPER_ASSETS.mkdir(exist_ok=True)

# Load summary data
df = pd.read_csv(OUTPUT / 'nonparametric_te_summary.csv')

print("Loaded data:")
print(df[['N', 'T', 'T/N', 'method', 'precision_mean', 'precision_std']])

# Filter to key T/N ratios (approximately)
# Due to float precision, use tolerance
df_filtered = []
for tn_target in [0.6, 2.5, 5.0]:
    subset = df[(df['T/N'] >= tn_target - 0.1) & (df['T/N'] <= tn_target + 0.1)]
    df_filtered.append(subset)

df = pd.concat(df_filtered, ignore_index=True)
print(f"\nFiltered to {len(df)} rows for T/N ≈ 0.6, 2.5, 5.0")
print(df[['N', 'T', 'T/N', 'method', 'precision_mean']])

# Pivot for plotting
methods = ['LASSO', 'KNN']  # Remove OLS (NaN at low T/N)
tn_labels = ['0.6', '2.5', '5.0']

# Group by approximate T/N
grouped = []
for tn_target in [0.6, 2.5, 5.0]:
    subset = df[(df['T/N'] >= tn_target - 0.1) & (df['T/N'] <= tn_target + 0.1)]
    # Take first N,T config for each T/N bucket (they might have multiple N,T pairs)
    if len(subset) > 0:
        tn_group = subset.groupby('method').first().reset_index()
        tn_group['T/N_label'] = f'{tn_target:.1f}'
        grouped.append(tn_group)

df_plot = pd.concat(grouped, ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(tn_labels))
width = 0.35  # Wider bars since only 2 methods

for i, method in enumerate(methods):
    subset = df_plot[df_plot['method'] == method].sort_values('T/N_label')
    
    if len(subset) == 0:
        continue
    
    means = subset['precision_mean'].values
    stds = subset['precision_std'].values
    
    # Ensure same length
    if len(means) != len(tn_labels):
        print(f"Warning: {method} has {len(means)} values, expected {len(tn_labels)}")
        # Pad with zeros if needed
        means_full = np.zeros(len(tn_labels))
        stds_full = np.zeros(len(tn_labels))
        for j, label in enumerate(subset['T/N_label'].values):
            idx = tn_labels.index(label)
            means_full[idx] = means[j]
            stds_full[idx] = stds[j]
        means = means_full
        stds = stds_full
    
    offset = (i - 0.5) * width  # Center around x positions
    bars = ax.bar(x + offset, means, width, 
                   label=method, 
                   yerr=stds, 
                   capsize=5,
                   alpha=0.8)
    
    # Add value labels on bars
    for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        if mean > 0:  # Only label non-zero bars
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.1%}',
                    ha='center', va='bottom', fontsize=9)

# Random baseline line (5% for sparse networks)
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, 
           label='Random baseline (5%)', alpha=0.7)

ax.set_xlabel('T/N Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Nonparametric vs Linear TE: Precision Comparison\n' + 
             'GARCH+Factor DGP, 5% Edge Density',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tn_labels)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(df_plot['precision_mean']) * 1.3 if len(df_plot) > 0 else 1.0)

plt.tight_layout()
plt.savefig(PAPER_ASSETS / 'figure_nonparametric_te.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {PAPER_ASSETS / 'figure_nonparametric_te.png'}")

# Also generate LaTeX table
print("\n" + "="*80)
print("LATEX TABLE CODE:")
print("="*80)

print(r"""
\begin{table}[H]
  \centering
  \caption{Nonparametric vs Linear TE: Precision Comparison (GARCH+Factor DGP)}
  \label{tab:nonparametric_te}
  \small
  \begin{tabular}{lccc}
    \toprule
    & \multicolumn{2}{c}{Precision (mean $\pm$ std)} \\
    \cmidrule(lr){2-3}
    $T/N$ & LASSO & KNN (Nonparametric) \\
    \midrule""")

for tn_target in [0.6, 2.5, 5.0]:
    subset = df[(df['T/N'] >= tn_target - 0.1) & (df['T/N'] <= tn_target + 0.1)]
    row_data = {}
    for method in methods:
        method_data = subset[subset['method'] == method]
        if len(method_data) > 0:
            mean = method_data['precision_mean'].values[0]
            std = method_data['precision_std'].values[0]
            row_data[method] = f"{mean:.1%} $\\pm$ {std:.1%}"
        else:
            row_data[method] = "---"
    
    lasso = row_data.get('LASSO', '---')
    knn = row_data.get('KNN', '---')
    
    print(f"    {tn_target:.1f} & {lasso} & {knn} \\\\")

print(r"""    \midrule
    \multicolumn{3}{l}{\textit{Random baseline: 5\% (sparse network, 5\% edge density)}} \\
    \multicolumn{3}{l}{\textit{OLS excluded: undefined at $T/N < 1$ (matrix not invertible)}} \\
    \bottomrule
  \end{tabular}
  \medskip\\
  \footnotesize
  \textit{Note:} KNN-based nonparametric TE (Kozachenko-Leonenko entropy estimator,
  $k=3$ nearest neighbors) performs near-random at all $T/N < 5$, with precision
  barely above the 5\% baseline. LASSO, despite relying on linear assumptions,
  achieves 6--77\% precision depending on $T/N$. At financial sample sizes
  ($T/N < 5$), nonparametric methods suffer catastrophic dimensionality curse;
  linear methods are the only feasible approach. Mean $\pm$ std from 50 trials.
\end{table}
""")

print("="*80)
