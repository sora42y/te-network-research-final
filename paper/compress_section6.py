"""
Compress Section 6 Discussion
"""

tex_file = r"C:\Users\soray\.openclaw\workspace\te-network-research-final\paper\main.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Section 6.2 + 6.3 with compressed version
old_6_2_start = r'\subsection{Aggregate Connectedness Recovery}'
old_6_3_end = r'node-level claims (requiring\nvalidation at the paper\'s actual $T/N$).'

new_6_2_compressed = r'''\subsection{Aggregate Connectedness Recovery}

A central claim in \citet{billio2012} is that \emph{aggregate} network
connectedness tracks systemic risk. We test this directly: simulate
$N=50$, $T=500$ with regime switch at $t=250$ (density 3\% → 12\%).

OLS-TE detects the crisis-period increase in 100\% of trials (100 Monte Carlo);
LASSO-TE detects it in 89\%. However, both underestimate the \emph{level}:
OLS reports 5.1\%--7.3\% vs.\ true 3\%--12\%; LASSO is compressed further
(0.17\%--0.39\%). Rolling window robustness: OLS-TE density correlates
$r=0.46$ with truth; LASSO $r=0.09$ (near-empty network has little room to move).

Conclusion: aggregate connectedness \emph{trends} are partially recoverable
by OLS-TE, but node-level hub identities are not. The two claims must be
separated. Aggregate measures (mean density, Diebold-Yilmaz connectedness index)
aggregate over many edges and benefit from error cancellation; they may
validly detect stress-period connectivity increases even when individual
hub identities fail.'''

# Replace Section 6.4 Billio with compressed version
old_6_4_full = r'\subsection{Hub Sector Identification: Billio-Style Test}'
old_6_4_end = r'sector-level\ninference also fails. Neither estimator reliably recovers the full\nedge topology at financially relevant $T/N$ ratios.'

new_6_4_compressed = r'''\subsection{Hub Sector Identification: Billio-Style Test}

\citet{billio2012} identify banks/insurers as dominant transmitters.
We test whether TE recovers a designated hub sector. Assign 5 hub nodes
out-edge density 30\% (hub→rest) vs.\ 5\% background ($N=50$, $T=500$).

\begin{table}[H]
  \centering
  \caption{Hub Sector Detection ($N_{\text{hub}}=5$, $N=50$, $T=500$, 100 trials)}
  \label{tab:hub_billio}
  \small
  \begin{tabular}{lcccc}
    \toprule
    & Hub $>$ median & Kendall $\tau$ & Hub NIO $t$ & Pr[$|t|>1.96$] \\
    \midrule
    OLS-TE   & 0.998 & 0.381 & 7.38 & 100\% \\
    LASSO-TE & 0.212 & 0.116 & 0.72 & 6\% \\
    \midrule
    Random baseline & 0.500 & 0.000 & --- & 5\% \\
    \bottomrule
  \end{tabular}
\end{table}

OLS-TE identifies hub nodes above median in 99.8\% of cases ($t=7.4$,
significant in 100\% of trials). LASSO-TE: 21.2\% (below random).
Trade-off: OLS preserves aggregate signal for sector-level inference
despite low edge precision; LASSO's near-empty networks suppress sector
rankings.'''

# Compress Constructive Strategies
old_strategies = r'\subsection{Constructive Strategies}'
old_strategies_end = r'Benjamini--Hochberg FDR control on edge $p$-values makes uncertainty'

new_strategies = r'''\subsection{Constructive Strategies}

Three approaches improve reliability at low $T/N$: (1) Use hourly/5-minute
data (increases $T$ by 10--80×; trade-off: microstructure noise).
(2) Aggregate to 10--20 sectors instead of 100 stocks ($N=12$ sectors,
$T=500$ → $T/N=41.7$, well above threshold; sacrifice granularity, gain validity).
(3) FDR-controlled inference (stability selection, Benjamini-Hochberg)
makes uncertainty'''

# Perform replacements
# (Simplified: just show what we're doing - full implementation would use proper regex)

print("Compression plan:")
print("- Section 6.2+6.3: ~60 lines → ~20 lines")
print("- Section 6.4: ~40 lines → ~20 lines")  
print("- Constructive Strategies: ~15 lines → ~5 lines")
print("\nTotal savings: ~70 lines")
print("\nManual edit recommended for precision")
