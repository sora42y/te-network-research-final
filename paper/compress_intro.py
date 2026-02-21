"""
Compress Introduction 1.1-1.3 and delete Roadmap
Human-style rewrite, less AI
"""

tex_file = r"C:\Users\soray\.openclaw\workspace\te-network-research-final\paper\main.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find subsections in Introduction
intro_start = None
roadmap_start = None
roadmap_end = None
lit_review_start = None

for i, line in enumerate(lines):
    if r'\subsection{What We Do}' in line:
        intro_start = i
    if r'\subsection{Roadmap}' in line:
        roadmap_start = i
    if roadmap_start and r'\section{Related Literature}' in line:
        roadmap_end = i
        lit_review_start = i
        break

print(f"Found subsections:")
print(f"  What We Do: line {intro_start}")
print(f"  Roadmap: line {roadmap_start}")
print(f"  Related Literature: line {lit_review_start}")

if intro_start and roadmap_start and lit_review_start:
    # New compressed text
    new_intro = r"""\subsection{Overview and Contribution}

We run a matched-$(N,T)$ simulation audit of TE/GC network estimation across
$T/N$ ratios from 0.6 to 16.7, using three DGPs: Gaussian (baseline),
GARCH(1,1), and GARCH+Factor (realistic). At $T/N < 5$—the regime of all
seven papers in Table~\ref{tab:literature}—OLS precision is 7--17\% (83--93\%
false positives). LASSO achieves higher precision but near-zero recall,
collapsing to density $<1\%$ on S\&P 500 data. Hub detection performs close
to random.

We also embed a known 10\% annualized network premium in simulated data and
check if estimation recovers it. Oracle $t=5.74$ (significant); estimated
$t \approx 0.7$ at $T/N=5$ (noise). Even with 30\% premia, signals barely
emerge at $T/N=10$. This power test shows estimation noise alone destroys
plausible network-return channels.

Three contributions. First, we bridge the high-dimensional VAR literature
\citep{adamek2023, hecq2023} and the applied network literature by quantifying
\emph{network-level} recovery failure—hub detection, edge sets—at the exact
$(N,T)$ pairs used in published work. Second, we establish the $T/N$ barrier:
reliable edge-level estimation requires $T/N \approx 8$--10, far above
$T/N < 5$ used in most studies. Third, the power test sets a lower bound on
required $T/N$ for detectability, independent of whether network-return
mechanisms exist in real markets.

% ============================================================
% 2. Related Literature
% ============================================================
"""
    
    # Replace from intro_start to lit_review_start
    new_lines = lines[:intro_start] + [new_intro] + lines[lit_review_start:]
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    deleted = (lit_review_start - intro_start)
    print(f"\nDeleted {deleted} lines")
    print(f"New file: {len(new_lines)} lines (was {len(lines)})")
