"""
Quick script to delete sections from LaTeX
"""

import re

tex_file = r"C:\Users\soray\.openclaw\workspace\te-network-research-final\paper\main.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match from \subsection{Nonlinear DGP} to \subsection{Mechanism Decomposition}
# We'll delete everything between line 587 and line 716

lines = content.split('\n')

# Find the line numbers
delete_start = None
delete_end = None

for i, line in enumerate(lines):
    if r'\subsection{Nonlinear DGP: Threshold-VAR}' in line:
        delete_start = i
    if r'\subsection{Mechanism Decomposition: GARCH vs.\ Common Factors}' in line and delete_start is not None:
        delete_end = i
        break

print(f"Delete from line {delete_start} to {delete_end}")

if delete_start and delete_end:
    # Insert replacement text
    replacement = r"""\paragraph{Robustness checks (Appendix).} We test robustness to nonlinear
dynamics (threshold-VAR), lag misspecification (VAR(2) estimated with lag-1),
and regularization family (Elastic Net $\alpha \in [0.1, 1.0]$). The $T/N$
barrier persists or worsens under all variations. No convex penalization
achieves F1 $> 0.55$ at $T/N \leq 5$ (Appendix~\ref{app:robustness}).

"""
    
    # Keep before delete_start, add replacement, keep from delete_end onwards
    new_lines = lines[:delete_start] + [replacement] + lines[delete_end:]
    
    new_content = '\n'.join(new_lines)
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"OK Deleted {delete_end - delete_start} lines")
    print(f"New file: {len(new_lines)} lines (was {len(lines)})")
