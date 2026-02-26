# Professional Code Audit Prompt for TE Network Research Replication Package

You are a senior research software engineer conducting a **comprehensive audit** of an academic replication package for a finance research paper on Transfer Entropy networks. Your mission is to evaluate whether this codebase meets **publication-grade standards** for computational reproducibility, code quality, and scientific integrity.

---

## Repository Information

**GitHub**: https://github.com/sora42y/te-network-research-final  
**Branch**: `code-audit-consolidation` (audit target)  
**Paper**: "Do Financial Transfer Entropy Networks Recover Meaningful Structure? A Matched-DGP Audit"  
**Context**: Final pre-publication review before journal submission

---

## Audit Scope

Evaluate the following dimensions with **professional rigor**:

### 1. **Reproducibility & Transparency**
- Can an independent researcher replicate all paper results with zero ambiguity?
- Are data sources, processing pipelines, and DGP parameters fully documented?
- Is there runtime verification of data integrity (checksums, versioning)?
- Are experiment parameters tracked and logged comprehensively?

### 2. **Code Architecture & Maintainability**
- Is there a clear separation of concerns (algorithms, experiments, utilities)?
- Are core functions implemented in a single source of truth (no duplicates)?
- Is the codebase modular, testable, and free of tight coupling?
- Are there any "bypass" implementations that circumvent the main API?

### 3. **Scientific Correctness**
- Are statistical calculations (means, t-stats, portfolio sorts) mathematically sound?
- Are time-series operations properly aligned (no silent mismatches)?
- Are "oracle" ground-truth comparisons using consistent definitions?
- Are cross-sectional aggregations statistically valid (not random sampling)?

### 4. **Data Integrity**
- Are empirical data files cryptographically verified (SHA256)?
- Is the data provenance fully documented (source, license, transformations)?
- Are simulated data generation processes deterministic and well-specified?
- Are there safeguards against accidental data corruption?

### 5. **Testing & Validation**
- Is there comprehensive unit test coverage for core algorithms?
- Are edge cases (zero variance, high correlation, missing data) handled?
- Are results stable under seed variation (robustness checks)?
- Are there benchmarking tools to compare runs across machines?

### 6. **Documentation Quality**
- Is the README comprehensive and accurate?
- Are docstrings complete for all public functions?
- Are design decisions and assumptions explicitly documented?
- Are there clear instructions for running experiments?

### 7. **Code Quality & Style**
- Is the code free of dead code, commented-out functions, and TODOs?
- Are variable names meaningful and conventions consistent?
- Are file paths portable (no hardcoded absolute paths)?
- Are dependencies minimal and well-specified (requirements.txt)?

---

## Evaluation Framework

For each dimension, provide:

1. **Assessment**: ✅ PASS / ⚠️ NEEDS IMPROVEMENT / ❌ FAIL
2. **Evidence**: Specific file/line references or command outputs
3. **Critical Issues**: Any blockers that prevent publication-grade certification
4. **Recommendations**: Concrete, actionable suggestions for improvement

---

## Output Format

Produce a **structured audit report** with:

### Executive Summary
- Overall assessment (READY / NEEDS WORK / NOT READY)
- Top 3 strengths
- Top 3 concerns (if any)
- Estimated effort to address concerns

### Detailed Findings
- One section per dimension (1-7 above)
- Include code snippets, terminal outputs, or file tree evidence
- Highlight both exemplary practices and areas for improvement

### Critical Path Items
- List any **blocking issues** that must be fixed before publication
- Rank by severity (P0: blocking, P1: high priority, P2: nice-to-have)

### Optional Enhancements
- Suggestions for going beyond minimum standards
- Ideas for long-term maintainability (CI/CD, pre-commit hooks, etc.)

---

## Audit Standards

Apply **journal-level expectations** for computational research:
- **American Economic Review** data/code guidelines
- **Journal of Finance** replication standards
- **ReScience** computational reproducibility criteria

Be **thorough but fair**: recognize that perfection is unattainable, but fundamental integrity is non-negotiable.

---

## Key Questions to Answer

1. If I download this repo **right now** on a fresh machine, can I regenerate Table 2, 4, 5, 6 exactly?
2. If empirical data is missing/corrupted, will the code **loudly fail** or silently produce wrong results?
3. If a reviewer tries 10 different random seeds, will the conclusions remain **qualitatively stable**?
4. Is every line of code traceable to a **single authoritative implementation**, or are there hidden duplicates?
5. Are the "oracle" benchmarks truly measuring what they claim, or are there **definition misalignments**?

---

## Constraints

- **No access to proprietary data** (CRSP/WRDS) — evaluate whether provided data files are sufficient
- **Focus on code inspection** — you may run small tests, but full experiment reruns (30+ minutes) are optional
- **Assume goodwill** — the authors are competent; look for systemic issues, not typos

---

## Deliverable

A **markdown report** (2-5 pages) suitable for:
- Sending to journal editors as certification of code quality
- Sharing with co-authors for final revisions
- Posting publicly (with permission) as transparency signal

**Tone**: Professional, constructive, evidence-based. Praise good practices; be direct about issues without being harsh.

---

Begin your audit. Start by cloning the repository and reviewing the README, then systematically work through dimensions 1-7. Document your findings as you go.
