# External Code Audit Report

**Repository**: `te-network-research-final` (branch: `code-audit-consolidation`)
**Paper**: "Do Financial Transfer Entropy Networks Recover Meaningful Structure? A Matched-DGP Audit"
**Auditor**: Independent code review (automated + manual inspection)
**Date**: 2026-02-26
**Environment**: Python 3.13.3, scikit-learn 1.8.0, numpy 2.4.0, pandas 2.3.3

---

## Executive Summary

**Overall Assessment: NEEDS WORK**

Two issues must be resolved before this codebase meets publication-grade standards. One is a silent runtime failure (P0); the other is a direction-convention error in a core metric (P1). Both are straightforward to fix.

### Top 3 Strengths

1. **Reproducibility infrastructure** is exemplary. SHA256 fingerprinting of data, source code, environment, and parameters exceeds what most published replication packages provide. The dual-mode seed system (FIXED/RANDOM) and the `compare_runs.py` benchmarking tool are well-designed.

2. **Architecture is clean and modular**. Core algorithms in `src/algorithms.py` are pure functions with no I/O or side effects. The `te_core.py` / `dgp.py` facades provide a single public API with no hidden bypass implementations. Experiment scripts are independent and can be run in isolation.

3. **Edge case handling in numerical code** is thorough. Zero-variance guards (`sigma2_full < 1e-12`), singular-matrix protection, and minimum-sample checks throughout the core algorithms demonstrate careful numerical programming.

### Top 3 Concerns

1. **P0 - LASSO-TE is silently broken** on scikit-learn >= 1.2 (the minimum required version is 1.3). The `normalize=False` parameter was removed from `LassoLarsIC`, causing a `TypeError` that is silently swallowed by a bare `except:` clause. Every LASSO call returns zero edges.

2. **P1 - NIO direction is inverted** relative to the stated convention. `compute_nio()` labels row sums as "out_flow" and column sums as "in_flow," but given the matrix convention `A[i,j] = j->i`, these are reversed. The computation is internally consistent (planted signal, oracle, and estimated NIO all use the same inverted labels), so comparative results remain valid. However, the semantic interpretation in the paper ("high NIO = information transmitter") describes the opposite of what the code computes.

3. **P1 - Bare `except:` blocks** (11 instances) mask errors across the codebase, including the LASSO failure above. These should be narrowed to specific exception types.

---

## Detailed Findings

### 1. Reproducibility & Transparency

**Assessment: ⚠️ NEEDS IMPROVEMENT** (conditional on P0 fix)

**Strengths:**
- All empirical data is pre-included (zero runtime downloads)
- SHA256 checksums for both data files are documented in `docs/MANIFEST.md` and verified at runtime in `src/empirical_portfolio_sort.py:195-213`
- `scripts/experiment_metadata.py` produces a 16-character fingerprint from the SHA256 of script + environment + parameters + all source code
- `scripts/results_manager.py` creates timestamped output directories with per-table metadata (SHA256, row count, column names)
- Seed management via `scripts/simulation_config.py` supports both deterministic (paper) and random (robustness) modes
- Git commit hash is recorded in every `run_metadata.json`

**Issues:**
- **Can a reviewer regenerate tables exactly?** Tables 2, 5, and 6 use OLS-TE and would likely reproduce. Any table relying on LASSO-TE will produce different results with scikit-learn >= 1.2 due to the P0 bug. Existing benchmark results in `results/bench_seed*` were likely generated with an older scikit-learn version.
- The seed formula for Table 2 (`seed = 1000 + trial` at `run_factor_neutral_sim.py:149`) does not use the centralized `SimulationConfig` seed strategy (`seed_base + trial*1000 + N + T` at `simulation_config.py:107`). This means `run_factor_neutral_sim.py` seeds are not controlled by the global seed configuration.
- Processing scripts that created the empirical data (`build_universe_500.py`, `weekly_te_pipeline_500.py`) are referenced in `docs/DATA_SOURCES.md` but not included in the repository.

**Evidence:**
```
$ python -c "from sklearn.linear_model import LassoLarsIC; LassoLarsIC(normalize=False)"
TypeError: LassoLarsIC.__init__() got an unexpected keyword argument 'normalize'
```

```
# LASSO silently returns zero edges on known causal structure:
LASSO edges detected: 0    # Should be >= 1
OLS edges detected: 1      # Correctly finds x -> y
```

### 2. Code Architecture & Maintainability

**Assessment: ✅ PASS**

The architecture follows a well-structured layered design:

```
Public API:     te_core.py, dgp.py          (re-export facades)
Algorithms:     algorithms.py               (pure TE/NIO/metrics)
Evaluation:     evaluation.py               (extended metrics, cross-sectional t-stat)
DGP:            extended_dgp.py             (GARCH+Factor VAR generation)
                extended_dgp_planted_signal.py (NIO premium variant)
Preprocessing:  factor_neutral_preprocessing.py (PCA-based factor adjustment)
Experiments:    run_factor_neutral_sim.py    (Table 2)
                all_experiments_v2.py        (Table 4)
                empirical_portfolio_sort.py  (Table 5)
                oracle_nio_power.py          (Table 6)
Config:         scripts/simulation_config.py
                scripts/experiment_metadata.py
                scripts/results_manager.py
```

- All paths use `pathlib.Path` relative to `__file__` — no hardcoded absolute paths
- All subprocess invocations use `sys.executable` and avoid `shell=True`
- Cross-platform compatible (tested on Windows; uses no OS-specific features)

**One duplication found:** `run_factor_neutral_sim.py:39-77` defines a local `compute_metrics()` function that reimplements the same precision/recall/F1 + hub recovery logic available in `algorithms.py:223-258` and `evaluation.py:16-59`. Line 33 imports `compute_precision_recall_f1` from `evaluation` but never uses it, calling the local duplicate instead at line 113.

### 3. Scientific Correctness

**Assessment: ⚠️ NEEDS IMPROVEMENT**

**Mathematically correct implementations:**

| Component | File:Lines | Status |
|-----------|-----------|--------|
| OLS-TE (restricted/unrestricted F-test) | `algorithms.py:50-109` | Correct |
| LASSO-TE (Frisch-Waugh + BIC) | `algorithms.py:112-178` | Correct (when it runs) |
| KNN entropy (Kozachenko-Leonenko) | `nonparametric_te.py:42-83` | Correct |
| GARCH(1,1) DGP | `extended_dgp.py:73-89` | Correct, cited to Engle & Bollerslev (1986) |
| Factor model DGP | `extended_dgp.py:91-124` | Correct (K=3 factors, AR(1) dynamics) |
| VAR(1) stability check | `extended_dgp.py:60-62` | Correct (spectral radius <= 0.9) |
| Cross-sectional t-stat | `evaluation.py:62-106` | Correct (OLS with proper DOF) |
| Precision/Recall/F1 | `algorithms.py:223-258` | Correct (diagonal exclusion) |
| Portfolio sort t-stat | `empirical_portfolio_sort.py:279-297` | Correct (date-aligned inner merge) |

**NIO direction issue (P1):**

The adjacency matrix convention is `A[i,j]=1 means j causes i`. Under this convention:
- **Out-degree of node k** = `A[:, k].sum()` (column sum)
- **In-degree of node k** = `A[k, :].sum()` (row sum)

But `compute_nio()` at `algorithms.py:206-218` computes:
```python
out_flow = (te_matrix[i, :] > 0).sum()  # This is IN-degree
in_flow = (te_matrix[:, i] > 0).sum()   # This is OUT-degree
nio[i] = (out_flow - in_flow) / (N - 1) # = (in_degree - out_degree) / (N-1)
```

The same inversion appears in `extended_dgp_planted_signal.py:46-48`:
```python
out_flow = A_true.sum(axis=1)  # Row sums = in-degree
in_flow = A_true.sum(axis=0)   # Col sums = out-degree
```

And in the test at `test_algorithms.py:126-127`:
```python
# Node 0 is a hub: 9 out-edges, 0 in-edges
te_matrix[0, 1:] = 1.0  # Actually creates 9 IN-edges to node 0
```

**Verification:**
```
# Node 0 has true out-degree=3, in-degree=0 (pure transmitter)
# compute_nio returns: nio[0] = -1.0 (LOWEST, should be HIGHEST)
```

Because ALL uses of NIO (planted signal, oracle, estimated, tests) share the same inversion, comparative results (oracle vs. estimated power, hub recovery) remain internally valid. The issue is interpretive: "high NIO" in this codebase means "high in-degree" (information receiver), not "high out-degree" (information transmitter) as the docstrings and paper narrative suggest.

### 4. Data Integrity

**Assessment: ✅ PASS**

- `te_features_weekly.csv` (32 MB): SHA256 `87544851...` documented and verified at runtime (`empirical_portfolio_sort.py:195-213`)
- `universe_500.csv` (4.7 MB): SHA256 `8cee923a...` documented in `docs/MANIFEST.md`
- Runtime SHA256 mismatch produces a visible `WARNING` and continues with notice — a reasonable balance between strictness and usability
- Data provenance fully documented: CRSP via WRDS, query date 2025-12-31, processing pipeline described in `docs/DATA_SOURCES.md`
- Results are saved to immutable timestamped directories via `ResultsManager`
- Each result file gets a companion `_meta.json` with auto-computed SHA256

**Minor gap:** The SHA256 runtime check is only implemented for `te_features_weekly.csv` in the portfolio sort script. `universe_500.csv` is checksummed in documentation but not verified at runtime.

### 5. Testing & Validation

**Assessment: ⚠️ NEEDS IMPROVEMENT**

**Current coverage:** 16 tests in `test_algorithms.py`, all passing (2.29s):

| Test Class | Tests | What's Covered |
|------------|-------|----------------|
| `TestOLS_TE` | 4 | Independent series, known causality, zero variance, reproducibility |
| `TestLASSO_TE` | 2 | Sparse network, empty network |
| `TestNIO` | 3 | Hub node, symmetric network, weighted vs binary |
| `TestMetrics` | 4 | Perfect/wrong/partial recovery, empty network |
| `TestEdgeCases` | 3 | Single asset, short series, high correlation |

**Critical test gaps:**

1. **LASSO tests have no minimum-edge assertions.** `test_sparse_network` asserts `density < 0.3` and `test_empty_network` asserts `density < 0.2` — both pass trivially when LASSO returns zero edges due to the P0 bug. A test like `assert adj.sum() >= 1` on known causal data would have caught this immediately.

2. **NIO hub test has inverted semantics.** `test_hub_node` at line 127 sets `te_matrix[0, 1:] = 1.0` with the comment "Node 0 points to all others," but given the `TE[i,j] = j->i` convention, this creates edges FROM all others TO node 0. The test passes because `compute_nio` shares the same inversion.

3. **No seed-variation robustness tests.** The `compare_runs.py` tool exists and benchmark results show CV < 1% across seeds 42/100/200, but there is no automated test that verifies seed stability.

4. **No tests under GARCH/fat-tailed data.** All algorithm tests use Gaussian innovations. The DGP supports t(5) innovations and GARCH dynamics, but these aren't tested.

5. **No integration tests.** End-to-end pipeline tests (DGP -> preprocessing -> TE estimation -> evaluation) are not present.

### 6. Documentation Quality

**Assessment: ✅ PASS**

- `README.md` (11 KB) is comprehensive: quick start, architecture overview, table descriptions, reproducibility claims
- `docs/DATA_SOURCES.md` provides full data lineage with processing pipeline description
- `docs/MANIFEST.md` catalogs all data files with SHA256 checksums
- `docs/REPRODUCIBILITY.md` explains the dual-mode seed framework
- All public functions in `algorithms.py`, `evaluation.py`, and `extended_dgp.py` have NumPy-style docstrings with Parameters/Returns sections
- Configuration parameters in `simulation_config.py` cite academic references (Engle & Bollerslev 1986)

**Minor issues:**
- Some files contain Chinese/Unicode comments (`nonparametric_te.py:3-6`, `all_experiments_v2.py:2-7`) — these should be translated to English for international review
- No type hints on any function signatures (reduces IDE support and static analysis)
- The `README.md` claims "Zero duplicate implementations" but `run_factor_neutral_sim.py:39-77` contains a duplicate `compute_metrics` function

### 7. Code Quality & Style

**Assessment: ⚠️ NEEDS IMPROVEMENT**

**Good practices:**
- Consistent `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- No dead code or commented-out functions
- No hardcoded paths
- Minimal dependencies (6 packages in `requirements.txt`)
- `pathlib.Path` used exclusively for file paths

**Issues:**

| Issue | Count | Severity | Locations |
|-------|-------|----------|-----------|
| Bare `except:` blocks | 11 | High | `algorithms.py:101,175`, `evaluation.py:103`, `nonparametric_te.py:212,219`, `simulation_config.py:148`, `compare_runs.py:223`, `results_manager.py:173,184,205`, `experiment_metadata.py:87` |
| Git utility duplication | 3 files | Low | `results_manager.py:165-174`, `simulation_config.py:137-149`, `experiment_metadata.py:79-88` |
| Deprecated parameter | 1 | Critical | `algorithms.py:153` (`normalize=False` in `LassoLarsIC`) |
| Unused import | 1 | Low | `run_factor_neutral_sim.py:33` imports `compute_precision_recall_f1` but uses local `compute_metrics` |
| No type hints | ~50 functions | Low | All files |

**The bare `except:` pattern is particularly harmful** here because it masks the LASSO `TypeError`. Replace with:
```python
except np.linalg.LinAlgError:
    t_stat = 0
```
or at minimum:
```python
except Exception as e:
    warnings.warn(f"LASSO fitting failed: {e}")
```

---

## Critical Path Items

### P0: LASSO-TE Silently Broken (BLOCKING)

**File:** `src/algorithms.py:153`
**Cause:** `LassoLarsIC(criterion='bic', max_iter=1000, normalize=False)` — the `normalize` parameter was removed in scikit-learn 1.2. The `TypeError` is caught by the bare `except:` at line 175, causing every LASSO call to return an empty adjacency matrix.
**Impact:** All LASSO-TE results in the paper cannot be reproduced with the specified dependencies (scikit-learn >= 1.3).
**Fix:** Remove `normalize=False` from line 153. The default behavior (no normalization) is equivalent to `normalize=False`, so removing it preserves the intended semantics:
```python
lasso = LassoLarsIC(criterion='bic', max_iter=1000)
```
**Verification:** After fix, re-run `test_algorithms.py` and verify LASSO detects edges in `test_sparse_network`. Add an assertion: `assert adj.sum() >= 1, "LASSO should detect at least one edge in causal data"`.

### P1: NIO Direction Convention Inverted

**Files:** `src/algorithms.py:206-218`, `src/extended_dgp_planted_signal.py:46-48`, `test_algorithms.py:126-127`
**Cause:** Row sums of `A[i,j]` (where `j->i`) give in-degree, not out-degree. The code labels row sums as `out_flow` and column sums as `in_flow`, inverting the meaning.
**Impact:** The sign of NIO is flipped: high NIO = high in-degree (receiver), not high out-degree (transmitter). All internal comparisons remain valid because the inversion is consistent. However, the paper's interpretive claims about NIO direction may be incorrect.
**Fix options:**
  - **(a) Swap the labels** in `compute_nio()` and `extended_dgp_planted_signal.py` so that `out_flow = column_sum` and `in_flow = row_sum`. This changes NIO sign, so all result CSVs must be regenerated.
  - **(b) Keep the computation, update documentation** to clarify that "NIO" in this codebase is defined as (in-degree - out-degree)/(N-1), and update the paper's narrative accordingly.
  - **(c) Transpose the convention** so that `A[i,j] = i->j`, then row sums naturally give out-degree. This requires changes throughout the codebase.

**Recommendation:** Option (a) is cleanest if the paper's narrative already describes NIO as out-minus-in. Fix the code to match the stated semantics.

### P1: Bare `except:` Clauses (11 instances)

**Impact:** Masks runtime errors including the P0 bug. In a scientific context, silent failures can produce subtly wrong results without any visible indication.
**Fix:** Replace each with specific exception types. The most critical instances:
- `algorithms.py:101` → `except np.linalg.LinAlgError:`
- `algorithms.py:175` → `except (ValueError, TypeError, np.linalg.LinAlgError):`
- `evaluation.py:103` → `except (np.linalg.LinAlgError, ValueError):`

### P2: Duplicate `compute_metrics` Function

**File:** `run_factor_neutral_sim.py:39-77`
**Impact:** If the canonical `compute_precision_recall_f1` in `algorithms.py` is updated (e.g., to fix diagonal handling), this duplicate won't reflect the change.
**Fix:** Delete the local `compute_metrics` function and use the imported `compute_precision_recall_f1` + `eval_metrics` from `evaluation.py`.

### P2: LASSO Test Assertions Too Weak

**File:** `test_algorithms.py:90-115`
**Impact:** Tests pass even when LASSO returns zero edges, providing false confidence.
**Fix:** Add positive assertions:
```python
def test_sparse_network(self):
    ...
    assert adj.sum() >= 1, "LASSO should detect edges in causal data"
    assert density < 0.3, "Network too dense"
```

### P2: Inconsistent Seed Strategy

**File:** `run_factor_neutral_sim.py:149`
**Impact:** Uses `seed = 1000 + trial` rather than the centralized `SimulationConfig` formula. Results from this script aren't controlled by the global seed configuration.
**Fix:** Import and use `SimulationConfig.get_seed(trial, N, T)`.

---

## Optional Enhancements

1. **Add type hints** to public API functions in `algorithms.py`, `evaluation.py`, and `te_core.py`. This improves IDE support and enables `mypy` static analysis.

2. **Consolidate git utility code** into a shared `scripts/_utils.py` module. The `_get_git_commit()` function is duplicated in `results_manager.py`, `simulation_config.py`, and `experiment_metadata.py`.

3. **Add integration tests** that run a minimal end-to-end pipeline: DGP -> TE estimation -> evaluation metrics. This would catch interface mismatches between modules.

4. **Add GARCH/fat-tail stress tests** for the core algorithms. All current tests use Gaussian innovations, but the paper's DGP uses t(5) distributions.

5. **Translate non-English comments** in `nonparametric_te.py` and `all_experiments_v2.py` for international reviewers.

6. **Add execution timing** to `ResultsManager` metadata. Wall-clock time per table is useful for cross-machine benchmarking.

7. **CI/CD pipeline** (GitHub Actions): run `pytest`, verify SHA256 checksums, and check for the `normalize` parameter on every push.

---

## Key Questions Answered

**1. Can I regenerate all tables on a fresh machine?**
No — not currently. The LASSO `normalize=False` bug causes silent failure with scikit-learn >= 1.2. After removing this parameter (a one-line fix), OLS-based tables (2, 5, 6) should reproduce. LASSO-based results need regeneration. Table 5 (empirical) should reproduce exactly given the SHA256-verified data.

**2. Will the code loudly fail if data is corrupted?**
Partially. `empirical_portfolio_sort.py` checks SHA256 and prints a warning on mismatch but continues execution. Other scripts do not verify data checksums at runtime. The code will not silently produce wrong results from corrupted data — pandas will raise parse errors on malformed CSV — but the checksum verification is not universal.

**3. Are conclusions stable under seed variation?**
For OLS-TE: Yes. Benchmark results across seeds 42/100/200 show CV < 1% for precision, recall, and F1. For LASSO-TE: Cannot be verified until the P0 bug is fixed.

**4. Is every function a single authoritative implementation?**
Nearly. The one exception is `compute_metrics()` in `run_factor_neutral_sim.py:39-77`, which duplicates `compute_precision_recall_f1` from `algorithms.py`. All other algorithms have a single implementation.

**5. Are oracle benchmarks measuring what they claim?**
The oracle NIO in `oracle_nio_power.py` correctly uses `NIO_true_std` from the planted signal DGP, ensuring the oracle benchmark matches the planted premium by construction. The comparison between oracle and estimated NIO is methodologically sound. However, the NIO direction inversion (P1) means the semantic interpretation may not match the paper's narrative.

---

## Summary of Required Actions

| Priority | Issue | Fix Effort | Impact |
|----------|-------|-----------|--------|
| **P0** | Remove `normalize=False` from `LassoLarsIC` | 1 line | LASSO completely non-functional |
| **P1** | Fix NIO direction labels or update paper narrative | ~10 lines + regenerate results OR update paper text | Interpretive correctness |
| **P1** | Replace bare `except:` with specific types | ~11 edits | Error visibility |
| **P2** | Remove duplicate `compute_metrics` | Delete function, update caller | Code hygiene |
| **P2** | Strengthen LASSO test assertions | ~2 lines | Test reliability |
| **P2** | Unify seed strategy in `run_factor_neutral_sim.py` | ~3 lines | Reproducibility consistency |

After addressing P0 and P1, this codebase would meet publication-grade standards for computational reproducibility. The infrastructure (SHA256 verification, metadata tracking, modular architecture) is stronger than what most published replication packages provide.
