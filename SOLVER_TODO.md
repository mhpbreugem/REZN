# SOLVER TODO — Posterior Method v3
# Standard: G=20, umax=5, mp50, ‖F‖∞ < 1e-25
# Weighting: ex-ante w = 0.5·(f₀³+f₁³) — ALWAYS weighted 1-R²
# Seed: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json (γ=0.5, τ=2, ‖F‖=7.44e-119)

---

## MASTER RUN TABLE

Two paper figures require REE convergence runs:
- Fig 4B: γ-sweep at τ=2 → 6 runs
- Fig 4A: τ-sweep at γ=0.5, 1.0, 4.0 → 36 runs total (τ=2 shared with Fig 4B)

### Fig 4B — γ-sweep at τ=2

| γ   | 1-R²  | slope | ‖F‖_max   | Status |
|-----|-------|-------|-----------|--------|
| 0.1 | 0.257 | 0.410 | mp50      | ✅ DONE |
| 0.25| 0.133 | 0.507 | mp50      | ✅ DONE |
| 0.5 | 0.088 | 0.523 | 7.44e-119 | ✅ DONE — SEED (mp300) |
| 1.0 | 0.047 | 0.550 | mp50      | ✅ DONE |
| 2.0 | 0.025 | 0.586 | mp50      | ✅ DONE |
| 4.0 | 0.016 | 0.605 | mp50      | ✅ DONE |

All 6 values in `results/full_ree/fig4B_G20_gamma_sweep.json`.
Note: pgfplots file `fig4B_G20_pgfplots.tex` has only 4 REE points (γ=0.5–4.0); needs γ=0.1 and γ=0.25 added.

### Fig 4A — τ-sweep, γ=0.5

| τ    | 1-R²  | ‖F‖_max | Status |
|------|-------|---------|--------|
| 0.3  | 0.037 | mp50*   | ⚠️ ON BRANCH `claude/add-claude-documentation-xGy0S` — pull & verify ‖F‖≤1e-25 |
| 0.5  | 0.038 | mp50*   | ⚠️ ON BRANCH — verify |
| 0.8  | 0.047 | mp50*   | ⚠️ ON BRANCH — verify |
| 1.0  | 0.052 | mp50*   | ⚠️ ON BRANCH — verify |
| 1.5  | 0.064 | mp50*   | ⚠️ ON BRANCH — verify |
| 2.0  | 0.088 | 7.44e-119 | ✅ DONE (seed) |
| 3.0  | 0.138 | mp50*   | ⚠️ ON BRANCH — verify |
| 4.0  | —     | —       | ❌ STUCK — posteriors hit 0/1 at boundary; try UMAX=4 or trim=0.10 |
| 5.0  | —     | —       | ❌ TODO |
| 7.0  | —     | —       | ❌ TODO |
| 10.0 | —     | —       | ❌ TODO |
| 15.0 | —     | —       | ❌ TODO |

### Fig 4A — τ-sweep, γ=1.0

| τ    | 1-R²   | ‖F‖_max | Status |
|------|--------|---------|--------|
| 0.3  | 0.030  | ~1e-12  | ❌ REDO — ‖F‖ < 1e-12, does not meet 1e-25 |
| 0.5  | 0.026  | ~1e-12  | ❌ REDO |
| 0.8  | 0.031  | ~1e-12  | ❌ REDO |
| 1.0  | 0.033  | ~1e-12  | ❌ REDO |
| 1.5  | 0.040  | ~1e-12  | ❌ REDO |
| 2.0  | 0.047  | mp50    | ✅ DONE (shared with Fig 4B) |
| 3.0  | 0.066  | ~1e-12  | ❌ REDO |
| 4.0  | 0.090⚠️| ~2.3e-5 | ❌ REDO |
| 5.0  | —      | —       | ❌ TODO |
| 7.0  | —      | —       | ❌ TODO |
| 10.0 | —      | —       | ❌ TODO |
| 15.0 | —      | —       | ❌ TODO |

### Fig 4A — τ-sweep, γ=4.0

| τ    | 1-R²  | ‖F‖_max | Status |
|------|-------|---------|--------|
| 0.3  | —     | ~1e-3   | ❌ REDO — wrong solver (K3 staggered NK float64, not posterior-method v3) |
| 0.5  | —     | ~1e-3   | ❌ REDO |
| 0.8  | —     | ~1e-3   | ❌ REDO |
| 1.0  | —     | ~1e-3   | ❌ REDO |
| 1.5  | —     | ~1e-3   | ❌ REDO |
| 2.0  | 0.016 | mp50    | ✅ DONE (shared with Fig 4B) |
| 3.0  | —     | ~1e-3   | ❌ REDO |
| 4.0  | —     | ~1e-3   | ❌ REDO |
| 5.0  | —     | ~1e-3   | ❌ REDO |
| 7.0  | —     | ~1e-3   | ❌ REDO |
| 10.0 | —     | ~1e-3   | ❌ REDO |
| 15.0 | —     | ~1e-3   | ❌ REDO |

### Summary

| Status | Count | Notes |
|--------|-------|-------|
| ✅ DONE | 8 | Seed + 5 Fig4B + 1 for Fig4A τ=2 per γ=1,4 (counted in Fig4B) |
| ⚠️ ON BRANCH | 6 | γ=0.5, τ=0.3–3.0 excl τ=2; pull & verify ‖F‖ |
| ❌ REDO (‖F‖>1e-25) | 8 | γ=1.0, τ=0.3–1.5,3.0,4.0 (wrong precision) |
| ❌ REDO (wrong solver) | 11 | γ=4.0, all τ≠2 |
| ❌ TODO (not attempted) | 9 | γ=0.5 τ=4–15 (4); γ=1.0 τ=5–15 (4); γ=0.5 τ=4 stuck (1) |
| **Total runs** | **36** | |

---

## PRIORITY TASKS

### P0 — REDO γ=4.0 τ-sweep (11 runs, fastest — closest to CARA)

Use posterior-method v3 mp50. Warm-start from γ=4.0 τ=2 checkpoint (in fig4B sweep).
Expected 1-R² range: 0.001–0.05.

```
τ=2 (done) → τ=3 → τ=4 → τ=5 → τ=7 → τ=10 → τ=15   (walk up)
τ=2 (done) → τ=1.5 → τ=1.0 → τ=0.8 → τ=0.5 → τ=0.3  (walk down)
```

Save each to: `results/full_ree/task3_g400_tXXXX_mp50.json`
Measure weighted 1-R² at convergence.

### P1 — REDO γ=1.0 τ-sweep (7 redo + 4 new)

Same strategy as P0. Previous runs achieved ‖F‖~1e-12 which does not meet the 1e-25 standard.

```
τ=2 (done) → τ=3 → τ=4 → τ=5 → τ=7 → τ=10 → τ=15
τ=2 (done) → τ=1.5 → τ=1.0 → τ=0.8 → τ=0.5 → τ=0.3
```

Save each to: `results/full_ree/task3_g100_tXXXX_mp50.json`

### P2 — Pull and verify γ=0.5 τ-sweep from branch

Pull `claude/add-claude-documentation-xGy0S`, check ‖F‖ for each of the 6 points.
If ‖F‖ ≤ 1e-25: accept, copy to main. If not: redo.
Then attempt τ=4.0 and above (try trim=0.10 or umax=4 to handle boundary issue).

### P3 — Assemble Fig 4A pgfplots

Once all τ-sweep runs are done for all 3 γ values:
- Collect (τ, 1-R²) for each γ
- Build `results/full_ree/fig4A_G20_pgfplots.tex`

Format:
```latex
% γ=0.5
\addplot coordinates {(0.3,0.037)(0.5,0.038)...};
% γ=1.0
\addplot coordinates {...};
% γ=4.0
\addplot coordinates {...};
```

### P4 — Fix fig4B pgfplots file

Add γ=0.1 and γ=0.25 REE coordinates to `fig4B_G20_pgfplots.tex`.
Data is already in `fig4B_G20_gamma_sweep.json`.

---

## FIGURE QUALITY (needed before journal submission)

### Figures with real data — WHITE background
| Fig | Source | Notes |
|-----|--------|-------|
| 1 | `fig2_knife_edge.tex` | No-learning 1-R² vs τ, G=15, γ=0.5/1/4 — check gammas |
| 3A | Analytical | CARA contours (straight) ✅ |
| 3B | `fig3B_G18_convex_pgfplots.tex` | G=18 mp300 convexity-enforced ✅ |
| 4B | `fig4B_G20_pgfplots.tex` | REE + NL vs γ — needs γ=0.1,0.25 added (P4) |
| 6A | Analytical | CARA posteriors Λ(T*/3) ✅ |
| 11 | `fig11_G20_pgfplots.tex` | Mechanisms bar chart ✅ |
| R1 | `figR1_G20_pgfplots.tex` | 1-R² vs K agents ✅ |

### Figures needing data — GRAY background
| Fig | What's needed | Location of data |
|-----|---------------|-----------------|
| 4A | τ-sweep 3 curves | This todo (P0–P2) |
| 5 | Price vs T* (asymmetric triples) | Done on branch `claude/extractions-oC85I` — pull |
| 6B | CRRA posteriors (artifact-free) | Done on branch `claude/extractions-oC85I` using G=18 — pull |
| 10 | Convergence path ‖F‖ vs iter | Done on branch `claude/extractions-oC85I` — pull |
| R2 | Lognormal variant | Done on branch `claude/extractions-oC85I` — pull |
| 7 | Trade volume | Skipped for SSRN |
| 8 | Value of information | Skipped for SSRN |
| 9 | GS resolution | Skipped for SSRN |

**Immediate action:** Pull `claude/extractions-oC85I` → Figs 5, 6B, 10, R2 are all done there.

---

## TECHNICAL NOTES

### Weighted 1-R² formula (ALWAYS use this)
```python
def signal_density(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2*np.pi)) * np.exp(-tau/2 * (u - mean)**2)

# For each triple (i,j,l):
w = 0.5 * (signal_density(u1,0,tau)*signal_density(u2,0,tau)*signal_density(u3,0,tau)
          + signal_density(u1,1,tau)*signal_density(u2,1,tau)*signal_density(u3,1,tau))

slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))

pred = slope * Tstar + intercept
mean_lp = np.average(logit_p, weights=weights)
var_tot = np.average((logit_p - mean_lp)**2, weights=weights)
var_res = np.average((logit_p - pred)**2, weights=weights)
weighted_1mR2 = var_res / var_tot
```

### File naming
```
results/full_ree/task3_g{gamma*100:03d}_t{tau*10:04d}_mp50.json
  e.g. g050_t0020 = γ=0.5, τ=2.0
       g100_t0030 = γ=1.0, τ=0.3
       g400_t0150 = γ=4.0, τ=15.0
```

### Strict convergence definition
‖F‖∞ ≤ 1e-25 over all active cells. Zero u-monotonicity violations. Zero p-monotonicity violations.
