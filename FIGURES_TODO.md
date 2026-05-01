# FIGURE GENERATION LIST FOR SOLVER — G=15, SSRN VERSION
# All figures use the v2 groupplot template (7cm x 7cm, scale only axis)
# Output: pgfplots coordinate lists OR raw data JSON files

## PARAMETERS
- G = 15 (strict convergence, ||F||∞ < 1e-14)
- γ_paper = [0.25, 1.0, 4.0] for all multi-γ figures
- τ_default = 2.0
- W = 1.0, K = 3

---

## FIG 1: KNIFE-EDGE (no-learning, 1-R² vs τ)
**File:** fig_knife_edge.pdf
**Status:** GRAY BG — wrong gammas (was 0.2, 1, 5)
**Data needed:**
- For each γ in [0.25, 1.0, 4.0]:
  - Compute no-learning 1-R² at 30 log-spaced τ from 0.1 to 20
  - G=15 (or G=20 — this is no-learning, no REE needed)
- Output: three lists of (τ, 1-R²) coordinates
**Computation:** ~5 min (no-learning is fast)

## FIG 3A: MULTICONTOUR CARA (contour lines at multiple prices)
**File:** fig_multicontour_A.pdf
**Status:** GRAY BG — invented contour shapes
**Data needed:**
- At u₁=1, τ=2, CARA demands:
  - Compute no-learning price P(1, u₂, u₃) on G=15 grid
  - Extract contour lines at 7 price levels spanning [0.2, 0.8]
  - Output: 7 lists of (u₂, u₃) coordinates per contour
**Computation:** ~2 min

## FIG 3B: MULTICONTOUR CRRA (contour lines at multiple prices)
**File:** fig_multicontour_B.pdf
**Status:** GRAY BG — invented contour shapes
**Data needed:**
- At u₁=1, τ=2, γ=0.5, converged REE μ* at G=15:
  - Compute REE price P(1, u₂, u₃) by solving market clearing with μ*
  - Extract contour lines at the SAME 7 price levels as CARA
  - Output: 7 lists of (u₂, u₃) coordinates per contour
**Computation:** ~10 min (needs converged μ*)

## FIG 4A: REE PANELS — 1-R² vs τ
**File:** fig_ree_panels_A.pdf
**Status:** GRAY BG — γ=1,4 curves projected
**Data needed:**
- For each γ in [0.25, 1.0, 4.0]:
  - Converge REE at 12-16 log-spaced τ from 0.2 to 8
  - Record 1-R² at each (γ, τ)
- Output: three lists of (τ, 1-R²) coordinates
**Computation:** ~3-6 hours (each (γ,τ) takes ~10-30 min at G=15)
**NOTE:** γ=0.25 sweep already has 14 points from solver branch.
Need γ=1 and γ=4 sweeps.

## FIG 4B: REE PANELS — 1-R² vs γ
**File:** fig_ree_panels_B.pdf
**Status:** PARTIALLY REAL — has γ=0.3,0.5,1,2 REE data
**Data needed:**
- Add γ = 0.1, 0.25, 4.0 REE at τ=2
- Add no-learning 1-R² at same γ values (already computed)
- Output: two lists of (γ, 1-R²) for REE and no-learning
**Computation:** ~1 hour (3 new γ points)

## FIG 5: REE vs NO-LEARNING PRICE FUNCTION
**File:** fig_ree_vs_nolearning.pdf
**Status:** GRAY BG — REE curve is interpolated
**Data needed:**
- Converged μ* at G=15, γ=0.5, τ=2
- For each triple (u_i, u_j, u_l) on G=15 grid (3375 points):
  - T* = τ(u_i + u_j + u_l)
  - p_NL = solve market clearing with private priors
  - p_REE = solve market clearing with converged μ*
  - p_FR = Λ(T*/K)
- Bin by T* (40 bins from -10 to 10), compute mean p in each bin
- Output: three lists of (T*, p) for FR, NL, REE
**Computation:** ~20 min

## FIG 6A: POSTERIORS CARA
**File:** fig4_posteriors_A.pdf
**Status:** WHITE BG but projected data
**Data needed:**
- Plot Λ(T*/3) as function of T* (analytical, no solver needed)
**Computation:** 0 (analytical)

## FIG 6B: POSTERIORS CRRA
**File:** fig4_posteriors_B.pdf
**Status:** GRAY BG — projected fan-out
**Data needed:**
- Converged μ* at G=15, γ=0.5, τ=2
- For range of T*, at representative signal splits:
  - Agent with u=+1: compute μ₁ from μ*(u=+1, p_REE)
  - Agent with u=-1: compute μ₂ from μ*(u=-1, p_REE)
  - Price p_REE from market clearing
- Output: three lists of (T*, value) for μ₁, μ₂, p
**Computation:** ~10 min

## FIG 7: TRADE VOLUME vs γ
**File:** fig7_volume.pdf
**Status:** GRAY BG — invented values
**Data needed:**
- For each γ in [0.1, 0.25, 0.5, 1, 2, 5, 10, 50]:
  - Converged REE μ* at G=15, τ=2
  - Compute E[|x_k|] = (1/K) Σ_k E[|demand(μ_k, p)|]
    averaged over signal distribution
- Output: list of (γ, E[|x|]) coordinates
**Computation:** ~4 hours (8 γ points × ~30 min each)

## FIG 8: VALUE OF INFORMATION
**File:** fig8_value_info.pdf
**Status:** GRAY BG — invented values
**Data needed:**
- For each γ in [0.25, 1.0, 4.0]:
  - For 10-15 τ values from 0 to 5:
    - Converge REE at (γ, τ)
    - Compute V(τ) = E[U(W + x*(v-p))] - E[U(W)] at the REE
    - This is the ex-ante expected utility gain from having a signal
- Output: three lists of (τ, V) coordinates
**Computation:** ~8 hours (45 points × ~10 min each)
**NOTE:** Most expensive figure. Consider subset first.

## FIG 9: GS RESOLUTION (V(τ) - c)
**File:** fig9_GS.pdf
**Status:** GRAY BG — invented values
**Data needed:**
- Uses V(τ) from Fig 8 at τ=2
- Plot V - c for c from 0 to max(V)
- Output: derived from Fig 8 data, no new computation
**Computation:** 0 (uses Fig 8 output)

## FIG 10: CONVERGENCE (appendix)
**File:** fig5_convergence.pdf
**Status:** GRAY BG — projected Picard+NK path
**Data needed:**
- Re-run posterior method at G=15, γ=0.5, τ=2
- Save ||F||∞ at each iteration (Picard and NK phases)
- Output: list of (iteration, ||F||∞) coordinates
**Computation:** ~30 min (one convergence run with logging)

## FIG 11: MECHANISMS BAR CHART
**File:** fig6_mechanisms.pdf
**Status:** GRAY BG — uses no-learning numbers from Table
**Data needed:**
- Run no-learning at het-γ and het-τ configurations (Table in paper)
- Already have most values from project_summary.txt
- Need: het-α CARA entry (to show it goes to zero at REE)
- Output: bar heights for 6 configurations
**Computation:** ~30 min (no-learning runs)

## FIG R1: ROBUSTNESS — K agents
**File:** fig_knife_edge_K.pdf
**Status:** WHITE BG but CHECK gammas
**Data needed:**
- Verify uses γ=0.25, 1, 4 (not old values)
- If wrong: recompute no-learning 1-R² vs K for K=3..20
**Computation:** ~10 min if recompute needed

## FIG R2: ROBUSTNESS — lognormal
**File:** fig_knife_edge_lognormal.pdf
**Status:** WHITE BG but CHECK gammas
**Data needed:**
- Same check as K figure
**Computation:** ~10 min if recompute needed

---

## PRIORITY ORDER (for SSRN deadline)

### Must-have (paper is incomplete without these):
1. Fig 1: knife-edge at correct gammas (~5 min) ← DO FIRST
2. Fig 5: REE vs NL price function (~20 min) ← core result
3. Fig 10: convergence path (~30 min) ← proves existence
4. Fig 4B: REE panels γ-sweep (~1 hour) ← 3 new points

### Should-have (strengthen the paper):
5. Fig 3A+3B: multi-contour at 7 price levels (~15 min) ← mechanism
6. Fig 6B: posteriors fan-out (~10 min) ← visual intuition
7. Fig 4A: REE panels τ-sweep (~3-6 hours) ← comparative statics

### Nice-to-have (can submit without):
8. Fig 7: trade volume (~4 hours)
9. Fig 8+9: value of info + GS (~8 hours)
10. Fig 11: mechanisms bar chart (~30 min)
11. Fig R1+R2: check robustness gammas (~20 min)

### Total computation time estimate:
- Must-have: ~2 hours
- Should-have: ~4-7 hours  
- Nice-to-have: ~13 hours
- Grand total: ~19-22 hours of solver time

---

## OUTPUT FORMAT

For each figure, the solver should output a JSON file:
```json
{
  "figure": "fig_knife_edge",
  "params": {"G": 15, "tau_range": [0.1, 20], "gammas": [0.25, 1, 4]},
  "curves": [
    {"gamma": 0.25, "points": [{"tau": 0.1, "1-R2": 0.025}, ...]},
    {"gamma": 1.0, "points": [{"tau": 0.1, "1-R2": 0.003}, ...]},
    {"gamma": 4.0, "points": [{"tau": 0.1, "1-R2": 0.000}, ...]}
  ]
}
```

Plus a pgfplots coordinate string ready to paste:
```
% gamma=0.25
\addplot coordinates {(0.1,0.025)(0.2,0.040)...};
```

Save to: results/full_ree/fig_NAME_data.json
Save pgfplots to: results/full_ree/fig_NAME_pgfplots.tex

---

## FIGURE QUALITY REVIEW (2026-05-01)

### INCLUDED (white background, real data):
- Fig 1 knife-edge: 30 pts/curve, G=15, γ=0.25/1/4. EXCELLENT. ✓

### REJECTED (kept as gray-bg placeholder):
- Fig 5 REE vs NL: Only 20 bins. Severe outlier at T*=8.25 (price drops
  from 0.973 to 0.803). Fix: use 80+ bins, or compute NL prices analytically
  at evenly-spaced T* values instead of binning random triples.
  
- Fig 6B CRRA posteriors: Only 17 points. Step-like flat regions at extremes
  (μ_high stuck at 0.333 for T*<0, μ_low stuck at 0.667 for T*>2).
  Fix: evaluate μ(u,p) directly from converged posterior function at 50+
  T* values, not from binning grid triples.

### KEY ISSUE FOR SOLVER:
The binning approach (sample G³ triples, bin by T*) gives too few points
at the extremes. Better approach for Fig 5 and 6B:
1. Choose 50 evenly-spaced T* values from -10 to +10
2. For each T*, find representative signal triples with that T*
   (e.g., symmetric u₁=u₂=u₃=T*/(3τ))
3. Compute the exact no-learning and REE prices for those triples
4. No binning artifacts, no outliers
