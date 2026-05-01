# FIGURES TODO — UPDATED 2026-05-01
# Standard gammas: γ = 0.5, 1.0, 4.0 (USE THESE EVERYWHERE)
# G = 15 for all REE figures (strict convergence)
# Output: JSON data + pgfplots coordinates (.tex)
# Save to: results/full_ree/

## CRITICAL: STANDARD PARAMETERS
```
γ_paper = [0.5, 1.0, 4.0]   ← USE THESE FOR ALL MULTI-γ FIGURES
τ_default = 2.0
W = 1.0
K = 3
G = 15 (strict convergence)
```

---

## STATUS SUMMARY

| Fig | Name | Status | Data source |
|-----|------|--------|-------------|
| 1 | Knife-edge (1-R² vs τ) | ✅ DONE | Real, G=15, γ=0.5/1/4 |
| 3A | Contour CARA | ⬜ GRAY | Projected |
| 3B | Contour CRRA | ⬜ GRAY | Projected |
| 4A | REE panels (1-R² vs τ) | ⬜ GRAY | Projected, needs γ=0.5/1/4 sweeps |
| 4B | REE panels (1-R² vs γ) | ✅ DONE | Real, G=15, 6 γ values |
| 5 | REE vs no-learning (p vs T*) | ⬜ GRAY | Binning artifacts, needs redo |
| 6A | Posteriors CARA | ⬜ GRAY | Solver gave no-learning, need REE |
| 6B | Posteriors CRRA | ⬜ GRAY | Step-like, needs higher G or interp |
| 7 | Trade volume vs γ | ⬜ GRAY | Invented |
| 8 | Value of info V(τ) | ⬜ GRAY | Invented |
| 9 | GS resolution V(τ)-c | ⬜ GRAY | Derived from Fig 8 |
| 10 | Convergence (appendix) | ⬜ GRAY | Projected |
| 11 | Mechanisms bar chart | ⬜ GRAY | No-learning data |
| R1 | Robustness: K agents | ⚠️ CHECK | May have wrong γ values |
| R2 | Robustness: lognormal | ⚠️ CHECK | May have wrong γ values |

---

## PRIORITY 1: MUST-HAVE FOR SSRN

### Fig 5: REE vs no-learning price function (p vs T*)
**Status:** REJECTED — binning artifacts, outlier at T*=8.25
**γ = 0.5, τ = 2, G = 15**
**Fix:** Do NOT bin random triples. Instead:
1. Choose 50 evenly-spaced T* values from -8 to 8
2. For each T*, use symmetric triple u₁=u₂=u₃=T*/(3τ)
3. Compute: p_FR = Λ(T*/3)
4. Compute: p_NL = solve Σ x(Λ(τuₖ), p) = 0
5. Compute: p_REE = solve Σ x(μ*(uₖ, p), p) = 0 using converged μ*
6. Output three smooth curves, no binning
**Time:** ~20 min

### Fig 10: Convergence path (appendix)
**Status:** GRAY — solver has real data but Picard phase 1 oscillates
**γ = 0.5, τ = 2, G = 15**
**Fix:** Plot the "best-so-far" envelope (monotone decreasing):
```python
best = residuals[0]
for i, r in enumerate(residuals):
    best = min(best, r)
    best_so_far[i] = best
```
This removes the oscillation noise and shows clean convergence.
Or: just plot phase 2 (Cesaro, α=0.01) which converges cleanly.
**Time:** ~10 min (reformat existing data)

### Fig R1 + R2: Robustness panels — CHECK GAMMAS
**Status:** May use old γ values (0.2, 1, 5)
**Fix:** Verify. If wrong, recompute at γ = 0.5, 1, 4
- Fig R1: 1-R² vs K for K=3..20, at τ=2
- Fig R2: 1-R² vs τ for lognormal payoff variant
Both are no-learning — fast computation.
**Time:** ~15 min if recompute needed

---

## PRIORITY 2: SHOULD-HAVE

### Fig 4A: REE panels (1-R² vs τ) — τ sweep at three γ values
**Status:** GRAY — γ=0.5 has partial data (from knife-edge sweep at REE)
**γ = 0.5, 1.0, 4.0 at G=15**
**Data needed:**
- For each γ in [0.5, 1.0, 4.0]:
  - Converge REE at 12 log-spaced τ from 0.3 to 8
  - Record 1-R² at each (γ, τ)
**Time:** ~6 hours (each point ~30 min)

### Fig 3A + 3B: Multi-contour (7 price levels, CARA vs CRRA)
**Status:** GRAY — projected contour shapes
**γ = 0.5, τ = 2, G = 15**
**Data needed:**
- At u₁=1:
  - CARA: compute P_CARA(1, u₂, u₃) = solve Σ (logit(Λ(τuₖ))-logit(p))/γ = 0
    Extract contours at 7 evenly-spaced prices from 0.15 to 0.85
  - CRRA: use converged μ*, compute P_REE(1, u₂, u₃)
    Extract contours at same 7 prices
- Output: 7 coordinate lists per panel
**Time:** ~20 min

### Fig 6A: Posteriors CARA (REE, not no-learning!)
**Status:** REJECTED — solver gave no-learning version (flat μ lines)
**Fix:** Under CARA REE, all posteriors equal the price:
  μ₁ = μ₂ = μ₃ = p = Λ(T*/3)
This is ANALYTICAL — just one sigmoid line. No solver needed.
Plot: one thick black line = Λ(T*/3) from T*=-10 to 10.
Label: "μ₁ = μ₂ = μ₃ = p"
**Time:** 0 (analytical, I can build this in pgfplots)

### Fig 6B: Posteriors CRRA (REE)
**Status:** REJECTED — step-like flat regions from G=15 grid
**γ = 0.5, τ = 2, G = 15**
**Fix:** The flat regions occur because μ(u,p) hits the p-grid boundary.
Options:
  (a) Clip T* range to [-2, 3] (transition zone only)
  (b) Interpolate between grid points for smoother curves
  (c) Use higher G (G=20+) for this figure only
**Preferred:** Option (a) — clip to [-2, 4], show the fan-out clearly.
The posteriors table in the text has the exact numbers.
**Time:** ~10 min (reprocess existing data)

---

## PRIORITY 3: NICE-TO-HAVE

### Fig 7: Trade volume E[|xₖ|] vs γ
**γ = 0.5, 1.0, 4.0 + additional points**
For each γ in [0.1, 0.25, 0.5, 1, 2, 4, 10, 50]:
  - Converge REE at G=15, τ=2
  - Compute E[|xₖ|] averaged over signal distribution
**Time:** ~4 hours

### Fig 8: Value of information V(τ)
**γ = 0.5, 1.0, 4.0**
For each γ, for 10-15 τ values from 0 to 5:
  - Converge REE, compute V(τ) = E[U(W+x*(v-p))] - E[U(W)]
**Time:** ~8 hours

### Fig 9: GS resolution V(τ)-c
Derived from Fig 8 data. No additional computation.

### Fig 11: Mechanisms bar chart
No-learning computation at heterogeneous configurations.
Already have most data. Need het-α CARA entry.
**Time:** ~30 min

---

## OUTPUT FORMAT

For each figure, save:
```
results/full_ree/fig_NAME_data.json     — raw data
results/full_ree/fig_NAME_pgfplots.tex  — ready-to-paste coordinates
```

JSON format:
```json
{
  "figure": "fig_NAME",
  "params": {"G": 15, "gamma": [0.5, 1, 4], "tau": 2.0},
  "curves": [
    {"name": "gamma_0.5", "points": [{"x": ..., "y": ...}, ...]},
    ...
  ]
}
```

pgfplots format:
```latex
% gamma=0.5
\addplot coordinates {(x1,y1)(x2,y2)...};

% gamma=1.0
\addplot coordinates {(x1,y1)(x2,y2)...};
```

## FIGURE STYLE REMINDER

All figures use the same pgfplots template:
- width=7.5cm, height=7.5cm, scale only axis
- Colors: BCgreen(0.11,0.35,0.02), BCred(0.7,0.11,0.11), BCblue(0,0.20,0.42)
- Line order: green solid (γ=0.5), red dashed (γ=1), blue dotted (γ=4), black dashdotted (CARA)
- very thick curves, ultra thick CARA
- ticklabel: scriptsize, fixed format
- yticklabel/ylabel: text width=10mm
- groupplot wrapper even for standalone figures

## KEY REMINDERS
- γ = 0.5, 1, 4 EVERYWHERE (not 0.25)
- G = 15 for REE (strict convergence)
- G = 15 or 20 for no-learning (fast, can use higher G)
- Fig 5 (p vs T*): DO NOT BIN. Use symmetric triples.
- Fig 6A (CARA posteriors): ANALYTICAL. One line: μ=p=Λ(T*/3). No solver.
- Fig 6B (CRRA posteriors): CLIP to T* ∈ [-2, 4].

---

## URGENT: Fig 3B high-resolution contours

The current Fig 3B contours have wobbles from the G=15 discrete grid.
We smoothed with cubic splines but the underlying data needs to be
computed at higher resolution.

**Task:** Recompute CRRA contour lines on a fine grid.

**Method:**
1. Load converged `mu_star` at G=15, γ=0.5, τ=2
2. Fix u₁=1 (agent 1's slice)
3. Create a **200×200** grid in (u₂, u₃) ∈ [-3.5, 3.5]²
4. At each (u₂, u₃), interpolate μ* and solve market clearing to get REE price
5. Use `matplotlib.pyplot.contour` to extract contour lines at p ∈ {0.2, 0.3, 0.5, 0.7, 0.8}
6. Output each contour as a list of (u₂, u₃) coordinates

**Key:** Do NOT trace contours on the G=15 grid with root-finding.
Instead evaluate the price function on the 200×200 fine grid by
interpolating μ* (use scipy.interpolate.RegularGridInterpolator on the
G=15 μ*(u,p) grid), then solve market clearing at each fine-grid point,
then let matplotlib's marching-squares find smooth contours.

**Standard gammas: γ = 0.5, 1.0, 4.0** (use γ=0.5 for this figure)

**Output:**
```
results/full_ree/fig_multicontour_B_hires_pgfplots.tex
```

Format:
```latex
% p=0.2
\addplot coordinates {(x1,y1)(x2,y2)...};
% p=0.3
\addplot coordinates {(x1,y1)(x2,y2)...};
% p=0.5
\addplot coordinates {(x1,y1)(x2,y2)...};
% p=0.7
\addplot coordinates {(x1,y1)(x2,y2)...};
% p=0.8
\addplot coordinates {(x1,y1)(x2,y2)...};
```

Thin each contour to ~40-50 points for pgfplots (evenly spaced along arc length).
