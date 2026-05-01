# FINAL COMPUTATION PLAN — ALL FIGURES
# Updated 2026-05-01
# Standard gammas: γ = 0.5, 1.0, 4.0

## CRITICAL: PAPER NUMBERS NEED UPDATING
The mp100 Newton-converged result gives DIFFERENT values than float64:
- 1-R² = 0.077 (was 0.108 in the paper)
- slope = 0.541 (was 0.341 in the paper)
The float64 posteriors had numerical wiggles that inflated the deficit.
ALL paper numbers need updating from the mp50+ converged solutions.

---

## A. DONE — NO COMPUTATION NEEDED

| Fig | Description | Status |
|-----|-------------|--------|
| 1   | Knife-edge 1-R² vs τ (no-learning) | ✅ Real G=15 data, γ=0.5/1/4 |
| 3A  | CARA contours (straight lines) | ✅ Analytical + solver endpoints |
| 6A  | CARA posteriors μ=p=Λ(T*/3) | ✅ Analytical, one sigmoid |

---

## B. CHEAP — NO-LEARNING (seconds each, any precision)

| Fig | Description | What to run |
|-----|-------------|-------------|
| R1  | Robustness: 1-R² vs K | No-learning at K=3..20, γ=0.5/1/4, τ=2 |
| R2  | Robustness: lognormal variant | No-learning with lognormal payoff |
| 11  | Mechanisms bar chart | No-learning at 6 het-γ/het-τ configs |

γ = 0.5, 1.0, 4.0 for R1 and R2. Check existing figures use these values.

---

## C. EXTRACT FROM EXISTING mp300 G=15 γ=0.5 τ=2 (no new runs)

| Fig | What to extract | Method |
|-----|-----------------|--------|
| 5   | Price vs T* (FR, NL, REE) | 50 symmetric triples u₁=u₂=u₃=T*/(3τ), solve MC with μ* |
| 6B  | Posteriors μ₁(u=+1), μ₂(u=-1), p vs T* | Evaluate μ*(u,p) at representative triples, clip T*∈[-3,4] |
| 10  | Convergence path (appendix) | Use mp100/mp300 iteration logs ||

---

## D. NEW RUNS NEEDED

### D1. Fig 3B — CRRA contours (THE showpiece figure)
**Precision:** mp300, **G=25** (currently running)
**Tolerance:** ||F||∞ < 10⁻¹⁰⁰
**γ=0.5, τ=2, u₁=1**
**After G=25 converges:**
1. Load converged μ* at G=25
2. Evaluate P(1, u₂, u₃) on 200×200 fine grid using interpolated μ*
3. matplotlib.pyplot.contour at p ∈ {0.2, 0.3, 0.5, 0.7, 0.8}
4. Arc-length resample each contour to ~50 points
5. Output pgfplots coordinates

### D2. Fig 4A — 1-R² vs τ at 3 γ values (the τ sweep)
**Precision:** mp50, **G=15**
**Tolerance:** ||F||∞ < 10⁻²⁵ (NOT machine precision — see below)
**36 runs:** γ ∈ {0.5, 1.0, 4.0} × τ ∈ {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0}

**WHY 10⁻²⁵ NOT MACHINE PRECISION:**
Machine precision at mp50 is ~10⁻⁵⁰. But converging to 10⁻⁵⁰ means
fitting interpolation artifacts of the G=15 grid to 50 digits.
The interpolation error is O(h²) ≈ 0.01 — fitting that to 50 digits
is wasted work and can introduce spurious structure.
Empirically: 1-R² stabilizes well above 10⁻²⁵. Going from 10⁻¹⁵ to
10⁻⁵⁵ changed 1-R² by 29% (0.108→0.077). Going from 10⁻⁵⁵ to
10⁻²⁰⁷ changed nothing. So 10⁻²⁵ captures the true answer with
margin to spare, and each NK iteration is faster (fewer bisection
steps, less Jacobian precision needed).

**Warm-start strategy (critical for speed):**
For each γ, warm-start from the τ=2 converged solution:
```
τ=2.0 (existing) → τ=1.5 → τ=1.0 → τ=0.8 → τ=0.5 → τ=0.3  (walk down)
τ=2.0 (existing) → τ=3.0 → τ=4.0 → τ=5.0 → τ=7.0 → τ=10.0 → τ=15.0  (walk up)
```
Each step: interpolate μ* onto new p-grid, 2-3 NK iterations → converged.

**Extract:** 1-R² and slope at each (γ, τ) point.

### D3. Fig 4B — 1-R² vs γ at τ=2 (two missing points)
**Precision:** mp50, **G=15**
**Tolerance:** ||F||∞ < 10⁻²⁵
**2 new runs:** γ=0.1, γ=0.25 (γ=0.5,1,2,4 already have mp100+ results)

**Extract:** 1-R² at each γ. Re-measure 1-R² for existing γ values
from mp100 converged μ* (current paper numbers are from float64).

---

## E. SKIP FOR SSRN

| Fig | Description | Why skip |
|-----|-------------|----------|
| 4A  | (if too expensive) | Knife-edge Fig 1 shows same τ-dependence |
| 7   | Trade volume vs γ | Needs 8 convergence runs |
| 8   | Value of info V(τ) | Needs 45 convergence runs |
| 9   | GS resolution V(τ)-c | Derived from Fig 8 |

---

## F. PAPER NUMBERS TO UPDATE (from mp100/mp50 results)

After D2/D3 complete, update main.tex:
- 1-R² at γ=0.5 τ=2: 0.108 → 0.077
- slope at γ=0.5 τ=2: 0.341 → 0.541
- γ-ladder table: all values from mp50+ convergence
- Posteriors table at (1,-1,1): recompute from mp300 μ*
- G-ladder table: add G=25 mp300 value

---

## G. PRIORITY ORDER

1. **Wait for G=25 mp300 to finish** → extract Fig 3B contours
2. **Run D3** (2 runs, mp50, ~1 hour) → complete Fig 4B
3. **Run D2** (36 runs, mp50, ~12-18 hours) → Fig 4A
4. **Extract C** (existing data) → Figs 5, 6B, 10
5. **Run B** (cheap, seconds) → Figs R1, R2, 11
6. **Update paper numbers** from all mp50+ results

---

## H. OUTPUT FORMAT

All results save to: `results/full_ree/`

```
fig_NAME_data.json          — raw data with full precision
fig_NAME_pgfplots.tex       — ready-to-paste pgfplots coordinates
```

pgfplots format:
```latex
% gamma=0.5
\addplot coordinates {(x1,y1)(x2,y2)...};
```
