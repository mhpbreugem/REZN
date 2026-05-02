# DEFINITIVE FIGURE COMPUTATION PLAN
# Updated 2026-05-02
# Seed: G=20 UMAX=5 notrim mp300, γ=0.5 τ=2 (||F||=7.4e-119)

## STANDARD PARAMETERS
```
γ_paper = [0.5, 1.0, 4.0]
G = 20, UMAX = 5 (grid: [-5, 5], spacing = 0.526)
Precision: mp50 for sweeps, mp300 for showpiece
Tolerance: ||F|| < 1e-25 (mp50), 1e-100 (mp300)
```

## SEED AVAILABLE
```
G=20 UMAX=5 notrim mp300: γ=0.5 τ=2 → ||F||=7.4e-119 ✅
```
All runs below warm-start from this seed or from neighboring converged points.

---

## A. ZERO RUNS — ANALYTICAL OR EXISTING

| Fig | Description | Source | Status |
|-----|-------------|--------|--------|
| 1   | Knife-edge 1-R² vs τ | No-learning, G=15 | ✅ DONE |
| 3A  | CARA contours | Analytical straight lines | ✅ DONE |
| 6A  | CARA posteriors | Analytical Λ(T*/3) | ✅ DONE |

---

## B. ZERO RUNS — EXTRACT FROM SEED (G=20 umax5 γ=0.5 τ=2)

| Fig | What to extract | Method |
|-----|-----------------|--------|
| 3B  | CRRA contour lines at 5 prices | 300×300 fine grid, market clearing with μ*, matplotlib contour |
| 5   | Price vs T* (FR, NL, REE) | 50 symmetric triples u₁=u₂=u₃=T*/(3τ), solve MC |
| 6B  | Posteriors μ₁(u=+1), μ₂(u=-1), p vs T* | Evaluate μ*(u,p) at representative triples |
| 10  | Convergence path | Use iteration history from mp300 convergence log |

**Fig 3B note:** G=20 umax=5 gives spacing 0.526. Use 300×300 fine grid
+ Gaussian smoothing σ=1.5 + arc-length resample to 50 pts per contour.
If G=25 finishes, redo from G=25 (spacing 0.42, even smoother).

---

## C. NEW CONVERGENCE RUNS

### C1. Fig 4B — γ-sweep at τ=2 (complete the ladder)
**Precision:** mp50, G=20, UMAX=5, tol < 1e-25

| γ | Warm-start from | Status |
|---|-----------------|--------|
| 0.5 | SEED (existing mp300) | ✅ DONE — just measure 1-R² |
| 1.0 | γ=0.5 seed | 🔲 RUN |
| 2.0 | γ=1.0 | 🔲 RUN |
| 4.0 | γ=2.0 | 🔲 RUN |
| 0.25 | γ=0.5 seed | 🔲 RUN |
| 0.1 | γ=0.25 | 🔲 RUN |

**6 points, 5 new runs.** Walk outward from γ=0.5:
```
γ=0.5 (done) → γ=1.0 → γ=2.0 → γ=4.0
γ=0.5 (done) → γ=0.25 → γ=0.1
```
**Extract:** 1-R² and slope at each γ.
**Also extract:** no-learning 1-R² at same γ values (cheap, no REE needed).

### C2. Fig 4A — τ-sweep at 3 γ values
**Precision:** mp50, G=20, UMAX=5, tol < 1e-25

For each γ ∈ {0.5, 1.0, 4.0}, sweep τ:
```
τ = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
```

**Warm-start chain** (from C1 converged points at τ=2):
```
γ=0.5: τ=2 (seed) → 1.5 → 1.0 → 0.8 → 0.5 → 0.3
                   → 3.0 → 4.0 → 5.0 → 7.0 → 10.0 → 15.0

γ=1.0: τ=2 (from C1) → same walk up/down

γ=4.0: τ=2 (from C1) → same walk up/down
```

**36 points total, 33 new runs** (3 at τ=2 already from C1).
~2 NK iterations per step (warm-started).

**Extract:** 1-R² at each (γ, τ).

---

## D. CHEAP — NO-LEARNING (seconds, no REE)

| Fig | Description | What to run |
|-----|-------------|-------------|
| R1  | 1-R² vs K | K=3..20, γ=0.5/1/4, τ=2. Check existing uses correct γ. |
| R2  | Lognormal variant | Same γ=0.5/1/4. Check existing uses correct γ. |
| 11  | Mechanisms bar chart | 6 het-γ/het-τ configs. Already have most data. |

---

## E. PAPER NUMBER UPDATES (from converged mp50+ solutions)

After C1 completes, update main.tex:
```
OLD (float64):        NEW (mp100):
1-R² = 0.108    →    1-R² = 0.077
slope = 0.341   →    slope = 0.541
G = 14           →    G = 20
```

Update: γ-ladder table, posteriors table, G-ladder table, all inline mentions.

---

## F. SKIP FOR SSRN v2

| Fig | Why |
|-----|-----|
| 7   | Trade volume — needs 8 additional γ convergence runs |
| 8   | Value of info V(τ) — needs 45 runs |
| 9   | GS resolution — derived from Fig 8 |

---

## EXECUTION ORDER

```
STEP 1: Extract B (figs 3B, 5, 6B, 10) from existing seed     → 1 hour compute
STEP 2: Run C1 (5 γ points, warm-started)                      → 3-5 hours  
STEP 3: Run D (no-learning, check R1/R2/11 gammas)             → minutes
STEP 4: Update paper numbers from C1 results                   → 30 min editing
STEP 5: Run C2 (33 τ points, warm-started from C1)             → overnight
STEP 6: Build pgfplots coordinates for all figures              → 1 hour
STEP 7: Upload SSRN v2                                         → done
```

Total new convergence runs: 38 (5 from C1 + 33 from C2)
All warm-started from G=20 UMAX=5 seed.
All at mp50, tol < 1e-25.

---

## OUTPUT FORMAT

Save to: `results/full_ree/`

```
fig_NAME_G20_data.json         — raw data
fig_NAME_G20_pgfplots.tex      — pgfplots coordinates
```

pgfplots coordinates: 40-50 points per curve, arc-length resampled.
Linear interpolation in pgfplots (no smooth keyword).
Style 29 colors for contours: blue(low p) → black(p=0.5) → red(high p).
BC20 colors for 1-R² plots: green(γ=0.5), red(γ=1), blue(γ=4).
