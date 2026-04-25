# MainRun1 — Parameter search for heterogeneous-preference CRRA REE

**Goal**: among heterogeneous (τ, γ) configurations, find **strictly converged**
Picard fixed points that **maximise 1 − R²** (the partial-revelation / Jensen-gap
measure), in order to map the PR frontier for the "Noise Traders Are Not a
Primitive" paper.

Snapshot of this run lives at
[`python/MainRun1_results.csv`](./MainRun1_results.csv) (full CSV) and
[`python/MainRun1_progress.log`](./MainRun1_progress.log) (stdout log). This
run is **in progress at the time of snapshot**; the CSV is re-copied when the
search completes.

## Fixed numerical setup

| Parameter | Value |
|---|---|
| Grid size G | 9 (729 cells per price tensor) |
| Grid range | u ∈ [−2, 2], equally spaced |
| Convergence criterion | ‖Φ(P) − P‖∞ < **10⁻⁸** AND ‖F(P)‖∞ < 1.0 |
| Picard damping ladder | α ∈ {1.0, 0.3, 0.1}, max-iter {1000, 2500, 4000} |
| Init ladder | cold → warm (cached nearest) → IDW (k=5 nearest, 1/d² weighted). **Never CARA-FR** (it's a formal Φ-fixed-point that traps Picard in the FR basin) |
| Target cell for posteriors | (u₁,u₂,u₃) = (1, −1, 1) |
| Budget | 3 h wall-clock on 1 CPU core (numba-accelerated) |

## Parameter search space

- **coarse grid** γ ∈ {0.3, 1.0, 3.0, 10, 50}, τ ∈ {0.3, 1.0, 3.0, 10} — homogeneous baselines then all heterogeneous combinations (combinations-with-replacement in γ × all τ; then homogeneous γ × combinations-with-replacement in τ)
- **random log-uniform** 200 samples with τ ∈ [0.1, 10], γ ∈ [0.1, 50]
- **extreme-misaligned** 100 samples with one agent at (low-τ, low-γ) and two at (high-τ, high-γ)
- **extreme-spread** 200 samples with at least 10× range in both γ and τ
- After dedup: **693 unique candidates**

γ = 0.1 was dropped: at G=9 it fails cold-start across the full damping ladder
(~35 s per config, never reaches 10⁻⁸ on ‖Φ-I‖∞).

## Two 1−R² flavours

For each converged price tensor:

| Column | Reference predictor | Meaning |
|---|---|---|
| `oneR2_eq`  | T = Σₖ τₖ·uₖ | CARA-FR reference when all γ equal |
| `oneR2_het` | T = (Σₖ (τₖ/γₖ)·uₖ) / Σₖ (1/γₖ) | CARA-FR reference with heterogeneous γ (weighted by 1/γₖ) |

1−R² is the (unweighted over grid cells) fraction of logit(p) variance NOT
explained by this linear combo. A post-processor
([`analyze_weighted_R2.py`](./analyze_weighted_R2.py)) will also compute a
**prior-weighted 1−R²** under the ex-ante Gaussian signal density and
per-agent **value functions + welfare decomposition** once MainRun1 completes.

## Snapshot: top 5 converged configs at t ≈ 93 min

Signal u_report = (1, −1, 1), γ=CRRA, τ=signal precision.

| rank | 1−R²_het |  τ             |  γ                  |  α  | iters | ‖Φ-I‖∞ | ‖F‖∞ | p\*(1,-1,1) | init |
|---:|-----:|---|---|---:|---:|---:|---:|---:|---|
| 1 | **0.6593** | (3, 3, 3) | (0.3, 50, 50)   | 0.3 | 2500 | 9.9e-9 | 1.4e-3 | 0.9980 | cold |
| 2 | 0.6324 | (3, 3, 3) | (0.3, 10, 10)   | 1.0 |  830 | 9.6e-9 | 1.3e-3 | 0.9957 | cold |
| 3 | 0.4581 | (3, 3, 3) | (0.3, 1, 10)    | 1.0 |  645 | 9.9e-9 | 1.3e-4 | 0.9250 | cold |
| 4 | 0.2627 | (3, 3, 3) | (0.3, 0.3, 3)   | 1.0 |  801 | 10e-9  | 2.0e-4 | 0.9518 | warm |
| 5 | 3.15e-3 | (3, 3, 3) | (50, 50, 50)    | 1.0 |  489 | 8.9e-9 | 1.7e-6 | 0.9763 | cold |

Full list of 14 converged-so-far configs is in the CSV (columns documented below).

## Key patterns (preliminary)

1. **Pure Jensen gap** at fixed τ=3 is small (1−R² ≈ 0.3–0.5 × 10⁻²) and nearly
   **γ-independent** across homogeneous γ ∈ {0.3, 1, 3, 10, 50}.
2. **Heterogeneous γ is the main PR amplifier** — the best configs so far all
   sit at τ=(3, 3, 3) with γ spread ranging from 10× (0.3/3) up to 167× (0.3/50).
   1−R² jumps by 2 orders of magnitude when you go from homogeneous γ to γ
   with a single low-γ agent among high-γ neighbours.
3. **Widest γ ratio wins** so far: γ=(0.3, 50, 50) → 1−R² = 0.66. One aggressive
   agent trading against two effectively-CARA partners — PR largely
   captured by the curvature needed to clear the nonlinear aggressive demand.
4. **Low τ (≤1) + high γ** → 1−R² → 0 (near-FR regime, prices uninformative
   anyway).

## Where welfare comes from (Milgrom-Stokey context)

Milgrom-Stokey no-trade requires common priors **plus no trade motive beyond
information**. Our heterogeneous-γ setup breaks the second:

- **Risk-sharing channel**: even under CARA/FR with the same μ for every agent,
  heterogeneous γ means agents want different exposure → they trade.
- **Belief-heterogeneity channel under PR**: μ₁ ≠ μ₂ ≠ μ₃ at each cell means
  agents disagree after observing the price → additional trading.
- **Value of own signal** ΔV_k = V_k^full − V_k^pub: under CARA ΔV = 0
  (price is a sufficient statistic); under CRRA/PR ΔV > 0.

Total welfare = risk-sharing surplus + PR-driven belief surplus + (value of
private info). Welfare numbers will be added once MainRun1 completes.

## CSV column reference

| column | meaning |
|---|---|
| `tau_1, tau_2, tau_3` | signal precisions |
| `gamma_1, gamma_2, gamma_3` | CRRA coefficients |
| `alpha` | Picard damping that produced the reported iterate |
| `iters` | Picard iterations |
| `time_s` | wall time for this config |
| `PhiI` | ‖Φ(P) − P‖∞ at reported iterate |
| `Finf` | ‖F(P)‖∞ (market-clear residual) |
| `oneR2_het` | 1 − R² with heterogeneous-γ CARA-FR reference |
| `oneR2_eq`  | 1 − R² with equal-weight reference |
| `p_star` | price at cell (1,−1,1) nearest grid point |
| `mu_1, mu_2, mu_3` | posteriors at (1,−1,1) |
| `pr_gap` | μ₁ − μ₂ (cell-level disagreement) |
| `converged` | 1 if both Phi-I < 1e-8 and F < 1.0, else 0 |
| `init` | which initialiser produced the accepted iterate: cold \| warm \| idw |

## Reproducing MainRun1

```bash
cd /home/user/REZN/python
python3 -u search_het.py --G 9 --budget-hours 3.0 --seed 42 \
    --abstol 1e-8 --f-tol 1.0 \
    --out MainRun1_results.csv
```
