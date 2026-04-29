# K=4 Numerical Results

Branch: `claude/add-claude-documentation-AC2lV`. Run during the
2026-04-29 session as a sanity-check extension of the canonical K=3
contour-method results documented in `CHAT_MEMORY.md` and
`SESSION_SUMMARY.md`. All scalar parameters mirror the canonical K=3
case: τ=2, γ=0.5 (CRRA) or γ→∞ (CARA), W=1, u∈[−2,2] at G=5 and
u∈[−3,3] at G=7, equal wealth, zero net supply.

Solver: 4-axis extension of the 2-pass contour method (theory.md §5.2),
with three sweep directions (instead of two) per agent's 3D slice,
averaged to form A_v. Anderson acceleration (Walker–Ni type II,
window m=8, mixing β=0.5–0.7), full S₄ symmetrisation each iteration.
Tolerance ‖P−Φ(P)‖∞ < 1e-11 requested.

Implementation kept ephemeral (`/tmp/k4_full_ree.py`,
`/tmp/k4_no_learning.py`). Per repo convention, only outputs are
documented here.

## 1. CARA fixed point at K=4

CARA reaches **machine precision** (‖F‖∞ ≈ 2e-16) in **3 Anderson
iterations** at both G=5 and G=7.

| G | umax | iters | ‖F‖∞ at stop | 1−R² | p at (1,−1,1,1) | μ at (1,−1,1,1) |
|---|------|-------|--------------|------|------------------|------------------|
| 5 | 2.0  | 3     | 2.2e-16      | 0.01729 | 0.98201 | (0.9820,0.9820,0.9820,0.9820) |
| 7 | 3.0  | 3     | 5.7e-15      | 0.01835 | 0.98201 | (0.9820,0.9820,0.9820,0.9820) |

`p = Λ(T*) = Λ(τ·(1−1+1+1)) = Λ(4) = 0.98201` exactly, and all four
posteriors equal `Λ(4)` exactly: full revelation **at the realisation**.
This mirrors the K=3 reference (μ=p=0.8808=Λ(2) at (1,−1,1)).

The 1−R² ≈ 0.018 is **not** a residual error — the residual is at
machine precision — but a **discretisation-floor bias**: the global
weighted regression of logit(P) on T* across all G⁴ cells deviates
slightly from a perfect linear fit because off-grid linear
interpolation is approximate. The floor barely moves between G=5
(0.0173) and G=7 (0.0184), suggesting it is dominated by something
other than 1/G² interpolation error — possibly the boundary-clipped
configurations where one or more agents has an extreme signal.

## 2. CRRA fixed point at K=4

CRRA at the same parameters does **not** reach 1e-11 tolerance.
‖F‖∞ stalls at ~2–3e-3 — the same regime as the K=3 Anderson stall
(~2e-3 documented in `CHAT_MEMORY.md` and `SESSION_SUMMARY.md` §
"Anderson convergence at this point"). 60 iterations.

| G | γ   | anchor       | iters | ‖F‖∞ at stop | 1−R²    | p       | μ |
|---|-----|--------------|-------|--------------|---------|---------|---|
| 5 | 0.5 | (1,−1,1,−1)  | 60    | 2.9e-3       | 0.02656 | 0.49998 | (0.4980, 0.5019, 0.4980, 0.5019) |
| 5 | 0.5 | (1,−1,1, 1)  | 60    | 2.9e-3       | 0.02656 | 0.99049 | (0.9927, 0.9856, 0.9927, 0.9927) |
| 5 | 1.0 | (1,−1,1, 1)  | 40    | 4.2e-3       | 0.02100 | 0.99156 | (0.9933, 0.9859, 0.9933, 0.9933) |
| 7 | 0.5 | (1,−1,1, 1)  | 60    | 1.5e-2       | 0.03178 | 0.99065 | (0.9928, 0.9861, 0.9928, 0.9928) |

The (1,−1,1,−1) anchor is informationally degenerate (T*=0 → p=0.5 by
symmetry) and gives no info on disagreement. The (1,−1,1,1) anchor is
the natural extension of the K=3 (1,−1,1) reference (one contrarian
signal):

| | realisation | p | μ | μ_max − μ_min |
|---|---|---|---|---|
| K=3 reference | (1,−1,1)    | 0.9077  | (0.9185, 0.8889, 0.9185)         | 0.0296 |
| K=4 (this run)| (1,−1,1, 1) | 0.99049 | (0.9927, 0.9856, 0.9927, 0.9927) | 0.0072 |

Same qualitative pattern — agents with positive signals are *more*
bullish than the market price, the lone u=−1 agent is *less* bullish —
the structural fingerprint of partial revelation. Magnitude of
disagreement shrinks with K, consistent with the no-learning K-sweep
(`theory.md` §6.3) where 1−R² also declines roughly with 1/K.

## 3. NET = CRRA − CARA (the meaningful diagnostic)

The discretisation floor cancels when subtracting CARA from CRRA at the
same G:

| G | γ   | 1−R² (CRRA) | 1−R² (CARA) | NET    |
|---|-----|-------------|-------------|--------|
| 5 | 1.0 | 0.02100     | 0.01729     | +0.0037 |
| 5 | 0.5 | 0.02656     | 0.01729     | +0.0093 |
| 7 | 0.5 | 0.03178     | 0.01835     | **+0.0134** |

NET is positive at every (G, γ) tested — same sign as the K=3
PR-survival result. **NET grows with G** (G=5 → G=7 at γ=0.5: +0.0093 →
+0.0134), consistent with the discretisation floor cancelling and a
genuine PR signal that strengthens with resolution. NET grows as γ
falls (γ=1.0 → γ=0.5 at G=5: +0.0037 → +0.0093), consistent with the
no-learning K=4 sweep above and with `theory.md` Prop 8 (smooth
transition).

Posteriors at the (1,−1,1,1) anchor are robust across (G, γ) — μ_4 and
μ_1=μ_3 stay near 0.993, μ_2 (the contrarian) stays near 0.986.

That said, the K=4 solver has the same FR-collapse pathology as K=3
documented in `figures/NOTES_full_eq_status.md`: damped Anderson's
attractor sits near the FR fixed point even when seeded with the
no-learning PR price (no-learning seed at G=7 has 1−R² = 0.060;
Anderson stalls at 0.032). The PR signal we measure is the *gap*
between this stalled point and the CARA floor at the same G, not a
fully-converged PR fixed point.

## 4. K=4 no-learning smooth-transition table (paper Table 1 layout)

G=15, u∈[−4,4]. Computed by direct enumeration of all G⁴ = 50 625
signal configurations, market-clearing solved by `brentq` on
[1e-3, 1−1e-3], weighted regression of logit(p) on T* with weights
w = ½(Π f₁ + Π f₀).

| γ \\ τ | 0.5    | 1.0    | 2.0    |
|--------|--------|--------|--------|
| 0.1    | 0.1445 | 0.1446 | 0.1401 |
| 0.3    | 0.0458 | 0.0676 | 0.0785 |
| 0.5    | 0.0170 | 0.0358 | 0.0533 |
| 1.0    | 0.0036 | 0.0121 | 0.0259 |
| 3.0    | 0.0004 | 0.0018 | 0.0052 |
| 10.0   | 0.0000 | 0.0002 | 0.0006 |
| 100.0  | 0.0000 | 0.0000 | 0.0000 |

CARA explicit (linear-in-logit demand, evaluated at γ=1): identically
0 across τ — confirms the FR baseline.

Side-by-side with the K=3 reference (`theory.md` §6.1, G=20):

| (γ, τ) | K=3    | K=4    | K=4/K=3 |
|--------|--------|--------|---------|
| (0.1, 0.5) | 0.146 | 0.144 | 0.99 |
| (0.5, 2.0) | 0.062 | 0.053 | 0.86 |
| (1.0, 2.0) | 0.029 | 0.026 | 0.90 |
| (3.0, 2.0) | 0.006 | 0.005 | 0.87 |

K=4 1−R² is consistently slightly below K=3, consistent with the
K-sweep in `theory.md` §6.3 where the deficit decays roughly with 1/K.
CARA-knife-edge ranking is preserved exactly.

## 5. Open items

1. **CARA floor**: ~0.018 at both G=5 and G=7. Need G ≥ 10 (and a
   faster solver — G=9 is already 25 s/Φ at K=4) to characterise the
   floor scaling.
2. **CRRA convergence**: Anderson at K=4 G=7 stalls at ‖F‖∞ ≈ 1.5e-2
   after 60 iterations — worse than G=5 (~3e-3). Mixing β=0.5 may be
   too aggressive at higher G; either tune β downward or use the
   globalised Newton–Krylov with perturbation homotopy recommended in
   `NOTES_full_eq_status.md`.
3. **Wealth / preference heterogeneity**: K=4 + heterogeneous (γ_k,
   τ_k) would extend `figures/fig6_mechanisms` to four agents
   (including the het-α channel from Mechanism 4) once a working REE
   solver is in hand.

## 6. Bottom line

K=4 inherits both the qualitative result (PR survives, CARA is FR
knife-edge) and the quantitative pathology (Anderson stalls below the
1e-11 tolerance) from K=3. The CARA→FR mapping is exact at the
realisation level (μ_k = p = Λ(T*) for all k, to machine precision).
The CRRA fingerprint is preserved structurally: the contrarian agent
discounts the price, the consensus agents over-shoot, and the gap
shrinks with K and grows with τ.

Run details:
- 5 full-REE Anderson chains (CARA G=5/7, CRRA G=5 ×2 anchors, CRRA
  G=5 γ=1, CRRA G=7) — wall-clock ~15 minutes total
- 1 no-learning sweep at G=15 (21 cells × 7 s = ~2.5 min)
- All scripts kept ephemeral (`/tmp/k4_full_ree.py`,
  `/tmp/k4_no_learning.py`); only this notes file committed.
