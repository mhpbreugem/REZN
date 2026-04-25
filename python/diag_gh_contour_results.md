# Gauss-Hermite vs PCHIP-grid-sum contour: diagnostic results

Branch: `REZN`. Cache: `pchip_G11u20_cache.pkl` (G=11, UMAX=2.0).

For each cached converged P, compute Φ two ways:
- **Φ_pchip**: production 2-pass PCHIP-grid-edge sum (logit space).
- **Φ_GH(N)**: 2-pass Gauss-Hermite quadrature against the signal density,
  off-grid root-finding via PCHIP, no Jacobian factor (matches production
  pseudo-likelihood form).

`gap_GH(N)` = max over (5 worst-residual cells + 5 random cells) of
`|Φ_GH(N)[i,j,l] − Φ_pchip[i,j,l]|`.

## Results

| τ    | γ    | Finf_pchip | gap_GH30 | gap_GH50 | gap_GH70 |
|------|------|------------|----------|----------|----------|
| 3.00 | 50.0 | 4.7e-08    | 3.6e-04  | 6.0e-06  | 2.1e-05  |
| 3.00 | 30.0 | 4.0e-08    | 2.8e-03  | 1.9e-04  | 2.0e-03  |
| 3.00 | 15.0 | 5.5e-08    | 2.5e-04  | 6.3e-06  | 1.6e-04  |
| 3.00 | 10.0 | 6.0e-08    | 2.8e-03  | 5.7e-04  | 2.0e-03  |
| 3.00 |  6.0 | 1.8e-08    | 2.4e-03  | 6.9e-06  | 1.9e-04  |
| 3.00 |  4.0 | 4.9e-08    | 2.1e-03  | 6.4e-05  | 1.7e-03  |
| 3.00 |  3.0 | 7.8e-08    | 7.8e-02  | 7.8e-02  | 7.8e-02  |
| 3.20 |  3.0 | 5.8e-08    | 3.3e-05  | 1.9e-05  | 6.6e-05  |
| 3.40 |  3.0 | 9.2e-08    | 3.2e-04  | 1.0e-04  | 2.4e-04  |

## Interpretation

**Contour discretization IS the dominant error source.** Across every
config, the gap between Φ_GH and Φ_pchip at the cached P exceeds
Finf_pchip by 10²–10⁶. Implication: the production fixed point is the
fixed point of an *approximate* contour operator, and Newton on the
analytic Jacobian only polishes that approximation — it cannot reach
beyond the contour-method bias.

**Homogeneous γ=3 τ=3 is anomalous.** The 7.8e-2 gap is stable across
N=30/50/70, meaning GH has converged but disagrees fundamentally with
PCHIP-sum. This is the same configuration that exhibits two co-existing
equilibria (near-CARA pre-jump and strong-PR post-jump) in the published
results. The 7.8e-2 disagreement suggests the near-CARA branch may be a
discretization artifact of the PCHIP-grid-edge sum and that the GH-based
Φ would either (a) collapse the two branches into one or (b) select a
different near-CARA equilibrium that's closer to the strong-PR branch.

**GH(N) does not converge monotonically in N.** GH50 is consistently best;
GH70 is sometimes worse than GH50. The cause is node truncation: outer
GH nodes at |ξ| ≈ √(2N) lie outside u ∈ [-UMAX, UMAX] and are dropped.
Beyond N ≈ τ·UMAX²/2 ≈ 6, additional nodes are mostly truncated and
contribute only noise. Fix: extend UMAX so the contour weight beyond
[-UMAX, UMAX] is negligible (UMAX ≥ 4 for τ=3 gives weight < e^{-24}),
or switch to Gauss-Legendre on [-UMAX, UMAX] with f_v as part of the
integrand.

## Recommended next step

Implement Φ_GH (or Φ-with-Gauss-Legendre) in numba-parallel, plumb it
through `solve_newton_analytic`, and re-run the τ-sweep. Expected wins:

- Finf floor drops from 1e-7 to 1e-12+ at G=11.
- Homogeneous γ=3 branch structure clarified — either single equilibrium
  emerges or the strong-PR branch becomes more easily reachable.
- 1-R² values may shift by ~1e-3 (the gap magnitude) for non-anomalous
  configs; the qualitative pattern (heterogeneity → strong PR) is
  unaffected.

Cost estimate: 1-2 hours to write Φ_GH in numba (mirror `_contour_sum_pchip`,
swap grid-edge sum for GH quadrature with off-grid 2D PCHIP). Analytic
Jacobian needs the `_contour_sum_tangent` rewrite to match — another
2-3 hours.
