# Status of "full-equilibrium" figures (Figs 4, 6, 7, 8, 9, 10)

## TL;DR
Six figures are flagged with a yellow background as placeholders. The
production PCHIP+contour kernel cannot reach the strong-PR branch of
the REE from any cold or perturbed seed I have tried tonight. All
solver paths (Picard α=0.3, Anderson m=8, scipy newton_krylov rdiff=1e-8)
collapse onto the near-CARA / FR fixed point, contradicting the
HANDOFF's reported strong-PR values (1-R² ≈ 0.057 at γ=3, τ=3).

## Evidence collected tonight
At γ=0.5, τ=2, G=11:
- No-learning P seed: 1-R²=0.113, PR-gap=0.092 (NL is *not* a fixed
  point — Finf=0.325)
- After Picard α=0.3 maxiters=2000: 1-R²=4e-6, PR-gap=0.0002
  (collapsed to FR)
- After Anderson m=8 + Picard chain: same FR result
- After analytic-Newton 15 iters: 1-R²=0.10, PR-gap=0.083, but
  Finf=0.086 (not converged, transient on PR manifold)

At γ=3, τ=3, G=11 (the HANDOFF's documented PR regime):
- Picard α=0.3: 1-R²=1.5e-7, PR-gap=0 (FR)
- NK from no-learning: 1-R²=nan due to clip, p=0.769 (near-FR)
- NK from NL+0.1·rand: non-converged spurious (Finf=9e-3, p=0.349)
- NK from NL+0.2·rand: non-converged (Finf=4e-2, p=0.794)
- NK from NL + sign-tilt: 1-R²=nan, p=0.767 (FR)
- *None* reached the HANDOFF target (1-R²=0.057, p=0.632,
  μ=(0.645, 0.633, 0.645))

## What's needed to actually reach PR
Per HANDOFF.md and `pchip_G11_backward_snap.csv`, the strong-PR branch
was found by:
1. γ-homotopy at fixed τ=3 from γ=50 (cold) down through γ=3,
   continuation step by step.
2. At τ ≈ 3.39 the basin boundary shifts during the τ-sweep — some
   warm starts drift into PR.
3. Backward τ-sweep from the post-jump PR solution preserves the
   branch down to τ ≈ 2.87.

Reproducing this requires running `pchip_continuation.py` end-to-end
(hours of compute) plus the specific homotopy ladder it implements.
That's the right next step but is well outside the figure-driver
scope I have for tonight.

## Effect on each placeholder

| Fig | Spec wants | Currently shows | Gap |
|-----|-----------|----------------|-----|
| 4 (posteriors) | full REE at γ=0.5, G=20 | full REE landing on FR branch | wrong fixed point |
| 6 (mechanisms) | full REE per config | no-learning value | needs PR REE |
| 7 (volume)     | full REE             | no-learning E[\|x\|]      | needs PR REE |
| 8 (V of info)  | full REE            | no-learning V(τ)         | needs PR REE |
| 9 (GS)         | derived from Fig 8  | derived from no-learning | inherits Fig 8 caveat |
| 10 (K agents)  | full REE per K       | no-learning              | needs PR REE |

## Files added during the hunt
- `python/probe_PR_seeds.py` — analytic-Newton from 6 seed strategies
- `python/probe_picard_G11.py` — Picard test at G=11
- `python/probe_picard_g3.py` — Picard sweep around γ=3
- `python/probe_nk_pr.py` — scipy newton_krylov from 4 seeds
- `python/ree_solver.py` — common Picard+Anderson helper

All output in the corresponding `.log` files (gitignored where
possible).

## Recommended next step
Either:
- run `pchip_continuation.py main()` end-to-end, save cache.pkl,
  then have the figure drivers warm-start from the cached PR P_stars
  (~3-4h compute, mostly unattended); or
- limit "full-eq" figures to the (γ=3, τ ≈ 3) region where PR is
  cheaply reachable, accept narrower parameter ranges in those plots.

The yellow background should stay on these figures until either
path lands.
