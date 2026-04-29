# Full REE contour solve

This directory records reproducible full-rational-expectations-equilibrium
contour solves from `python/full_ree_solver.py`.

## Baseline CRRA solve

Command:

```bash
python3 python/full_ree_solver.py \
  --G 5 --umax 2 --tau 2 --gamma 0.5 \
  --seed no-learning --max-iter 1300 --damping 0.3 --tol 1e-12
```

Result:

| quantity | value |
|---|---:|
| grid | `G=5`, `u in [-2,2]` |
| parameters | `tau=2`, `gamma=0.5` |
| seed | no-learning price function |
| Picard damping | `0.3` |
| iterations | `1029` |
| converged | `true` |
| residual `||Phi(P)-P||_inf` | `9.2604e-13` |
| revelation deficit `1-R^2` | `3.0073e-04` |
| max absolute distance from FR price array | `0.2705` |

At the representative realization `(u1,u2,u3)=(1,-1,1)`:

| variable | value |
|---|---:|
| private prior `mu1` | `0.880797` |
| private prior `mu2` | `0.119203` |
| private prior `mu3` | `0.880797` |
| FR price | `0.880797` |
| CRRA REE posterior `mu1` | `0.916035` |
| CRRA REE posterior `mu2` | `0.887061` |
| CRRA REE posterior `mu3` | `0.916035` |
| CRRA REE price | `0.905507` |
| logit price | `2.259972` |

The converged price tensor is saved in
`G5_tau2_gamma0.5_no-learning_prices.npz` and can be used as a continuation
seed for finer grids.

## G=15 continuation from the G=5 solution

The `G=15` run uses the converged `G=5` non-FR tensor as an intelligent
starting point. The solver trilinearly interpolates
`G5_tau2_gamma0.5_no-learning_prices.npz` onto the `G=15` grid, then applies
the same contour map.

Picard continuation command:

```bash
python3 python/full_ree_solver.py \
  --G 15 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G5_tau2_gamma0.5_no-learning_prices.npz \
  --label from_G5 --max-iter 200 --damping 0.2 --tol 1e-8 \
  --save-array --progress
```

This run did not converge to `1e-8`; after 200 iterations it reached
`||Phi(P)-P||_inf = 2.4385e-03`, with `1-R^2 = 2.5277e-04`.

Anderson continuation from that `G=15` checkpoint:

```bash
python3 python/full_ree_solver.py \
  --G 15 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G15_tau2_gamma0.5_from_G5_prices.npz \
  --label from_G5_anderson --max-iter 100 --damping 0.3 \
  --anderson 5 --anderson-beta 0.7 --tol 1e-8 \
  --save-array --progress
```

This also did not converge to `1e-8`; after 100 Anderson-accelerated
iterations it reached `||Phi(P)-P||_inf = 1.1144e-03`, with
`1-R^2 = 3.6302e-04`. The representative grid point nearest
`(1,-1,1)` is `(1.142857,-1.142857,1.142857)`; at the final checkpoint:

| variable | value |
|---|---:|
| private prior `mu1` | `0.907687` |
| private prior `mu2` | `0.092313` |
| private prior `mu3` | `0.907687` |
| FR price | `0.907687` |
| CRRA checkpoint posterior `mu1` | `0.909888` |
| CRRA checkpoint posterior `mu2` | `0.908853` |
| CRRA checkpoint posterior `mu3` | `0.909888` |
| CRRA checkpoint price | `0.909459` |

The `G=15` arrays are therefore checkpoints, not converged fixed points. They
are useful continuation seeds for a stronger nonlinear solve, but should not
be reported as solved equilibria.

## Branches of the contour map

The same script confirms that the fully revealing price array is also an
exact fixed point:

```bash
python3 python/full_ree_solver.py \
  --G 5 --umax 2 --tau 2 --gamma 0.5 \
  --seed fr --max-iter 50 --damping 0.3
```

This returns residual at machine precision and price/posteriors all equal to
`0.880797` at `(1,-1,1)`.

Thus, at the coarse `G=5` contour discretization, the map has at least two
reproducible fixed points:

1. a fully revealing branch, reached by starting exactly at the FR array; and
2. a non-FR CRRA branch, reached from the no-learning and tilted seeds.

The non-FR branch is the economically relevant one for the paper's CRRA
mechanism, but it remains a coarse-grid result. Higher-resolution production
runs should use the same checks: report the FR seed, no-learning seed,
residual, revelation deficit, representative posteriors, and the max distance
between the solved price array and the FR array.

## G=9 damped Newton attempt

The `G=9` non-FR branch was attempted with damped Newton-Krylov, seeded from
the interpolated converged `G=5` tensor.

Best non-FR checkpoint:

| stage | residual `||Phi(P)-P||_inf` | `1-R^2` | converged |
|---|---:|---:|---|
| Picard-Anderson preconditioner, 300 steps | `2.3567e-04` | `4.9726e-04` | false |
| Damped Newton from preconditioner | `2.1361e-04` | `5.1173e-04` | false |

The Newton step uses matrix-free finite-difference Jacobian-vector products,
GMRES, and backtracking line search.  From the best checkpoint, tighter Krylov
settings (`gmres_max_iter=80`, `fd_eps=1e-7`) did not find a descent step, so
the non-FR `G=9` solve is not converged under the current method.

At `(u1,u2,u3)=(1,-1,1)` for the best non-FR checkpoint:

| variable | value |
|---|---:|
| FR price | `0.880797` |
| checkpoint posterior `mu1` | `0.918962` |
| checkpoint posterior `mu2` | `0.904241` |
| checkpoint posterior `mu3` | `0.918962` |
| checkpoint price | `0.913803` |

As a sanity check, the `G=9` fully revealing branch is an exact fixed point:
starting Newton from the FR tensor gives residual `3.33e-16`.

## G=7 continuation from the G=6 solution

The `G=7` non-FR branch was attempted from the converged `G=6` tensor using
the requested conservative Picard damping:

```bash
python3 python/full_ree_solver.py \
  --G 7 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G6_tau2_gamma0.5_G6_tight_prices.npz \
  --label G7_picard_adaptive --method picard --max-iter 123 \
  --damping 0.1 --adaptive-picard --min-damping 1e-5 \
  --anderson 5 --anderson-beta 0.7 --tol 1e-10 \
  --save-array --progress
```

This checkpoint did not converge. The residual fell from `6.2844e-02` to
`2.3024e-06`, but the adaptive line search then had to reduce damping to very
small values and progress stalled.

Newton continuation from that checkpoint used a maximum Newton damping of
`0.5`:

```bash
python3 python/full_ree_solver.py \
  --G 7 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G7_tau2_gamma0.5_G7_picard_adaptive_prices.npz \
  --label G7_newton_d05 --method newton --max-iter 8 \
  --newton-damping 0.5 --gmres-max-iter 50 --gmres-tol 1e-6 \
  --fd-eps 1e-6 --tol 1e-10 --save-array --progress
```

The Newton step improved the residual only marginally, to
`||Phi(P)-P||_inf = 2.2292e-06`, and then the line search stalled. Thus the
`G=7` non-FR branch is a checkpoint, not a converged fixed point.

At the representative grid point `(1.333333,-0.666667,1.333333)`:

| variable | value |
|---|---:|
| FR price | `0.982014` |
| checkpoint posterior `mu1` | `0.987784` |
| checkpoint posterior `mu2` | `0.983939` |
| checkpoint posterior `mu3` | `0.987784` |
| checkpoint price | `0.986383` |
| checkpoint `1-R^2` | `2.4075e-04` |

As a sanity check, the `G=7` fully revealing branch is exact: starting from the
FR tensor gives residual `3.33e-16`.

## G=6 continuation from the G=5 solution

The `G=6` non-FR branch uses the converged `G=5` non-FR tensor as an
interpolated starting point. Picard-Anderson converged, and a short
continuation pass pushed the fixed-point error below `1e-12`.

Command sequence:

```bash
python3 python/full_ree_solver.py \
  --G 6 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G5_tau2_gamma0.5_no-learning_prices.npz \
  --label G6_picard_pre --method picard --max-iter 500 \
  --damping 0.25 --anderson 5 --anderson-beta 0.7 --tol 1e-10 \
  --save-array --progress

python3 python/full_ree_solver.py \
  --G 6 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G6_tau2_gamma0.5_G6_picard_pre_prices.npz \
  --label G6_tight --method picard --max-iter 100 \
  --damping 0.25 --anderson 5 --anderson-beta 0.7 --tol 1e-12 \
  --save-array --progress
```

Final non-FR result:

| quantity | value |
|---|---:|
| grid | `G=6`, `u in [-2,2]` |
| parameters | `tau=2`, `gamma=0.5` |
| total continuation iterations | `89 + 17` |
| converged | `true` |
| residual `||Phi(P)-P||_inf` | `6.9529e-13` |
| revelation deficit `1-R^2` | `2.2149e-04` |
| `R^2` | `0.9997785142` |
| max absolute distance from FR price array | `0.1692` |

The representative grid point nearest `(1,-1,1)` is `(1.2,-1.2,1.2)`:

| variable | value |
|---|---:|
| private prior `mu1` | `0.916827` |
| private prior `mu2` | `0.083173` |
| private prior `mu3` | `0.916827` |
| FR price | `0.916827` |
| CRRA REE posterior `mu1` | `0.951722` |
| CRRA REE posterior `mu2` | `0.925407` |
| CRRA REE posterior `mu3` | `0.951722` |
| CRRA REE price | `0.941718` |

As a sanity check, the `G=6` fully revealing branch is also an exact fixed
point: starting from the FR tensor returns residual `1.11e-16`.
