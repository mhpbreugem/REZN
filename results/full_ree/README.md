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
