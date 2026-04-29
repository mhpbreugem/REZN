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

Direct Newton from the interpolated `G=6` tensor was also attempted, without
the adaptive Picard preconditioner:

```bash
python3 python/full_ree_solver.py \
  --G 7 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G6_tau2_gamma0.5_G6_tight_prices.npz \
  --label G7_direct_newton_from_G6 --method newton --max-iter 12 \
  --newton-damping 0.5 --gmres-max-iter 60 --gmres-tol 1e-6 \
  --fd-eps 1e-6 --tol 1e-10 --save-array --progress
```

This direct Newton path performed worse: after 12 Newton steps the residual was
`7.8038e-03`, with `1-R^2 = 1.6106e-02`. It moved away from the low-deficit
checkpoint rather than converging.

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

### Extrapolated seed from `G=5` and `G=6`

We also tried a more aggressive continuation seed that combines the converged
`G=5` and `G=6` tensors. Both tensors were interpolated to the `G=7` grid and
linearly extrapolated in logit-price space according to grid spacing:

```text
logit P_7^seed = logit P_6 + ((h_7-h_6)/(h_6-h_5)) (logit P_6 - logit P_5),
where h_5=1.0, h_6=0.8, h_7=2/3.
```

This seed was not an improvement. Its initial residual was `7.4792e-02`, and
after 120 adaptive Picard iterations it remained at
`||Phi(P)-P||_inf = 1.6638e-03`, with `1-R^2 = 1.6248e-04`. The previous
`G=6`-only interpolation followed by adaptive Picard/Newton remains the best
`G=7` non-FR checkpoint (`2.2292e-06`).

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
| residual `||Phi(P)-P||_inf` | `6.5850e-15` |
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

## G=6 random-seed basin search

The `G=6` branch structure was also tested with 100 random seeds.  The search
uses random perturbations around the FR branch, the non-FR branch, mixtures of
FR/non-FR, and mixtures involving the no-learning price tensor.  Stage 1 runs
each seed with Picard-Anderson for 90 iterations to identify the nearest basin;
Stage 2 tightens the distinct basin representatives to the requested
`1e-14` tolerance.

Command:

```bash
python3 python/g6_random_seed_search.py \
  --seeds 100 --tol 1e-14 --stage1-tol 1e-7 \
  --stage1-iter 90 --refine-iter 260 --workers 4 \
  --outdir results/full_ree/g6_random_search
```

Result:

| equilibrium | stage-1 nearest seeds | final residual | `1-R^2` |
|---|---:|---:|---:|
| FR | 50 | `2.22e-16` | `0` |
| non-FR | 50 | `6.5850e-15` | `2.214858e-04` |

No third equilibrium was found among the 100 random seeds under the
`1e-14` refinement criterion.  The full per-seed outcomes are stored in
`g6_random_search/seed_results.csv`.

## G=6 random-seed search with slightly heterogeneous precision

The heterogeneous-precision experiment perturbs the common precision by
`0.0001` across agents:

```text
tau = (1.9999, 2.0000, 2.0001).
```

Because heterogeneous precision breaks exchangeability, the solver for this
experiment does not symmetrize the price tensor.  Contours are traced in
logit-price space; otherwise the discretized contour method creates a small
artificial residual even at the analytically fully revealing price tensor.

Command:

```bash
python3 python/g6_hetero_tau_search.py \
  --seeds 100 --tol 1e-14 --stage1-iter 60 --refine-iter 220 \
  --workers 4 --outdir results/full_ree/g6_hetero_tau_search
```

Result:

| candidate | stage-1 nearest seeds | final residual | `1-R^2` | strict convergence |
|---|---:|---:|---:|---|
| FR | 45 | `2.22e-16` | `1.11e-16` | true |
| non-FR candidate | 55 | `7.1666e-07` | `1.05e-09` | false |

Thus, for `tau=(1.9999,2.0,2.0001)`, the 100-seed search found only the fully
revealing equilibrium below the `1e-14` tolerance.  The non-FR candidate is
pulled close to the FR tensor and does not satisfy the strict residual
criterion under the current hetero-tau contour map.  Per-seed outcomes are
stored in `g6_hetero_tau_search/seed_results.csv`.

A second run forces all 100 random seeds to be extremely close to the
homogeneous `tau=2` PR solution.  The seeds are logit-space perturbations of
the homogeneous PR tensor with amplitudes between `1e-14` and `1e-8`:

```bash
python3 python/g6_hetero_tau_search.py \
  --seeds 100 --seed-mode near_nonfr \
  --amp-min-log10 -14 --amp-max-log10 -8 \
  --tol 1e-14 --stage1-iter 120 --refine-iter 500 \
  --workers 4 --outdir results/full_ree/g6_hetero_tau_close_pr_search
```

All 100 seeds remained nearest to the non-FR candidate in Stage 1, but the
candidate still did not converge below `1e-14` after refinement:

| candidate | stage-1 nearest seeds | final residual | `1-R^2` | strict convergence |
|---|---:|---:|---:|---|
| FR | 0 | `2.22e-16` | `1.11e-16` | true |
| non-FR candidate | 100 | `2.9030e-07` | `1.46e-10` | false |

The close-seed run therefore supports the same conclusion: even when initialized
arbitrarily close to the homogeneous PR equilibrium, the slightly heterogeneous
`tau` case does not yield a distinct non-FR fixed point below `1e-14` under the
current contour solver.  Per-seed outcomes are stored in
`g6_hetero_tau_close_pr_search/seed_results.csv`.

## Symmetric K-agent exploratory runs at G=5

The script `python/generic_k_symmetric_solver.py` generalizes the contour
method to symmetric `K=4` and `K=5` cases at `G=5`.  It keeps the full
`G^K` price tensor, but exploits symmetry by averaging each map evaluation over
all agent permutations.  For an agent's `(K-1)`-dimensional slice, it sweeps
`K-2` coordinates and root-finds the remaining coordinate, averaging over all
choices of root axis.

These runs are exploratory: the cost rises quickly with `K`, especially for
`K=5`, and the current Picard-Anderson iteration did not converge to a non-FR
fixed point.

| run | seed | iterations | residual `||Phi(P)-P||_inf` | status |
|---|---|---:|---:|---|
| `K=4,G=5` | no-learning | 220 | `1.3792e-03` | not converged |
| `K=4,G=5` | from converged `K=3` PR tensor | 220 | `1.0210e-02` | not converged |
| `K=4,G=5` | 20 slight perturbations of best checkpoint | 80 each | best `1.5666e-03` | not converged |
| `K=4,G=5` | FR tensor | 1 | `6.66e-16` | converged FR sanity check |
| `K=5,G=5` | from best `K=4` checkpoint | 8 | `3.5232e-02` | not converged |
| `K=5,G=5` | FR tensor | 1 | `7.94e-09` | FR sanity check; contour floor |

The `K=4` no-learning run reaches a non-FR-like checkpoint at
`(1,-1,1,1)` with price `0.990187` and posteriors
`(0.992472,0.985309,0.992472,0.992472)`, versus FR price `0.982014`.
Twenty logit-space perturbation restarts around that checkpoint, with
amplitudes between `1e-8` and `1e-4`, did not improve it.  The best perturbed
run had residual `1.5666e-03`, so the original no-learning checkpoint remains
the best `K=4` non-FR attempt.
The `K=5` bounded checkpoint at `(1,-1,1,1,1)` has price `0.998534` and
posteriors around `(0.999087,0.998360,0.999087,0.999087,0.999087)`, versus
FR price `0.997527`.

The generic `K` solver confirms exact/near-exact FR behavior, but the non-FR
branches for `K=4` and `K=5` require a stronger nonlinear solver or more
runtime than the bounded Picard-Anderson attempts here.

## K=3, G=25 continuation from the G=6 non-FR solution

The `G=25` run uses the converged `G=6` non-FR tensor
(`G6_tau2_gamma0.5_G6_floor_check_prices.npz`) as an interpolated starting
point.  A single map evaluation at `G=25` costs roughly 33 seconds.

Probe command:

```bash
python3 python/full_ree_solver.py \
  --G 25 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G6_tau2_gamma0.5_G6_floor_check_prices.npz \
  --label G25_probe_from_G6 --method picard --max-iter 1 \
  --damping 0.15 --anderson 0 --tol 1e-12 --save-array --progress
```

The interpolated `G=6` seed has residual `5.5664e-02` before the update.  The
saved one-step checkpoint has residual `4.8056e-02`.

Bounded continuation command:

```bash
python3 python/full_ree_solver.py \
  --G 25 --umax 2 --tau 2 --gamma 0.5 \
  --seed array \
  --seed-array results/full_ree/G25_tau2_gamma0.5_G25_probe_from_G6_prices.npz \
  --label G25_from_G6_picard --method picard --max-iter 24 \
  --damping 0.12 --anderson 5 --anderson-beta 0.7 \
  --tol 1e-12 --save-array --progress
```

Result after 24 Picard-Anderson iterations:

| quantity | value |
|---|---:|
| residual `||Phi(P)-P||_inf` | `1.6990e-03` |
| `1-R^2` | `6.7486e-05` |
| max absolute distance from FR price array | `0.1610` |
| converged to `1e-12` | false |

At `(1,-1,1)` the checkpoint price is `0.889993`; the posteriors are
`(0.891037,0.889249,0.891037)`, versus FR price `0.880797`.

A short Newton-Krylov continuation was attempted from this checkpoint.  The
first Newton step cost about 540 seconds and improved the residual only from
`1.6990e-03` to `1.5615e-03`; the second step was stopped because the marginal
improvement was too small for the cost.  The `G=25` non-FR branch is therefore
not solved to `1e-12` under the current implementation.
