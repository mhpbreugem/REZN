# mp100 / 1e-50 status

## Current state

The branch contains:

1. **`code/mp_solver.py`** — pure-Python mpmath K=3 contour-evidence
   kernel, gmpy2 backend. Validated against the float64 numba kernel at
   G_inner=4 dps=50 (matches to ~1e-15 per cell on a single phi step).
   Public functions:
   - `set_dps(dps)` — set mpmath precision
   - `init_no_learning_K3_mp(...)` — build no-learning halo at the
     current dps
   - `phi_K3_mp(...)` — full Bayes update + market clearing for the
     inner cube
   - `clear_crra_mp` — uses `mp.findroot` (anderson) instead of
     bisection for fast (and tight) market clearing
   - `residual_inf` — ‖φ(P)−P‖_∞ over the inner cube
   - `f64_array_to_mp`, `mp_array_to_strings`, `strings_to_mp` — I/O

2. **`code/task3_g400_mp100.py`** — driver that loads existing float64
   checkpoints, builds an mp100 halo, injects the warm-started inner
   cube, and runs Picard at mp100 until ‖F‖<1e-50 or `max_iter`.

3. **`results/full_ree/task3_g400_t*_mp50.json`** — the existing
   per-tau checkpoints. After this WIP commit:
   - tau in {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}: G_inner=20
     (under-converged float64, F_inner ~0.03–0.11)
   - tau in {7.0, 10.0, 15.0}: G_inner=14 from the previous commit

## Why the full mp100 sharpening didn't complete in this session

A trial run of `task3_g400_mp100 --gamma 4.0 --only-tau 2.0 --dps 100`
shows the no-learning halo build at G_full=28 (= G_inner=20 + 2*pad)
takes ~3+ minutes (it was still running when I killed it). The phi step
on the inner cube is significantly slower than the halo because each of
the 20³=8000 inner cells does 3 agent_evidence sweeps over 28×28 plane
slices, plus a market clear via `mp.findroot`.

Per-tau cost estimate at G_inner=20 dps=100 with the gmpy2 backend:
- halo build (one-time per tau): ~3–8 min
- phi step (per Picard iter): ~5–15 min
- iters needed from a float64 warm start with F~1e-2: ~5
  (Picard near the K=3 fixed point is approximately quadratic per the
  seed file `posterior_v3_G20_umax5_notrim_mp300.json` history, where
  F_max went 0.4 → 0.005 → 9e-6 → 5e-11 → 2e-21 → 2e-42 → 1e-81 → 7e-119
  in 7 iters; each iter ~10 min at G=20 mp300)
- per tau wall time: ~30–80 min
- 12 tau total: **6–16 hours**

## What's needed to actually finish at the requested spec

Pick one:

A) **Run the existing driver overnight.** It already saves per-tau
   checkpoints incrementally. Submit
   ```
   python -m code.task3_g400_mp100 --gamma 4.0 --dps 100 --tol 1e-50 --max-iter 8
   ```
   and let it run for ~10 hours. Each tau's mp100 string array is
   ~G_inner^3 * dps bytes ≈ 800 KB, so storage is fine.

B) **Reduce G_inner to 14** (the value used in the previous commit's
   results). Same number of cells as the original G=14 work, but with
   the existing float64 result already at F~5e-3 ready to be
   sharpened. Per-tau cost drops to ~10–25 min, total ~3–5 h.

C) **Run only a subset of tau** (e.g. tau ∈ {0.5, 1.0, 2.0, 4.0}) to
   demonstrate the methodology and provide accurate sharpened values
   at the most relevant points. Total ~2 h.

D) **Lower dps to 50** (the original spec): each elementary mp op is
   roughly 2x faster, halo build and phi step both halve. mp50 still
   reaches ‖F‖<1e-30 comfortably; reaching 1e-50 may need an extra
   iter near the precision limit.

The architecture is in place either way; the barrier is wall-clock
compute on a pure-Python mpmath kernel. A C/Cython port of the
contour-evidence inner loop would take this from hours to minutes,
matching the seed file's mp300 timings (~10 min/iter at G=20).
