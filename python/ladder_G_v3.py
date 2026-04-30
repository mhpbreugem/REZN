"""G-ladder for posterior method v3 at fixed γ, τ.

Tests how 1-R², slope, and convergence behave as G increases.
"""
import json
import time
import numpy as np
from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, measure_R2,
)

GS = [8, 10, 12]
TAU = 2.0
GAMMA = 0.5
UMAX = 4.0
DAMPING = 0.05
MAX_ITER = 800
TOL = 1e-9

results = []
for G in GS:
    print(f"\n=== G = {G} ===", flush=True)
    u_grid = np.linspace(-UMAX, UMAX, G)
    t0 = time.time()
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    mu0 = np.zeros((G, G))
    for i, u in enumerate(u_grid):
        mu0[i, :] = Lam(TAU * u)

    mu_final, hist, conv = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
        damping=DAMPING, anderson=0, max_iter=MAX_ITER, tol=TOL, progress=False,
    )
    fit_t = time.time() - t0
    print(f"  iters={len(hist)}, residual={hist[-1][0]:.3e}, "
          f"active={hist[-1][1]}, conv={conv}, t={fit_t:.1f}s", flush=True)

    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  1-R² = {r2def:.6e}, slope = {slope:.4f}, n_samples = {n}", flush=True)

    results.append({
        "G": G, "tau": TAU, "gamma": GAMMA, "umax": UMAX,
        "iterations": len(hist), "converged": bool(conv),
        "residual_inf": float(hist[-1][0]),
        "active_cells": int(hist[-1][1]),
        "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n),
        "elapsed_s": fit_t,
    })

print("\n=== SUMMARY ===")
print(f"{'G':>3} {'iters':>6} {'resid':>10} {'active':>7} {'1-R²':>12} {'slope':>8} {'conv':>6}")
for r in results:
    print(f"{r['G']:>3d} {r['iterations']:>6d} {r['residual_inf']:>10.2e} "
          f"{r['active_cells']:>7d} {r['1-R^2']:>12.4e} {r['slope']:>8.4f} "
          f"{str(r['converged']):>6}")

with open("results/full_ree/posterior_v3_G_ladder.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/full_ree/posterior_v3_G_ladder.json")
