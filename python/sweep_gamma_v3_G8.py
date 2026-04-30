"""γ-sweep at G=8 with the v3 posterior method (only G size that converges).

Sanity-check: does (1-R²) decrease as γ → ∞ (CARA limit, FR)?
And does slope → 1?
"""
import json
import time
import numpy as np
from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, measure_R2,
)

GAMMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
TAU = 2.0
Gu = Gp = 8
UMAX = 3.0

results = []
u_grid = np.linspace(-UMAX, UMAX, Gu)

for gamma in GAMMAS:
    print(f"\n=== γ = {gamma} ===", flush=True)
    t0 = time.time()
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, gamma, Gp)
    mu0 = np.zeros((Gu, Gp))
    for i, u in enumerate(u_grid):
        mu0[i, :] = Lam(TAU * u)

    mu_final, hist, conv = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, TAU, gamma,
        damping=0.05, anderson=0, max_iter=600, tol=1e-9, progress=False,
    )
    fit_t = time.time() - t0
    print(f"  iters={len(hist)}, residual={hist[-1][0]:.3e}, conv={conv}, t={fit_t:.1f}s",
          flush=True)

    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    print(f"  1-R² = {r2def:.6e}, slope = {slope:.4f}", flush=True)

    results.append({
        "gamma": gamma, "tau": TAU, "Gu": Gu, "Gp": Gp,
        "iterations": len(hist), "converged": bool(conv),
        "residual_inf": float(hist[-1][0]),
        "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n),
        "elapsed_s": fit_t,
    })

print("\n=== SUMMARY ===")
print(f"{'γ':>6} {'iters':>6} {'resid':>10} {'1-R²':>12} {'slope':>8} {'conv':>6}")
for r in results:
    print(f"{r['gamma']:>6.2f} {r['iterations']:>6d} {r['residual_inf']:>10.2e} "
          f"{r['1-R^2']:>12.4e} {r['slope']:>8.4f} {str(r['converged']):>6}")

with open("results/full_ree/posterior_v3_G8_gamma_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/full_ree/posterior_v3_G8_gamma_sweep.json")
