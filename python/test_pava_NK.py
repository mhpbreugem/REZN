"""Find the monotone fixed point of Ψ = PAVA ∘ Φ via Newton-Krylov.

The plain Φ has a non-monotone fixed point at G=14 (1-R²=0.1084, with
22 monotonicity violations). PAVA(μ*) is monotone but NOT Φ-fixed.

The composite map Ψ(μ) = PAVA(Φ(μ)) projects every Φ-iterate onto the
monotone cone. Its fixed point is the monotone economic-equilibrium μ.

We find it with Newton-Krylov on F(μ) = Ψ(μ) - μ.
"""
import numpy as np
import time
import warnings
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, project_monotone, measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5

# Load NK-converged (non-monotone) G=14 as warm start
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu0 = ckpt["mu"].copy()
u_grid = ckpt["u_grid"]
p_grid = ckpt["p_grid"]; p_lo = ckpt["p_lo"]; p_hi = ckpt["p_hi"]
shape = mu0.shape

# Project the warm-start onto monotone cone first
mu0 = project_monotone(mu0, u_grid, None)
print(f"Warm start (G=14, projected): shape {shape}", flush=True)

def F_psi(mu_flat):
    """F(μ) = Ψ(μ) - μ = PAVA(Φ(μ)) - μ"""
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    cand = project_monotone(cand, u_grid, None)
    F = (cand - mu).ravel()
    F[~active.ravel()] = 0.0
    return F

warnings.filterwarnings("ignore", category=RuntimeWarning)
print("Running NK on F(μ) = PAVA(Φ(μ)) - μ ...", flush=True)
t0 = time.time()
nk_status = "ok"
try:
    sol = newton_krylov(F_psi, mu0.ravel(), f_tol=1e-12, maxiter=200,
                         method="lgmres", verbose=False)
    mu_final = sol.reshape(shape)
except NoConvergence as e:
    nk_status = "noconv"
    mu_final = e.args[0].reshape(shape) if e.args else mu0
mu_final = np.clip(project_monotone(mu_final, u_grid, None), EPS, 1 - EPS)

# Measure
cand, active, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
psi_resid = float(np.max(np.abs(project_monotone(cand, u_grid, None) - mu_final)[active]))
phi_resid = float(np.max(np.abs(cand - mu_final)[active]))
r2, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"NK status: {nk_status}", flush=True)
print(f"  Ψ residual: {psi_resid:.3e}", flush=True)
print(f"  Φ residual: {phi_resid:.3e}", flush=True)
print(f"  1-R² = {r2:.6e}", flush=True)
print(f"  slope = {slope:.4f}", flush=True)
print(f"  elapsed = {time.time()-t0:.1f}s", flush=True)

# Compare to non-monotone NK fixed point
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu_nonmon = ckpt["mu"]
r2_nm, slope_nm, _ = measure_R2(mu_nonmon, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\n=== Comparison ===")
print(f"  Non-monotone Φ-FP: 1-R²={r2_nm:.6e}, slope={slope_nm:.4f}")
print(f"  Monotone Ψ-FP:     1-R²={r2:.6e}, slope={slope:.4f}")

# Save monotone solution
np.savez("results/full_ree/posterior_v3_G14_PAVA_mu.npz",
         mu=mu_final, u_grid=u_grid, p_grid=p_grid,
         p_lo=p_lo, p_hi=p_hi, tau=TAU, gamma=GAMMA)
print(f"\nSaved monotone Ψ-FP to results/full_ree/posterior_v3_G14_PAVA_mu.npz")
