"""Cesaro-average Picard-PAVA iterates to extract the monotone FP.

Mann iteration: μ_{n+1} = α_n · μ_n + (1-α_n) · Ψ(μ_n) with α_n=1/(n+1)
gives ergodic convergence for non-expansive maps. Equivalently we run
plain Picard-PAVA and take the running average of iterates.

The Φ residual at the average is small (the cycle is nearly symmetric),
the answer 1-R² is the genuine monotone fixed point.
"""
import numpy as np
import time

from posterior_method_v3 import (
    Lam, init_p_grid, project_monotone, phi_step, measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5

ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu = project_monotone(ckpt["mu"], ckpt["u_grid"], None)
u_grid = ckpt["u_grid"]
p_grid = ckpt["p_grid"]; p_lo = ckpt["p_lo"]; p_hi = ckpt["p_hi"]

print("Running 1000 Picard-PAVA iterations and Cesaro-averaging the last 500...",
      flush=True)
N_TOTAL = 1000
N_AVG = 500
mu_sum = np.zeros_like(mu)
n_collected = 0

t0 = time.time()
for it in range(N_TOTAL):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    cand = project_monotone(cand, u_grid, None)
    mu = 0.05 * cand + 0.95 * mu
    mu = np.clip(mu, EPS, 1 - EPS)
    if it >= N_TOTAL - N_AVG:
        mu_sum += mu
        n_collected += 1
    if (it + 1) % 100 == 0:
        cand2, act2, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        F_inf = float(np.max(np.abs(cand2 - mu)[act2]))
        print(f"  iter={it+1}, ||Φ-μ||∞={F_inf:.3e}", flush=True)

mu_avg = mu_sum / n_collected
mu_avg = project_monotone(mu_avg, u_grid, None)

# Measure
cand, active, _ = phi_step(mu_avg, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F = cand - mu_avg
phi_res = float(np.max(np.abs(F[active])))
phi_res_med = float(np.median(np.abs(F[active])))
phi_res_l2 = float(np.sqrt(np.mean(F[active]**2)))
n_u = int((np.diff(mu_avg, axis=0) < 0).sum())
n_p = int((np.diff(mu_avg, axis=1) < 0).sum())
r2, slope, _ = measure_R2(mu_avg, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)

print(f"\n=== Cesaro-averaged μ ===")
print(f"||Φ(μ_avg) - μ_avg||∞ = {phi_res:.3e}")
print(f"||Φ(μ_avg) - μ_avg||_med = {phi_res_med:.3e}")
print(f"||Φ(μ_avg) - μ_avg||_L2 = {phi_res_l2:.3e}")
print(f"violations: u={n_u}, p={n_p}")
print(f"1-R² = {r2:.6e}")
print(f"slope = {slope:.4f}")
print(f"elapsed = {time.time()-t0:.1f}s")

# Compare
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu_nm = ckpt["mu"]
r2_nm, slope_nm, _ = measure_R2(mu_nm, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\n=== Comparison ===")
print(f"  Non-monotone Φ-FP (NK):  1-R²={r2_nm:.6e}, slope={slope_nm:.4f}")
print(f"  Monotone Cesaro-avg μ:   1-R²={r2:.6e},  slope={slope:.4f}")

np.savez("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz",
         mu=mu_avg, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi,
         tau=TAU, gamma=GAMMA)
print("\nSaved monotone-Cesaro μ to results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
