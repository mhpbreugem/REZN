"""Test whether PAVA modifies existing NK-converged μ.

If the converged μ is already monotone, PAVA is a no-op and the answer
is robust. If not, the NK solution had non-monotone artifacts and PAVA
gives a corrected version.
"""
import numpy as np
from posterior_method_v3 import (
    Lam, project_monotone, phi_step, measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5

# Load NK-converged G=14, UMAX=4 (best solution)
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu = ckpt["mu"]; u_grid = ckpt["u_grid"]
p_grid = ckpt["p_grid"]; p_lo = ckpt["p_lo"]; p_hi = ckpt["p_hi"]
print(f"G=14 UMAX=4 checkpoint: mu shape {mu.shape}")

# Check monotonicity violations
def count_violations(mu):
    Gu, Gp = mu.shape
    n_u_viol = 0
    for j in range(Gp):
        diffs = np.diff(mu[:, j])
        n_u_viol += int(np.sum(diffs < 0))
    n_p_viol = 0
    for i in range(Gu):
        diffs = np.diff(mu[i, :])
        n_p_viol += int(np.sum(diffs < 0))
    return n_u_viol, n_p_viol

n_u, n_p = count_violations(mu)
print(f"\nMonotonicity violations BEFORE PAVA:")
print(f"  ∂μ/∂u < 0: {n_u} pairs (out of {(mu.shape[0]-1)*mu.shape[1]})")
print(f"  ∂μ/∂p < 0: {n_p} pairs (out of {mu.shape[0]*(mu.shape[1]-1)})")

# Apply PAVA
mu_proj = project_monotone(mu, u_grid, None)
n_u2, n_p2 = count_violations(mu_proj)
print(f"\nAfter PAVA:")
print(f"  ∂μ/∂u < 0: {n_u2}")
print(f"  ∂μ/∂p < 0: {n_p2}")

# Distance moved
diff = mu_proj - mu
print(f"\n||PAVA(μ) - μ||∞ = {float(np.max(np.abs(diff))):.4e}")
print(f"||PAVA(μ) - μ||₂ = {float(np.linalg.norm(diff)):.4e}")
print(f"max above-zero shift: {float(diff.max()):+.4e}")
print(f"min below-zero shift: {float(diff.min()):+.4e}")

# Re-measure 1-R² with the projected μ
r2_orig, slope_orig, _ = measure_R2(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
r2_proj, slope_proj, _ = measure_R2(mu_proj, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\n1-R² before PAVA: {r2_orig:.6e}, slope = {slope_orig:.4f}")
print(f"1-R² after PAVA:  {r2_proj:.6e}, slope = {slope_proj:.4f}")

# Check Phi residual after PAVA
cand, active, _ = phi_step(mu_proj, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F = cand - mu_proj
print(f"\nResidual ||Φ(PAVA(μ)) - PAVA(μ)||∞ = {float(np.max(np.abs(F[active]))):.4e}")
print(f"For comparison ||Φ(μ) - μ||∞ = ", end="")
cand0, active0, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"{float(np.max(np.abs((cand0 - mu)[active0]))):.4e}")
