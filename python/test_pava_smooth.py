"""Smoothed-isotonic NK approach.

Define the C¹ penalty
    Pen(μ) = ½ Σ_{i,j} ReLU(μ[i,j] - μ[i+1,j])²        (u-direction)
           + ½ Σ_{i,j} ReLU(μ[i,j] - μ[i,j+1])²        (p-direction)

The augmented residual
    F_aug(μ) = (Φ(μ) - μ) + λ · ∇Pen(μ)

is C¹ in μ (because ReLU² is C¹: its derivative is 2·ReLU(·), continuous).
NK's FD-JVP is no longer zero because the penalty perturbation is non-trivial.

For a TRUE monotone Φ-fixed-point: ∇Pen = 0 AND Φ(μ) = μ → F_aug = 0.
For a NON-monotone Φ-fixed-point: Φ(μ) = μ but ∇Pen ≠ 0 → F_aug ≠ 0
(rejected). So zeros of F_aug = monotone Φ-fixed-points exactly.

We sweep λ and look for the limit as λ → 0 where the answer stabilizes.
"""
import numpy as np
import time
import warnings
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    init_p_grid, picard_anderson, project_monotone, phi_step,
    measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5
LAMBDAS = [10.0, 1.0, 0.1, 0.01, 0.001]


def pen_grad(mu):
    """∇Pen(μ) where Pen = ½ Σ ReLU(violation)²."""
    g = np.zeros_like(mu)
    # u-direction violations: μ[i,j] - μ[i+1,j] > 0 (μ decreasing in u)
    diff_u = mu[:-1, :] - mu[1:, :]
    viol_u = np.maximum(diff_u, 0.0)        # ReLU(violation)
    # ∂Pen/∂μ[i,j] += 2·ReLU(μ[i,j] - μ[i+1,j])  (when i<Gu-1)
    g[:-1, :] += viol_u
    # ∂Pen/∂μ[i+1,j] -= 2·ReLU(μ[i,j] - μ[i+1,j])
    g[1:, :] -= viol_u
    # p-direction violations: μ[i,j] - μ[i,j+1] > 0
    diff_p = mu[:, :-1] - mu[:, 1:]
    viol_p = np.maximum(diff_p, 0.0)
    g[:, :-1] += viol_p
    g[:, 1:] -= viol_p
    return g


def F_aug(mu_flat, shape, lam, u_grid, p_grid, p_lo, p_hi):
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F = cand - mu
    F[~active] = 0.0
    F = F + lam * pen_grad(mu)
    return F.ravel()


def count_violations(mu):
    n_u = int((np.diff(mu, axis=0) < 0).sum())
    n_p = int((np.diff(mu, axis=1) < 0).sum())
    return n_u, n_p


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load monotone-projected G=14 warm
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu0 = project_monotone(ckpt["mu"], ckpt["u_grid"], None)
u_grid = ckpt["u_grid"]; p_grid = ckpt["p_grid"]
p_lo = ckpt["p_lo"]; p_hi = ckpt["p_hi"]
shape = mu0.shape

print(f"=== smoothed-PAVA via softplus penalty ===")
print(f"G={shape[0]}, τ={TAU}, γ={GAMMA}\n")

results = []
mu_warm = mu0.copy()
for lam in LAMBDAS:
    print(f"--- λ = {lam} ---", flush=True)
    t0 = time.time()
    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_aug(x, shape, lam, u_grid, p_grid, p_lo, p_hi),
            mu_warm.ravel(),
            f_tol=1e-12, maxiter=300,
            method="lgmres", verbose=False,
        )
        mu_final = sol.reshape(shape)
    except NoConvergence as e:
        nk_status = "noconv"
        mu_final = e.args[0].reshape(shape) if e.args else mu_warm
    except (ValueError, RuntimeError) as e:
        nk_status = f"err:{type(e).__name__}"
        mu_final = mu_warm
    mu_final = np.clip(mu_final, EPS, 1 - EPS)
    cand, active, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    phi_res = float(np.max(np.abs(cand - mu_final)[active]))
    pen = float(0.5 * np.sum(np.maximum(np.diff(mu_final, axis=0) * -1, 0)**2)
                + 0.5 * np.sum(np.maximum(np.diff(mu_final, axis=1) * -1, 0)**2))
    n_u, n_p = count_violations(mu_final)
    r2, slope, _ = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    elapsed = time.time() - t0
    print(f"  NK_status={nk_status}, Φ-resid={phi_res:.3e}, Pen={pen:.3e}, "
          f"viol u={n_u}, p={n_p}, 1-R²={r2:.4e}, slope={slope:.4f}, "
          f"t={elapsed:.1f}s",
          flush=True)
    results.append({
        "lambda": lam, "nk_status": nk_status,
        "phi_residual": phi_res, "penalty": pen,
        "violations_u": n_u, "violations_p": n_p,
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed,
    })
    # Warm-start next λ
    mu_warm = mu_final

print(f"\n=== SUMMARY ===")
print(f"{'λ':>10} {'NK':>8} {'Φ-resid':>10} {'Pen':>10} "
      f"{'viol':>6} {'1-R²':>11} {'slope':>7}")
for r in results:
    print(f"{r['lambda']:>10.3g} {r['nk_status']:>8} "
          f"{r['phi_residual']:>10.2e} {r['penalty']:>10.2e} "
          f"{r['violations_u']+r['violations_p']:>6d} "
          f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f}")
print("\nAs λ → 0: should converge to the true monotone Φ-FP.")
import json
with open("results/full_ree/posterior_v3_PAVA_smooth_lambda_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
