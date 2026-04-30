"""Extended G-ladder for posterior method v3 at γ=0.5, τ=2.

Methods tried per G:
  1. Picard with damping=0.05 (baseline; 2-cycles at G≥12)
  2. Newton-Krylov on F = Φ(μ) - μ  (scipy.optimize.newton_krylov)

Plus Richardson extrapolation: predict 1-R²(G=12) from G=8, G=10
under linear-in-1/G and linear-in-1/G² models.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)

GS = [8, 10, 12, 14, 16]
TAU = 2.0
GAMMA = 0.5
UMAX = 4.0


def F_residual(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """F(μ) = Φ(μ) - μ as flat vector. Inactive cells get 0 residual."""
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu).ravel()
    # Zero out inactive cells (they're consistent by construction)
    F[~active.ravel()] = 0.0
    return F


def solve_G(G, method="picard", max_iter=800, damping=0.05, verbose=True):
    """Solve at grid size G with the chosen method."""
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    mu0 = np.zeros((G, G))
    for i, u in enumerate(u_grid):
        mu0[i, :] = Lam(TAU * u)

    t0 = time.time()
    if method == "picard":
        mu_final, hist, conv = picard_anderson(
            mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
            damping=damping, anderson=0, max_iter=max_iter, tol=1e-9, progress=False,
        )
        residual = float(hist[-1][0])
        active = int(hist[-1][1])
        iters = len(hist)
    elif method == "picard_warmed_nk":
        # Warm with picard then refine with Newton-Krylov
        mu_warm, hist, _ = picard_anderson(
            mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
            damping=damping, anderson=0, max_iter=max_iter // 2, tol=1e-9,
            progress=False,
        )
        try:
            mu_final = newton_krylov(
                lambda x: F_residual(x, mu_warm.shape, u_grid, p_grid,
                                      p_lo, p_hi, TAU, GAMMA),
                mu_warm.ravel(),
                f_tol=1e-7, maxiter=80, verbose=False,
                method="lgmres",
            ).reshape(mu_warm.shape)
            mu_final = np.clip(mu_final, EPS, 1 - EPS)
            cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi,
                                             TAU, GAMMA)
            residual = float(np.max(np.abs(cand - mu_final)[active_mask]))
            active = int(active_mask.sum())
            conv = residual < 1e-7
        except (NoConvergence, ValueError) as e:
            print(f"    [NK failed: {e}, falling back to Picard warm]", flush=True)
            mu_final = mu_warm
            cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi,
                                             TAU, GAMMA)
            residual = float(np.max(np.abs(cand - mu_final)[active_mask]))
            active = int(active_mask.sum())
            conv = False
        iters = len(hist) + 80
    elif method == "newton_krylov":
        try:
            mu_final = newton_krylov(
                lambda x: F_residual(x, mu0.shape, u_grid, p_grid,
                                      p_lo, p_hi, TAU, GAMMA),
                mu0.ravel(),
                f_tol=1e-7, maxiter=120, verbose=False,
                method="lgmres",
            ).reshape(mu0.shape)
            mu_final = np.clip(mu_final, EPS, 1 - EPS)
            cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi,
                                             TAU, GAMMA)
            residual = float(np.max(np.abs(cand - mu_final)[active_mask]))
            active = int(active_mask.sum())
            conv = residual < 1e-7
        except (NoConvergence, ValueError) as e:
            print(f"    [NK failed: {e}]", flush=True)
            mu_final = mu0
            residual = float("nan")
            active = 0
            conv = False
        iters = 120
    else:
        raise ValueError(f"unknown method {method}")

    fit_t = time.time() - t0
    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    if verbose:
        print(f"  [{method}] iters={iters}, resid={residual:.3e}, active={active}, "
              f"1-R²={r2def:.4e}, slope={slope:.4f}, conv={conv}, t={fit_t:.1f}s",
              flush=True)
    return {
        "G": G, "method": method, "iterations": iters,
        "converged": bool(conv), "residual_inf": residual,
        "active_cells": active, "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n), "elapsed_s": fit_t,
    }


def richardson_predict(g_list, val_list, target_G):
    """Fit val(G) = c + a/G^p for p in {1, 2} from two points,
    predict at target_G."""
    g1, g2 = g_list[0], g_list[1]
    v1, v2 = val_list[0], val_list[1]
    out = {}
    for p in (1, 2):
        denom = (1.0/g1**p) - (1.0/g2**p)
        a = (v1 - v2) / denom if denom != 0 else 0.0
        c = v1 - a / g1**p
        pred = c + a / target_G**p
        out[f"1/G^{p}"] = {"a": float(a), "c": float(c), "predicted": float(pred)}
    return out


print(f"=== G-ladder, γ={GAMMA}, τ={TAU}, umax={UMAX} ===\n", flush=True)

results = []
warnings.filterwarnings("ignore", category=RuntimeWarning)
for G in GS:
    print(f"--- G = {G} ---", flush=True)
    # Always try picard first (fast warm start)
    r1 = solve_G(G, method="picard", max_iter=600, damping=0.05)
    results.append(r1)
    # If not converged, try picard+Newton-Krylov refinement
    if not r1["converged"] and G >= 12:
        r2 = solve_G(G, method="picard_warmed_nk", max_iter=400, damping=0.05)
        results.append(r2)

print("\n=== ALL RESULTS ===", flush=True)
print(f"{'G':>3} {'method':>20} {'iters':>6} {'resid':>10} "
      f"{'1-R²':>11} {'slope':>7} {'conv':>6}")
for r in results:
    print(f"{r['G']:>3d} {r['method']:>20} {r['iterations']:>6d} "
          f"{r['residual_inf']:>10.2e} {r['1-R^2']:>11.4e} "
          f"{r['slope']:>7.4f} {str(r['converged']):>6}")

# Richardson extrapolation from G=8,10 → predict G=12,14,16
conv_results = {r["G"]: r for r in results if r["method"] == "picard" and r["converged"]}
if 8 in conv_results and 10 in conv_results:
    g_list = [8, 10]
    r2_list = [conv_results[8]["1-R^2"], conv_results[10]["1-R^2"]]
    slope_list = [conv_results[8]["slope"], conv_results[10]["slope"]]
    print(f"\n=== RICHARDSON EXTRAPOLATION (from G=8, G=10) ===")
    for target in [12, 14, 16, 20]:
        pred_r2 = richardson_predict(g_list, r2_list, target)
        pred_slope = richardson_predict(g_list, slope_list, target)
        print(f"  G={target}:")
        for model in ("1/G^1", "1/G^2"):
            print(f"    {model}: 1-R² ~ {pred_r2[model]['predicted']:.4e}, "
                  f"slope ~ {pred_slope[model]['predicted']:.4f}")
        if target in conv_results or any(r["G"]==target and r["1-R^2"] for r in results):
            actual = next(r for r in results if r["G"]==target)
            print(f"    actual:    1-R² = {actual['1-R^2']:.4e}, "
                  f"slope = {actual['slope']:.4f} "
                  f"({'conv' if actual['converged'] else 'cycle/avg'})")

with open("results/full_ree/posterior_v3_G_ladder_extended.json", "w") as f:
    json.dump({"results": results, "params": {
        "tau": TAU, "gamma": GAMMA, "umax": UMAX,
    }}, f, indent=2)
print("\nSaved: results/full_ree/posterior_v3_G_ladder_extended.json")
