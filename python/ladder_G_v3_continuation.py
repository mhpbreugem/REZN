"""G-ladder with continuation: solve at G_n by NK from G_{n-1} interpolated up.

This avoids picard's 2-cycle being a bad NK warm-start at large G.
Each NK call sees a warm-start that is already very close to the FP.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence
from scipy.interpolate import RegularGridInterpolator

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)

GS = [8, 10, 12, 14, 16, 18, 20]
TAU = 2.0
GAMMA = 0.5
UMAX = 4.0


def F_residual(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu).ravel()
    F[~active.ravel()] = 0.0
    return F


def interp_mu_to_grid(mu_old, u_old, p_old, p_lo_old, p_hi_old,
                      u_new, p_new):
    """Map μ_old(u_old, p_old[i,:]) → μ_new(u_new, p_new[i,:]).

    Per-row: each u_new[i'] gets its mu by interpolating row-by-row
    in u then in p (per-row p-grid).
    """
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    # 1D per-row p-grid → first interpolate each old row in p to a
    # uniform p-axis, then do u-interp, then back to new per-row p-axis.
    # Simpler: for each (i', j'), find u_new[i'], p_new[i',j'], then
    #   evaluate mu_old(u, p) by:
    #     row index r in old-grid s.t. u_new[i'] near u_old[r]
    #     interpolate mu_old[r, :] in p along p_old[r, :] to p_new[i', j']
    #     do this for r = floor and ceil, then interp in u.
    for i_new in range(G_new):
        u_target = u_new[i_new]
        # Find old row indices bracketing
        r_above = np.searchsorted(u_old, u_target)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        if r_above == r_below:
            w = 1.0
        else:
            w = (u_target - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            # interp mu_old[r_below, :] in p_old[r_below]
            p_clamped = np.clip(p_target,
                                 p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_clamped, p_old[r_below, :], mu_old[r_below, :])
            p_clamped = np.clip(p_target,
                                 p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_clamped, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def solve_G(G, mu_warm=None, u_warm=None, p_warm=None,
            p_lo_warm=None, p_hi_warm=None,
            picard_iters=200, nk_iters=120):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)

    if mu_warm is None:
        mu0 = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu0[i, :] = Lam(TAU * u)
    else:
        mu0 = interp_mu_to_grid(mu_warm, u_warm, p_warm, p_lo_warm, p_hi_warm,
                                 u_grid, p_grid)
        print(f"  warm-start interp done; max(mu0)={mu0.max():.4f}, "
              f"min={mu0.min():.4f}", flush=True)

    t0 = time.time()
    # Brief picard polish
    mu_warm_out, hist, _ = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
        damping=0.05, anderson=0, max_iter=picard_iters, tol=1e-9, progress=False,
    )
    picard_resid = float(hist[-1][0])
    print(f"  [picard {picard_iters}] resid={picard_resid:.3e} "
          f"after {len(hist)} iters", flush=True)

    # NK refine
    try:
        mu_final = newton_krylov(
            lambda x: F_residual(x, mu_warm_out.shape, u_grid, p_grid,
                                  p_lo, p_hi, TAU, GAMMA),
            mu_warm_out.ravel(),
            f_tol=1e-9, maxiter=nk_iters, verbose=False,
            method="lgmres",
        ).reshape(mu_warm_out.shape)
        mu_final = np.clip(mu_final, EPS, 1 - EPS)
        cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi,
                                         TAU, GAMMA)
        residual = float(np.max(np.abs(cand - mu_final)[active_mask]))
        active = int(active_mask.sum())
        conv = residual < 1e-7
    except (NoConvergence, ValueError) as e:
        print(f"  [NK failed: {e}]", flush=True)
        mu_final = mu_warm_out
        cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi,
                                         TAU, GAMMA)
        residual = float(np.max(np.abs(cand - mu_final)[active_mask]))
        active = int(active_mask.sum())
        conv = False

    fit_t = time.time() - t0
    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  [final] resid={residual:.3e}, active={active}, "
          f"1-R²={r2def:.4e}, slope={slope:.4f}, conv={conv}, t={fit_t:.1f}s",
          flush=True)
    return {
        "G": G, "method": "continuation_picard_nk",
        "picard_iters": picard_iters, "picard_resid": picard_resid,
        "iterations": picard_iters + nk_iters, "converged": bool(conv),
        "residual_inf": residual, "active_cells": active,
        "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n), "elapsed_s": fit_t,
    }, mu_final, u_grid, p_grid, p_lo, p_hi


def richardson(g_list, val_list, target_G):
    g1, g2 = g_list[0], g_list[1]
    v1, v2 = val_list[0], val_list[1]
    out = {}
    for p in (1, 2):
        denom = (1.0/g1**p) - (1.0/g2**p)
        a = (v1 - v2) / denom if denom != 0 else 0.0
        c = v1 - a / g1**p
        out[f"1/G^{p}"] = c + a / target_G**p
    return out


print(f"=== G-ladder with continuation, γ={GAMMA}, τ={TAU} ===\n", flush=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
results = []
mu_prev = u_prev = p_prev = p_lo_prev = p_hi_prev = None

for G in GS:
    print(f"--- G = {G} ---", flush=True)
    r, mu, u_g, p_g, p_lo_g, p_hi_g = solve_G(
        G, mu_warm=mu_prev, u_warm=u_prev, p_warm=p_prev,
        p_lo_warm=p_lo_prev, p_hi_warm=p_hi_prev,
        picard_iters=200, nk_iters=150,
    )
    results.append(r)
    if r["converged"]:
        mu_prev, u_prev, p_prev, p_lo_prev, p_hi_prev = mu, u_g, p_g, p_lo_g, p_hi_g

print("\n=== ALL RESULTS ===")
print(f"{'G':>3} {'iters':>6} {'resid':>10} {'1-R²':>11} {'slope':>7} {'conv':>6}")
for r in results:
    print(f"{r['G']:>3d} {r['iterations']:>6d} {r['residual_inf']:>10.2e} "
          f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f} {str(r['converged']):>6}")

# Richardson extrapolation: predict each new G from earliest two
conv_results = [r for r in results if r["converged"]]
if len(conv_results) >= 2:
    print(f"\n=== RICHARDSON EXTRAPOLATION ===")
    print(f"Predict G=12,14,16,20 from G=8,G=10 converged values:")
    g8 = next(r for r in conv_results if r["G"] == 8)
    g10 = next(r for r in conv_results if r["G"] == 10)
    for target in [12, 14, 16, 18, 20]:
        pred_r2 = richardson([8, 10], [g8["1-R^2"], g10["1-R^2"]], target)
        pred_slope = richardson([8, 10], [g8["slope"], g10["slope"]], target)
        actual = next((r for r in conv_results if r["G"] == target), None)
        actual_str = (f"actual 1-R²={actual['1-R^2']:.4e}, slope={actual['slope']:.4f}"
                      if actual else "(not solved)")
        print(f"  G={target:>2}: 1-R² pred 1/G={pred_r2['1/G^1']:.4e}, "
              f"1/G²={pred_r2['1/G^2']:.4e}; "
              f"slope pred 1/G={pred_slope['1/G^1']:.4f}, "
              f"1/G²={pred_slope['1/G^2']:.4f}; {actual_str}")

with open("results/full_ree/posterior_v3_G_continuation.json", "w") as f:
    json.dump({"results": results, "params": {
        "tau": TAU, "gamma": GAMMA, "umax": UMAX,
    }}, f, indent=2)
print("\nSaved: results/full_ree/posterior_v3_G_continuation.json")
