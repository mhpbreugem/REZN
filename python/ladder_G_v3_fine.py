"""Fine-step continuation ladder with strict f_tol=1e-14 enforcement.

Step from G=8 in single-G increments (G=8,9,10,...,20). Each step uses
the previous NK-converged μ as warm-start, interpolated to the new grid.

NO compromise on tolerance: a step is "converged" only if max-residual
on active cells < 1e-12 (effectively machine precision after NK).

Strategy when a step doesn't converge:
  1. Try with more NK iters (up to 800)
  2. Try a half-step refinement: solve at G_prev + 1 first, then continue
  3. Mark non-converged but report

Outputs both the strict-converged and partial-NK 1-R² values per G.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)

GS = list(range(8, 21))   # 8, 9, 10, ..., 20
TAU = 2.0
GAMMA = 0.5
UMAX = 4.0
F_TOL = 1e-14


def F_residual(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu).ravel()
    F[~active.ravel()] = 0.0
    return F


def interp_mu_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        r_above = np.searchsorted(u_old, u_target)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        if r_above == r_below:
            w = 1.0
        else:
            w = (u_target - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_clamped_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_clamped_b, p_old[r_below, :], mu_old[r_below, :])
            p_clamped_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_clamped_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def solve_G(G, mu_warm=None, u_warm=None, p_warm=None,
            picard_iters=200, nk_iters=800, f_tol=F_TOL, label=""):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    if mu_warm is None:
        mu0 = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu0[i, :] = Lam(TAU * u)
    else:
        mu0 = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)

    t0 = time.time()
    mu_warm_out, hist, _ = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
        damping=0.05, anderson=0, max_iter=picard_iters, tol=1e-12, progress=False,
    )
    picard_resid = float(hist[-1][0])

    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual(x, mu_warm_out.shape, u_grid, p_grid,
                                  p_lo, p_hi, TAU, GAMMA),
            mu_warm_out.ravel(),
            f_tol=f_tol, maxiter=nk_iters, verbose=False,
            method="lgmres",
        )
        mu_final = sol.reshape(mu_warm_out.shape)
    except NoConvergence as e:
        nk_status = "noconv"
        mu_final = e.args[0].reshape(mu_warm_out.shape) if e.args else mu_warm_out
    except (ValueError, RuntimeError) as e:
        nk_status = f"err:{type(e).__name__}"
        mu_final = mu_warm_out

    mu_final = np.clip(mu_final, EPS, 1 - EPS)
    cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_arr = np.abs(cand - mu_final)
    residual_max = float(F_arr[active_mask].max())
    residual_median = float(np.median(F_arr[active_mask]))
    active = int(active_mask.sum())
    strict_conv = residual_max < 1e-12
    fit_t = time.time() - t0
    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  [{label}] G={G} picard_resid={picard_resid:.3e}  "
          f"NK_status={nk_status}  max={residual_max:.3e}  "
          f"med={residual_median:.3e}  1-R²={r2def:.4e}  "
          f"slope={slope:.4f}  conv={strict_conv}  t={fit_t:.1f}s",
          flush=True)
    return {
        "G": G, "label": label,
        "picard_iters": picard_iters, "picard_resid": picard_resid,
        "nk_iters": nk_iters, "nk_status": nk_status,
        "residual_max": residual_max, "residual_median": residual_median,
        "active_cells": active,
        "converged": bool(strict_conv),
        "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n), "elapsed_s": fit_t,
    }, mu_final, u_grid, p_grid, p_lo, p_hi


def richardson_pred(g_list, val_list, target_G):
    g1, g2 = g_list[0], g_list[1]
    v1, v2 = val_list[0], val_list[1]
    out = {}
    for p in (1, 2):
        denom = (1.0/g1**p) - (1.0/g2**p)
        a = (v1 - v2) / denom if denom != 0 else 0.0
        c = v1 - a / g1**p
        out[f"1/G^{p}"] = c + a / target_G**p
    return out


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== fine-step ladder, γ={GAMMA}, τ={TAU}, f_tol={F_TOL} ===\n", flush=True)

results = []
mu_prev = u_prev = p_prev = None

for G in GS:
    print(f"--- G = {G} ---", flush=True)
    r, mu, ug, pg, p_lo, p_hi = solve_G(
        G, mu_warm=mu_prev, u_warm=u_prev, p_warm=p_prev,
        picard_iters=300, nk_iters=800, f_tol=F_TOL,
        label="warm" if mu_prev is not None else "cold",
    )
    results.append(r)
    if r["converged"]:
        mu_prev, u_prev, p_prev = mu, ug, pg
    else:
        # Try cold-start NK as fallback (sometimes a different attractor
        # is reachable without warm-start bias)
        print(f"  [retry cold-start] G={G}", flush=True)
        r2, mu2, ug2, pg2, p_lo2, p_hi2 = solve_G(
            G, mu_warm=None, u_warm=None, p_warm=None,
            picard_iters=600, nk_iters=800, f_tol=F_TOL, label="cold-retry",
        )
        results.append(r2)
        if r2["converged"]:
            mu_prev, u_prev, p_prev = mu2, ug2, pg2

print("\n=== ALL RESULTS ===")
print(f"{'G':>3} {'label':>12} {'NK':>8} {'max':>10} {'med':>10} "
      f"{'1-R²':>11} {'slope':>7} {'conv':>6}")
for r in results:
    print(f"{r['G']:>3d} {r['label']:>12} {r['nk_status']:>8} "
          f"{r['residual_max']:>10.2e} {r['residual_median']:>10.2e} "
          f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f} "
          f"{str(r['converged']):>6}")

# Richardson from earliest two converged
conv_list = [r for r in results if r["converged"]]
if len(conv_list) >= 2:
    print(f"\n=== RICHARDSON EXTRAPOLATION (from converged G={conv_list[0]['G']},"
          f" G={conv_list[1]['G']}) ===")
    g1, g2 = conv_list[0]["G"], conv_list[1]["G"]
    r2_list = [conv_list[0]["1-R^2"], conv_list[1]["1-R^2"]]
    sl_list = [conv_list[0]["slope"], conv_list[1]["slope"]]
    for r in results:
        if r["converged"] and r["G"] != g1 and r["G"] != g2:
            t = r["G"]
            pr2 = richardson_pred([g1, g2], r2_list, t)
            psl = richardson_pred([g1, g2], sl_list, t)
            print(f"  G={t:>2}: 1-R² actual={r['1-R^2']:.4e}, "
                  f"pred 1/G={pr2['1/G^1']:.4e}, 1/G²={pr2['1/G^2']:.4e}; "
                  f"slope actual={r['slope']:.4f}, "
                  f"pred 1/G={psl['1/G^1']:.4f}, 1/G²={psl['1/G^2']:.4f}")

with open("results/full_ree/posterior_v3_G_fine_ladder.json", "w") as f:
    json.dump({"results": results, "params": {
        "tau": TAU, "gamma": GAMMA, "umax": UMAX, "f_tol": F_TOL,
    }}, f, indent=2)
print("\nSaved: results/full_ree/posterior_v3_G_fine_ladder.json")
