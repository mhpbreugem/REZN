"""Two-axis ladder: first widen u-grid (UMAX 4→5→6→7), then increase G.

Phase A — UMAX expansion at fixed G:
  (G=14, UMAX=4)  preexisting checkpoint
  (G=14, UMAX=5)  warm from previous
  (G=14, UMAX=6)  warm from previous
  (G=14, UMAX=7)  warm from previous

Phase B — G expansion at UMAX=7:
  (G=18, UMAX=7), (G=22, UMAX=7), (G=26, UMAX=7), (G=30, UMAX=7)
  each warmed from previous.

Strict NK tolerance 1e-14. No cold-starts (cold finds spurious eqs at high G).
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5
F_TOL = 1e-14

# (G, UMAX) ladder — finer UMAX steps so warm-start is closer
PHASE_A = [(14, 4.0), (14, 4.5), (14, 5.0), (14, 5.5), (14, 6.0), (14, 7.0)]
PHASE_B = [(18, 7.0), (22, 7.0), (26, 7.0), (30, 7.0)]
LADDER = PHASE_A + PHASE_B


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
        # Clamp to old u-range so wider new grid uses edge values
        u_clamped = np.clip(u_target, u_old[0], u_old[-1])
        r_above = np.searchsorted(u_old, u_clamped)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        if r_above == r_below:
            w = 1.0
        else:
            w = (u_clamped - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_clamped_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_clamped_b, p_old[r_below, :], mu_old[r_below, :])
            p_clamped_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_clamped_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def solve_GU(G, UMAX, mu_warm, u_warm, p_warm,
             picard_iters=300, nk_iters=800):
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
            f_tol=F_TOL, maxiter=nk_iters, verbose=False, method="lgmres",
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
    residual_med = float(np.median(F_arr[active_mask]))
    strict_conv = residual_max < 1e-12
    fit_t = time.time() - t0
    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    return {
        "G": G, "UMAX": UMAX,
        "picard_resid": picard_resid, "nk_status": nk_status,
        "residual_max": residual_max, "residual_median": residual_med,
        "strict_conv": bool(strict_conv),
        "1-R^2": float(r2def), "slope": float(slope),
        "n_samples": int(n), "elapsed_s": fit_t,
        "active_cells": int(active_mask.sum()),
    }, mu_final, u_grid, p_grid, p_lo, p_hi


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== ladder UMAX→G, γ={GAMMA}, τ={TAU}, f_tol={F_TOL} ===\n", flush=True)

# Seed phase A from G=14, UMAX=4 checkpoint
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu_warm = ckpt["mu"]; u_warm = ckpt["u_grid"]; p_warm = ckpt["p_grid"]
print(f"Seed: G=14, UMAX=4 checkpoint loaded", flush=True)

# Verify seed re-solves
r0, mu0_, u0, p0, p_lo0, p_hi0 = solve_GU(14, 4.0, mu_warm, u_warm, p_warm)
print(f"Seed verify: max={r0['residual_max']:.2e}, "
      f"1-R²={r0['1-R^2']:.4e}, slope={r0['slope']:.4f}, "
      f"strict={r0['strict_conv']}", flush=True)
results = [r0]

mu_prev, u_prev, p_prev = mu0_, u0, p0
for G, UMAX in LADDER:
    print(f"\n--- (G={G}, UMAX={UMAX}) ---", flush=True)
    r, mu, ug, pg, p_lo, p_hi = solve_GU(G, UMAX, mu_prev, u_prev, p_prev,
                                          picard_iters=300, nk_iters=800)
    print(f"  picard_resid={r['picard_resid']:.2e}  NK={r['nk_status']}  "
          f"max={r['residual_max']:.2e}  med={r['residual_median']:.2e}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  "
          f"strict={r['strict_conv']}  t={r['elapsed_s']:.1f}s",
          flush=True)
    results.append(r)
    if r["strict_conv"]:
        # save checkpoint
        np.savez(f"results/full_ree/posterior_v3_G{G}_U{UMAX:g}_mu.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=p_lo, p_hi=p_hi,
                 tau=TAU, gamma=GAMMA, umax=UMAX)
        mu_prev, u_prev, p_prev = mu, ug, pg
    else:
        # Don't update warm: skip and use prev for next step
        # but if median is very small we might still want to use it
        if r["residual_median"] < 1e-10:
            print(f"  [accepting partial as warm-start (median<1e-10)]",
                  flush=True)
            mu_prev, u_prev, p_prev = mu, ug, pg
    # incremental save
    with open("results/full_ree/posterior_v3_umax_G_ladder.json", "w") as f:
        json.dump({"results": results, "params": {
            "tau": TAU, "gamma": GAMMA, "f_tol": F_TOL,
        }}, f, indent=2)

print("\n=== SUMMARY ===")
print(f"{'G':>3} {'UMAX':>5} {'NK':>8} {'max':>10} {'med':>10} "
      f"{'1-R²':>11} {'slope':>7} {'strict':>7}")
for r in results:
    print(f"{r['G']:>3d} {r['UMAX']:>5.1f} {r['nk_status']:>8} "
          f"{r['residual_max']:>10.2e} {r['residual_median']:>10.2e} "
          f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f} "
          f"{str(r['strict_conv']):>7}")
print("\nSaved: results/full_ree/posterior_v3_umax_G_ladder.json")
