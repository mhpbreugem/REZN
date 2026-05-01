"""Fig 4A: REE 1-R² vs τ at γ ∈ {0.25, 1, 4}, G=15.

Adaptive bisection: if a step fails strict, try midpoint to last good.
Keep bisecting until success or step too small.
"""
import json, time, warnings, os
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; G = 15; TOL_MAX = 1e-13; TRIM = 0.05
RESULTS_DIR = "results/full_ree"

GAMMAS = [0.25, 1.0, 4.0]
TAU_GRID = list(np.exp(np.linspace(np.log(0.2), np.log(8.0), 60)))
MIN_STEP_LOG = 0.005   # bisect until log-spacing < this


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    out = np.empty((G_new, p_new.shape[1]))
    for i in range(G_new):
        u = u_new[i]; u_c = np.clip(u, u_old[0], u_old[-1])
        ra = np.searchsorted(u_old, u_c)
        rb = max(ra - 1, 0); ra = min(ra, len(u_old) - 1)
        w = (u_c - u_old[rb]) / (u_old[ra] - u_old[rb]) if ra != rb else 1.0
        for j in range(p_new.shape[1]):
            p = p_new[i, j]
            p_c1 = np.clip(p, p_old[rb, 0], p_old[rb, -1])
            mb = np.interp(p_c1, p_old[rb, :], mu_old[rb, :])
            p_c2 = np.clip(p, p_old[ra, 0], p_old[ra, -1])
            ma = np.interp(p_c2, p_old[ra, :], mu_old[ra, :])
            out[i, j] = (1 - w) * mb + w * ma
    return np.clip(out, EPS, 1 - EPS)


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, n, na, alpha, tau, gamma):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi(x, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def solve_one(tau, gamma, mu_warm, u_warm, p_warm):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G, trim=TRIM)
    mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 3000, 1500, 0.05, tau, gamma)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 3000, 1500, 0.01, tau, gamma)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.003, tau, gamma)
    d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    if d["max"] < TOL_MAX and d["u_viol"] == 0 and d["p_viol"] == 0:
        return mu, d, u_grid, p_grid, p_lo, p_hi, "strict_picard", time.time()-t0
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, tau, gamma),
            mu.ravel(), f_tol=TOL_MAX, maxiter=200, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu; nk_status = "err"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    if d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_NK", time.time()-t0
    if d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "non_strict_monotone", time.time()-t0
    return mu, d, u_grid, p_grid, p_lo, p_hi, "fallback_picard", time.time()-t0


def solve_with_bisection(tau_target, gamma, last_good_tau, mu_w, u_w, p_w,
                         label=""):
    """Try τ_target. If fails, bisect toward last_good_tau in log space.
    Update warm-start as we go. Return final result for τ_target."""
    # First try direct
    mu, d, ug, pg, plo, phi_, status, t = solve_one(tau_target, gamma,
                                                       mu_w, u_w, p_w)
    print(f"    [{label}] direct: max={d['max']:.2e}, status={status}",
          flush=True)
    if status.startswith("strict"):
        return mu, d, ug, pg, plo, phi_, status

    # Bisect toward last_good_tau
    lo_log = np.log(min(tau_target, last_good_tau))
    hi_log = np.log(max(tau_target, last_good_tau))
    cur_warm_mu = mu_w; cur_warm_u = u_w; cur_warm_p = p_w
    cur_warm_tau = last_good_tau
    n_bisect = 0
    while abs(np.log(tau_target) - np.log(cur_warm_tau)) > MIN_STEP_LOG \
          and n_bisect < 6:
        # Midpoint between cur_warm_tau and tau_target
        tau_mid = float(np.exp(0.5 * (np.log(cur_warm_tau)
                                        + np.log(tau_target))))
        n_bisect += 1
        mu_m, d_m, ug_m, pg_m, plo_m, phi_m, status_m, t_m = solve_one(
            tau_mid, gamma, cur_warm_mu, cur_warm_u, cur_warm_p)
        print(f"    [{label}] bisect#{n_bisect} τ={tau_mid:.3g}: "
              f"max={d_m['max']:.2e}, status={status_m}",
              flush=True)
        if status_m.startswith("strict"):
            cur_warm_mu = mu_m; cur_warm_u = ug_m; cur_warm_p = pg_m
            cur_warm_tau = tau_mid
        else:
            # midpoint failed too — give up bisection; use whatever we have
            # but still use the closer warm start
            return mu_m, d_m, ug_m, pg_m, plo_m, phi_m, status_m
    # Now retry tau_target with the newly-found closer warm
    mu, d, ug, pg, plo, phi_, status, t = solve_one(
        tau_target, gamma, cur_warm_mu, cur_warm_u, cur_warm_p)
    print(f"    [{label}] retry direct: max={d['max']:.2e}, status={status}",
          flush=True)
    return mu, d, ug, pg, plo, phi_, status


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Fig 4A: REE τ-sweep with adaptive bisection ===\n", flush=True)
all_curves = {}

for gamma in GAMMAS:
    print(f"\n--- γ = {gamma} ---", flush=True)
    ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15_gamma{gamma:g}.npz"
    if not os.path.exists(ck_path) and abs(gamma - 0.5) < 1e-6:
        ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15.npz"
    if not os.path.exists(ck_path):
        ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15.npz"
    ck = np.load(ck_path)
    print(f"  Seeded from {ck_path}", flush=True)

    # Sweep down then sweep up from anchor (τ closest to 2)
    mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
    points = []
    last_good_tau = 2.0  # paper baseline

    # Sort taus into "down" (< 2) descending and "up" (>= 2) ascending
    taus_down = sorted([t for t in TAU_GRID if t < 2.0], reverse=True)
    taus_up = sorted([t for t in TAU_GRID if t >= 2.0])

    # First try anchor at τ=2 (in case ckpt has different τ)
    for direction, tau_seq in [("down", taus_down), ("up", taus_up)]:
        # Restart from anchor
        cur_warm_mu = ck["mu"]; cur_warm_u = ck["u_grid"]
        cur_warm_p = ck["p_grid"]
        cur_warm_tau = 2.0
        for tau in tau_seq:
            mu, d, ug, pg, plo, phi_, status = solve_with_bisection(
                tau, gamma, cur_warm_tau, cur_warm_mu, cur_warm_u,
                cur_warm_p, label=f"γ={gamma}, τ={tau:.3g}")
            r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, tau, gamma)
            points.append({"tau": float(tau), "1-R2": float(r2),
                            "slope": float(slope), "max": d["max"],
                            "status": status})
            print(f"    => 1-R²={r2:.4e}, slope={slope:.4f}",
                  flush=True)
            if status.startswith("strict"):
                cur_warm_mu = mu; cur_warm_u = ug; cur_warm_p = pg
                cur_warm_tau = tau
            all_curves[f"{gamma:g}"] = {"gamma": gamma, "points": points}
            with open(f"{RESULTS_DIR}/fig_4A_data.json", "w") as f:
                json.dump({"figure": "fig_4A_REE_vs_tau",
                            "params": {"G": G, "gammas": GAMMAS,
                                        "tau_grid": list(TAU_GRID)},
                            "curves": list(all_curves.values())}, f, indent=2)

print("\n=== DONE Fig 4A ===")
