"""P1 Task 6: knife-edge figure data at full REE.

Sweep τ ∈ [0.1, 10] log-spaced (20 points) for γ = 0.25, 1.0, 4.0.
Each (γ, τ): solve to strict 1e-14 + monotone, measure 1-R² and slope.

Warm-start: continue along τ-line for each γ.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 14
TOL_MAX = 1e-14


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        u_clamped = np.clip(u_target, u_old[0], u_old[-1])
        r_above = np.searchsorted(u_old, u_clamped)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        w = ((u_clamped - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
             if r_above != r_below else 1.0)
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_b, p_old[r_below, :], mu_old[r_below, :])
            p_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                      n_iter, n_avg, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n_iter):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n_iter - n_avg:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def F_phi_residual(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def nk_polish(mu_warm, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    shape = mu_warm.shape
    try:
        sol = newton_krylov(
            lambda x: F_phi_residual(x, shape, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma),
            mu_warm.ravel(), f_tol=TOL_MAX, maxiter=300,
            method="lgmres", verbose=False)
        return np.clip(sol.reshape(shape), EPS, 1 - EPS), "ok"
    except NoConvergence as e:
        mu_nk = (np.clip(e.args[0].reshape(shape), EPS, 1 - EPS)
                  if e.args else mu_warm)
        return mu_nk, "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        return mu_warm, f"err:{type(exc).__name__}"


def strict_solve_fast(tau, gamma, mu_warm, u_warm, p_warm, label=""):
    """Fast strict solve: 2 picard rounds + NK. Returns rec dict + new μ."""
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    if mu_warm is None:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(tau * u)
    else:
        mu = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)

    t0 = time.time()
    mu = picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                            5000, 2500, 0.05)
    mu = picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                            5000, 2500, 0.01)
    mu_picard = mu
    mu_nk, nk_status = nk_polish(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    elapsed = time.time() - t0
    if (d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0):
        status = "strict_conv"; mu_use = mu_nk
    elif d_nk["u_viol"] > 0 or d_nk["p_viol"] > 0:
        status = "fallback_picard"; mu_use = mu_picard
    else:
        status = "no_strict"; mu_use = mu_nk
    d = measure(mu_use, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    r2, slope, _ = measure_R2(mu_use, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    print(f"  [{label}] τ={tau:>5.2g} γ={gamma:>5.2g}: NK={nk_status}, "
          f"max={d['max']:.2e}, viol u/p={d['u_viol']}/{d['p_viol']}, "
          f"1-R²={r2:.4e}, slope={slope:.4f}, status={status}, "
          f"t={elapsed:.0f}s", flush=True)
    return {
        "tau": tau, "gamma": gamma, "G": G,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status,
    }, mu_use, u_grid, p_grid, p_lo, p_hi


warnings.filterwarnings("ignore", category=RuntimeWarning)
print("=== P1 Task 6: knife-edge REE sweep ===\n", flush=True)
TAUS = np.exp(np.linspace(np.log(0.2), np.log(8.0), 16))
GAMMAS = [0.25, 1.0, 4.0]

results = {"tau_grid": list(TAUS), "gammas": GAMMAS, "data": {}}
for gamma in GAMMAS:
    print(f"\n--- γ = {gamma} ---", flush=True)
    # Warm-start from existing strict-converged checkpoint at τ=2 if available
    ck_path = f"results/full_ree/posterior_v3_strict_G14_gamma{gamma:g}.npz"
    try:
        ck = np.load(ck_path)
        mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
        print(f"  Seeded from existing checkpoint at γ={gamma}", flush=True)
    except FileNotFoundError:
        # Fall back to nearest available γ
        ck = np.load("results/full_ree/posterior_v3_strict_G14_gamma0.5.npz")
        mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
        print(f"  No γ={gamma} ckpt; seeded from γ=0.5", flush=True)
    results["data"][str(gamma)] = []

    # Walk τ-line: start at τ=2 (if existing) and split: down then up
    # Actually simpler: just walk all τs in order, warming each from previous
    for tau in TAUS:
        rec, mu, ug, pg, plo, phi_ = strict_solve_fast(
            tau, gamma, mu_warm, u_warm, p_warm,
            label=f"γ={gamma}")
        results["data"][str(gamma)].append(rec)
        # Update warm if good
        if rec["status"] in ("strict_conv", "fallback_picard"):
            mu_warm = mu; u_warm = ug; p_warm = pg
        with open("results/full_ree/fig_knife_edge_data.json", "w") as f:
            json.dump(results, f, indent=2)

print("\n=== knife-edge sweep done ===")
print("Saved: results/full_ree/fig_knife_edge_data.json")
