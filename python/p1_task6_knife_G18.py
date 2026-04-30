"""Knife-edge τ sweep at G=18 — relaxed acceptance criterion.

Each cell does Picard-PAVA-Cesaro then NK polish. Accept the result if:
  - monotone (zero violations)
  - max < 1e-3 OR med < 1e-6
This is a grid-noise-floor criterion (~0.001 in residual at G=18 floor).

NK is applied but result kept only if monotone. Else fallback to Picard.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 18


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
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


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, n, na, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
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


def solve_relaxed(tau, gamma, mu_warm, u_warm, p_warm):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid) if mu_warm is not None else None
    if mu is None:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(tau * u)
    mu = pava_2d(mu)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, 5000, 2500, 0.05)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, 5000, 2500, 0.01)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, 10000, 5000, 0.003)
    mu_picard = mu
    # NK polish
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, tau, gamma),
            mu.ravel(), f_tol=1e-12, maxiter=200,
            method="lgmres", verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu
        nk_status = f"err:{type(exc).__name__}"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    # Use NK only if monotone
    use_nk = (d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0)
    mu_use = mu_nk if use_nk else mu_picard
    d = measure(mu_use, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    elapsed = time.time() - t0
    r2, slope, _ = measure_R2(mu_use, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    accept = (d["u_viol"] == 0 and d["p_viol"] == 0
              and (d["max"] < 1e-3 or d["med"] < 1e-6))
    return {
        "tau": tau, "gamma": gamma, "G": G,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "use_nk": use_nk, "nk_status": nk_status,
        "accept": accept,
    }, mu_use, u_grid, p_grid, p_lo, p_hi


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Knife-edge sweep at G={G}, relaxed (monotone + small resid) ===\n",
      flush=True)
TAUS = np.exp(np.linspace(np.log(0.2), np.log(8.0), 16))
GAMMAS = [0.25, 1.0, 4.0]
results = {"tau_grid": list(TAUS), "gammas": GAMMAS, "G": G, "data": {}}

for gamma in GAMMAS:
    print(f"\n--- γ = {gamma} ---", flush=True)
    # Try to seed from existing G=14 strict at γ if available
    seed_path = f"results/full_ree/posterior_v3_strict_G14_gamma{gamma:g}.npz"
    try:
        ck = np.load(seed_path)
        mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
        print(f"  Seeded from G=14 strict γ={gamma}", flush=True)
    except FileNotFoundError:
        ck = np.load("results/full_ree/posterior_v3_strict_G14_gamma0.5.npz")
        mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
        print(f"  Seeded from G=14 γ=0.5 (no γ={gamma} ckpt)", flush=True)
    results["data"][str(gamma)] = []

    for tau in TAUS:
        rec, mu, ug, pg, plo, phi_ = solve_relaxed(tau, gamma, mu_warm,
                                                     u_warm, p_warm)
        print(f"  τ={tau:>5.2g}: NK={rec['nk_status']:>12} "
              f"max={rec['max']:.2e} med={rec['med']:.2e} "
              f"u/p={rec['u_viol']}/{rec['p_viol']} "
              f"1-R²={rec['1-R^2']:.4e} slope={rec['slope']:.4f} "
              f"accept={rec['accept']} t={rec['elapsed_s']:.0f}s",
              flush=True)
        results["data"][str(gamma)].append(rec)
        if rec["accept"]:
            mu_warm = mu; u_warm = ug; p_warm = pg
        with open(f"results/full_ree/fig_knife_edge_G{G}_data.json", "w") as f:
            json.dump(results, f, indent=2)

print(f"\n=== Knife-edge G={G} done ===")
print(f"Saved: results/full_ree/fig_knife_edge_G{G}_data.json")
