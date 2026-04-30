"""Full ladder via Picard-PAVA-2D + Cesaro averaging — the working method.

Phase 1: G-ladder at γ=0.5, τ=2 (G=10..30 in steps of 2)
Phase 2: γ-ladder at G_best, τ=2
Phase 3: τ-ladder at G_best, γ=0.5

Each step does N_TOTAL Picard-PAVA iterations, averages the last N_AVG
to get the monotone fixed-point estimate, and projects to monotone.
Each new (G, γ, τ) cell warms from the previous via 2D μ interpolation.
"""
import json, time, warnings
import numpy as np

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, project_monotone, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
N_TOTAL = 800   # Picard-PAVA iterations per cell
N_AVG = 400     # Number of trailing iterations to average


def pava_2d(mu):
    """Apply PAVA in p then in u."""
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


def solve_pava_cesaro(G, tau, gamma, mu_warm=None, u_warm=None, p_warm=None,
                       n_total=N_TOTAL, n_avg=N_AVG, damping=0.05):
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
    mu_sum = np.zeros_like(mu)
    n_collected = 0

    for it in range(n_total):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = damping * cand + (1 - damping) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n_total - n_avg:
            mu_sum += mu
            n_collected += 1

    mu_avg = mu_sum / n_collected
    mu_avg = pava_2d(mu_avg)

    # Diagnostics
    cand, active, _ = phi_step(mu_avg, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F_arr = np.abs(cand - mu_avg)
    res_max = float(F_arr[active].max()) if active.any() else float("nan")
    res_med = float(np.median(F_arr[active])) if active.any() else float("nan")
    res_l2 = float(np.sqrt(np.mean(F_arr[active]**2))) if active.any() else float("nan")
    n_u = int((np.diff(mu_avg, axis=0) < 0).sum())
    n_p = int((np.diff(mu_avg, axis=1) < 0).sum())
    elapsed = time.time() - t0
    r2, slope, _ = measure_R2(mu_avg, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return {
        "G": G, "tau": tau, "gamma": gamma,
        "phi_resid_max": res_max, "phi_resid_med": res_med,
        "phi_resid_l2": res_l2,
        "viol_u": n_u, "viol_p": n_p,
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed,
        "good": res_med < 1e-3 and n_u == 0 and n_p == 0,
    }, mu_avg, u_grid, p_grid, p_lo, p_hi


warnings.filterwarnings("ignore", category=RuntimeWarning)
all_results = {"G_ladder": [], "gamma_ladder": [], "tau_ladder": []}

# ===== Phase 0: verify at G=14 =====
print(f"\n=== Phase 0: G=14 PAVA-2D-Cesaro verification ===\n", flush=True)
ck = np.load("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
r, mu, ug, pg, plo, phi = solve_pava_cesaro(
    14, 2.0, 0.5, mu_warm=ck["mu"], u_warm=ck["u_grid"], p_warm=ck["p_grid"],
)
print(f"  G=14: max={r['phi_resid_max']:.3e}  med={r['phi_resid_med']:.3e}  "
      f"u-viol={r['viol_u']}, p-viol={r['viol_p']}  "
      f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  good={r['good']}  "
      f"t={r['elapsed_s']:.1f}s", flush=True)

mu_seed = mu; u_seed = ug; p_seed = pg

# ===== Phase 1: G-ladder =====
print(f"\n=== Phase 1: G-ladder at γ=0.5, τ=2 ===\n", flush=True)
G_LADDER = [10, 12, 14, 16, 18, 20, 24, 30]
mu_warm = mu_seed; u_warm = u_seed; p_warm = p_seed
for G in G_LADDER:
    print(f"--- G = {G} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_pava_cesaro(
        G, 2.0, 0.5, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
    )
    print(f"  max={r['phi_resid_max']:.3e}  med={r['phi_resid_med']:.3e}  "
          f"u={r['viol_u']}, p={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  good={r['good']}  "
          f"t={r['elapsed_s']:.1f}s", flush=True)
    all_results["G_ladder"].append(r)
    if r["good"]:
        np.savez(f"results/full_ree/posterior_v3_pava_G{G}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_pava_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# Pick best G
G_good = [r for r in all_results["G_ladder"] if r["good"]]
G_BEST = max(G_good, key=lambda r: r["G"])["G"] if G_good else 14
print(f"\n=== G_BEST = {G_BEST} ===\n", flush=True)
ck_best = np.load(f"results/full_ree/posterior_v3_pava_G{G_BEST}.npz")

# ===== Phase 2: γ-ladder =====
print(f"\n=== Phase 2: γ-ladder at G={G_BEST}, τ=2 ===\n", flush=True)
GAMMAS = [2.0, 1.0, 0.5, 0.3, 0.1]   # ladder from γ=2 down (start from CARA-like end)
# Reset warm to G_BEST converged at γ=0.5
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for gamma in GAMMAS:
    print(f"--- γ = {gamma} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_pava_cesaro(
        G_BEST, 2.0, gamma, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
    )
    print(f"  max={r['phi_resid_max']:.3e}  med={r['phi_resid_med']:.3e}  "
          f"u={r['viol_u']}, p={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  good={r['good']}  "
          f"t={r['elapsed_s']:.1f}s", flush=True)
    all_results["gamma_ladder"].append(r)
    if r["good"]:
        np.savez(f"results/full_ree/posterior_v3_pava_G{G_BEST}_gamma{gamma:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_pava_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ===== Phase 3: τ-ladder =====
print(f"\n=== Phase 3: τ-ladder at G={G_BEST}, γ=0.5 ===\n", flush=True)
TAUS = [2.0, 1.0, 0.5, 4.0, 8.0]
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for tau in TAUS:
    print(f"--- τ = {tau} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_pava_cesaro(
        G_BEST, tau, 0.5, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
    )
    print(f"  max={r['phi_resid_max']:.3e}  med={r['phi_resid_med']:.3e}  "
          f"u={r['viol_u']}, p={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  good={r['good']}  "
          f"t={r['elapsed_s']:.1f}s", flush=True)
    all_results["tau_ladder"].append(r)
    if r["good"]:
        np.savez(f"results/full_ree/posterior_v3_pava_G{G_BEST}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_pava_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ===== Summary =====
print(f"\n=== FINAL SUMMARY ===\n", flush=True)
for ph_name, ph in [("G ladder", "G_ladder"), ("γ ladder", "gamma_ladder"),
                    ("τ ladder", "tau_ladder")]:
    print(f"-- {ph_name} --")
    print(f"  {'p':>6} {'max':>10} {'med':>10} {'u/p':>5} {'1-R²':>11} "
          f"{'slope':>7} {'good':>5}")
    for r in all_results[ph]:
        param = (r["G"] if ph == "G_ladder"
                 else (r["gamma"] if ph == "gamma_ladder" else r["tau"]))
        print(f"  {param:>6.2f} {r['phi_resid_max']:>10.2e} "
              f"{r['phi_resid_med']:>10.2e} "
              f"{r['viol_u']:>2}/{r['viol_p']:<2} "
              f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f} "
              f"{str(r['good']):>5}")
    print()
print("Saved: results/full_ree/posterior_v3_pava_full_ladder.json")
