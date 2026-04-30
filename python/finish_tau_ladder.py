"""Finish the τ-ladder for τ=4, 8 after the τ-cap fix."""
import json, time, warnings
import numpy as np

from posterior_method_v3 import Lam, init_p_grid, phi_step, measure_R2, EPS
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
N_TOTAL = 800
N_AVG = 400


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


def solve(G, tau, gamma, mu_warm, u_warm, p_warm):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    mu = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    mu_sum = np.zeros_like(mu)
    n_collected = 0
    for it in range(N_TOTAL):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = 0.05 * cand + 0.95 * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= N_TOTAL - N_AVG:
            mu_sum += mu
            n_collected += 1
    mu_avg = pava_2d(mu_sum / n_collected)
    cand, active, _ = phi_step(mu_avg, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F_arr = np.abs(cand - mu_avg)
    res_max = float(F_arr[active].max())
    res_med = float(np.median(F_arr[active]))
    n_u = int((np.diff(mu_avg, axis=0) < 0).sum())
    n_p = int((np.diff(mu_avg, axis=1) < 0).sum())
    r2, slope, _ = measure_R2(mu_avg, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return mu_avg, u_grid, p_grid, p_lo, p_hi, res_max, res_med, n_u, n_p, r2, slope


warnings.filterwarnings("ignore", category=RuntimeWarning)
G = 14

# Load existing JSON to append
with open("results/full_ree/posterior_v3_pava_full_ladder.json") as f:
    all_results = json.load(f)
print(f"Existing tau_ladder: {len(all_results['tau_ladder'])} entries")

# Find best converged warm-start (largest τ from results)
ck = np.load(f"results/full_ree/posterior_v3_pava_G{G}_tau2.npz")
mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]

for tau in [4.0, 8.0]:
    print(f"--- τ = {tau} ---", flush=True)
    t0 = time.time()
    mu, ug, pg, plo, phi, mx, md, nu, np_, r2, slope = solve(
        G, tau, 0.5, mu_warm, u_warm, p_warm)
    elapsed = time.time() - t0
    print(f"  max={mx:.3e}  med={md:.3e}  u={nu}, p={np_}  "
          f"1-R²={r2:.4e}  slope={slope:.4f}  t={elapsed:.1f}s", flush=True)
    good = md < 1e-3 and nu == 0 and np_ == 0
    rec = {
        "G": G, "tau": tau, "gamma": 0.5,
        "phi_resid_max": mx, "phi_resid_med": md,
        "phi_resid_l2": float(np.sqrt(np.mean((mx)**2))),
        "viol_u": nu, "viol_p": np_,
        "1-R^2": r2, "slope": slope,
        "elapsed_s": elapsed, "good": good,
    }
    all_results["tau_ladder"].append(rec)
    if good:
        np.savez(f"results/full_ree/posterior_v3_pava_G{G}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_pava_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

print("\n=== FINAL τ ladder ===")
for r in all_results["tau_ladder"]:
    print(f"  τ={r['tau']:>6.2f}  1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  "
          f"med={r['phi_resid_med']:.2e}  good={r['good']}")
