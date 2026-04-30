"""Full ladder: gap-reparam-u + PAVA-u + PAVA-p hybrid NK at strict tol.

Phase 1: G-ladder at γ=0.5, τ=2 (G=10..30 in steps of 2)
Phase 2: γ-ladder at G_best, τ=2
Phase 3: τ-ladder at G_best, γ=0.5

Each step warms from previous successful checkpoint via 2D interpolation.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)
from gap_reparam import (
    encode, decode, pack, unpack, pava_p_only, pava_u_only,
    F_residual_gap, progress_callback,
)

UMAX = 4.0
F_TOL = 1e-9


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


def solve_hybrid(G, tau, gamma, mu_warm=None, u_warm=None, p_warm=None,
                  picard_polish=100, nk_iters=400):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    if mu_warm is None:
        mu0 = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu0[i, :] = Lam(tau * u)
    else:
        mu0 = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)

    # Picard-PAVA warm-up to get into the basin
    mu0 = pava_u_only(pava_p_only(mu0))
    if picard_polish > 0:
        mu = mu0.copy()
        for it in range(picard_polish):
            cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
            cand = pava_u_only(pava_p_only(cand))
            mu = 0.05 * cand + 0.95 * mu
            mu = np.clip(mu, EPS, 1 - EPS)
        mu0 = mu

    # NK with hybrid F
    base0, c0 = encode(mu0)
    x0 = pack(base0, c0)
    t0 = time.time()
    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual_gap(x, G, G, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma, pava_p=True,
                                       pava_u_pre_encode=True),
            x0, f_tol=F_TOL, maxiter=nk_iters,
            method="lgmres", verbose=False,
        )
        x_final = sol
    except NoConvergence as e:
        nk_status = "noconv"
        x_final = e.args[0] if e.args else x0
    except (ValueError, RuntimeError) as exc:
        nk_status = f"err:{type(exc).__name__}"
        x_final = x0

    base_f, c_f = unpack(x_final, G, G)
    mu_final = decode(base_f, c_f)

    cand, active, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F_arr = np.abs(cand - mu_final)
    res_max = float(F_arr[active].max()) if active.any() else float("nan")
    res_med = float(np.median(F_arr[active])) if active.any() else float("nan")
    n_u = int((np.diff(mu_final, axis=0) < 0).sum())
    n_p = int((np.diff(mu_final, axis=1) < 0).sum())
    elapsed = time.time() - t0
    r2, slope, _ = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return {
        "G": G, "tau": tau, "gamma": gamma,
        "nk_status": nk_status, "phi_resid_max": res_max,
        "phi_resid_med": res_med,
        "viol_u": n_u, "viol_p": n_p,
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "converged": nk_status == "ok",
    }, mu_final, u_grid, p_grid, p_lo, p_hi


warnings.filterwarnings("ignore", category=RuntimeWarning)

all_results = {"G_ladder": [], "gamma_ladder": [], "tau_ladder": []}

# ======================================================================
# Phase 0: VERIFY at G=14 that hybrid matches PAVA-Cesaro
# ======================================================================
print(f"=== Phase 0: verification at G=14 ===\n", flush=True)
ck = np.load("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
r, mu, ug, pg, plo, phi = solve_hybrid(
    14, 2.0, 0.5, mu_warm=ck["mu"], u_warm=ck["u_grid"], p_warm=ck["p_grid"],
    picard_polish=200, nk_iters=300,
)
print(f"  G=14 verify: NK={r['nk_status']}  max={r['phi_resid_max']:.3e}  "
      f"med={r['phi_resid_med']:.3e}  u-viol={r['viol_u']}, p-viol={r['viol_p']}  "
      f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  t={r['elapsed_s']:.1f}s",
      flush=True)
print(f"  PAVA-Cesaro reference: 1-R²=0.0967, slope=0.337", flush=True)

mu_seed = mu; u_seed = ug; p_seed = pg
np.savez(f"results/full_ree/posterior_v3_G14_hybrid_verify.npz",
         mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)

# ======================================================================
# Phase 1: G-ladder at γ=0.5, τ=2
# ======================================================================
print(f"\n=== Phase 1: G-ladder at γ=0.5, τ=2 ===\n", flush=True)
G_LADDER = [10, 12, 14, 16, 18, 20, 24, 30]
mu_warm = mu_seed; u_warm = u_seed; p_warm = p_seed
for G in G_LADDER:
    print(f"--- G = {G} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_hybrid(
        G, 2.0, 0.5, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
        picard_polish=150, nk_iters=400,
    )
    print(f"  NK={r['nk_status']}  max={r['phi_resid_max']:.3e}  "
          f"med={r['phi_resid_med']:.3e}  u-viol={r['viol_u']}, p-viol={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)
    all_results["G_ladder"].append(r)
    if r["converged"]:
        np.savez(f"results/full_ree/posterior_v3_hybrid_G{G}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# Pick best G for next ladders (highest converged G with reasonable 1-R²)
G_best_results = [r for r in all_results["G_ladder"]
                  if r["converged"] and 0.05 < r["1-R^2"] < 0.5]
G_BEST = max(G_best_results, key=lambda r: r["G"])["G"] if G_best_results else 14
print(f"\n=== G_BEST = {G_BEST} ===\n", flush=True)

# Load G_BEST checkpoint as seed for next ladders
ck_best = np.load(f"results/full_ree/posterior_v3_hybrid_G{G_BEST}.npz")

# ======================================================================
# Phase 2: γ-ladder at G_BEST, τ=2
# ======================================================================
print(f"\n=== Phase 2: γ-ladder at G={G_BEST}, τ=2 ===\n", flush=True)
GAMMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for gamma in GAMMAS:
    print(f"--- γ = {gamma} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_hybrid(
        G_BEST, 2.0, gamma, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
        picard_polish=150, nk_iters=400,
    )
    print(f"  NK={r['nk_status']}  max={r['phi_resid_max']:.3e}  "
          f"med={r['phi_resid_med']:.3e}  u-viol={r['viol_u']}, p-viol={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)
    all_results["gamma_ladder"].append(r)
    if r["converged"]:
        np.savez(f"results/full_ree/posterior_v3_hybrid_G{G_BEST}_gamma{gamma:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ======================================================================
# Phase 3: τ-ladder at G_BEST, γ=0.5
# ======================================================================
print(f"\n=== Phase 3: τ-ladder at G={G_BEST}, γ=0.5 ===\n", flush=True)
TAUS = [0.5, 1.0, 2.0, 4.0, 8.0]
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for tau in TAUS:
    print(f"--- τ = {tau} ---", flush=True)
    r, mu, ug, pg, plo, phi = solve_hybrid(
        G_BEST, tau, 0.5, mu_warm=mu_warm, u_warm=u_warm, p_warm=p_warm,
        picard_polish=150, nk_iters=400,
    )
    print(f"  NK={r['nk_status']}  max={r['phi_resid_max']:.3e}  "
          f"med={r['phi_resid_med']:.3e}  u-viol={r['viol_u']}, p-viol={r['viol_p']}  "
          f"1-R²={r['1-R^2']:.4e}  slope={r['slope']:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)
    all_results["tau_ladder"].append(r)
    if r["converged"]:
        np.savez(f"results/full_ree/posterior_v3_hybrid_G{G_BEST}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_full_ladder.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ======================================================================
# Final summary
# ======================================================================
print(f"\n=== FINAL SUMMARY ===\n", flush=True)
for ph_name, ph in [("G ladder", "G_ladder"), ("γ ladder", "gamma_ladder"),
                    ("τ ladder", "tau_ladder")]:
    print(f"-- {ph_name} --")
    print(f"  {'param':>6} {'NK':>8} {'max':>10} {'med':>10} "
          f"{'1-R²':>11} {'slope':>7}")
    for r in all_results[ph]:
        param = r["G"] if ph == "G_ladder" else (
            r["gamma"] if ph == "gamma_ladder" else r["tau"])
        print(f"  {param:>6.2f} {r['nk_status']:>8} {r['phi_resid_max']:>10.2e} "
              f"{r['phi_resid_med']:>10.2e} {r['1-R^2']:>11.4e} "
              f"{r['slope']:>7.4f}")
    print()
print("Saved: results/full_ree/posterior_v3_full_ladder.json")
