"""Test P0.6: trim p-grid to 95% feasible range, then run strict at G=16,18,20.

If trim works:
  - G=16+ achieves max < 1e-14 with both monotonicities
  - 1-R² unchanged from G=14 (which itself is unchanged from untrimmed)

Then we can run knife-edge sweep at higher G with strict tolerance.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
TAU = 2.0
GAMMA = 0.5
TOL_MAX = 1e-14
TRIM = 0.025  # 2.5% from each side → keep central 95%


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    out = np.empty((G_new, p_new.shape[1]))
    for i in range(G_new):
        u = u_new[i]
        u_c = np.clip(u, u_old[0], u_old[-1])
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


def strict_solve_trim(G, mu_warm, u_warm, p_warm, label=""):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G, trim=TRIM)
    if mu_warm is not None:
        mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    else:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(TAU * u)
    mu = pava_2d(mu)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.05)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.01)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 10000, 5000, 0.003)
    mu_picard = mu
    d_p = measure(mu_picard, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  [{label}] picard: max={d_p['max']:.3e}, med={d_p['med']:.3e}, "
          f"u/p={d_p['u_viol']}/{d_p['p_viol']}, t={time.time()-t0:.0f}s",
          flush=True)
    if d_p["max"] < TOL_MAX and d_p["u_viol"] == 0 and d_p["p_viol"] == 0:
        return mu_picard, d_p, u_grid, p_grid, p_lo, p_hi, "strict_picard", time.time()-t0
    print(f"  [{label}] NK polish...", flush=True)
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA),
            mu.ravel(), f_tol=TOL_MAX, maxiter=300,
            method="lgmres", verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu_picard
        nk_status = f"err:{type(exc).__name__}"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  [{label}] post-NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
          f"u/p={d_nk['u_viol']}/{d_nk['p_viol']}, NK={nk_status}",
          flush=True)
    if (d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0
        and d_nk["p_viol"] == 0):
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_NK", time.time()-t0
    if d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "non_strict_NK_monotone", time.time()-t0
    return mu_picard, d_p, u_grid, p_grid, p_lo, p_hi, "fallback_picard", time.time()-t0


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Test trim p-grid (95% domain) at G=14, 16, 18, 20 ===\n",
      flush=True)
results = []

# Verify G=14 still works with trim (sanity)
print("--- G = 14 (with trim) ---", flush=True)
mu, d, ug, pg, plo, phi_, status, t = strict_solve_trim(
    14, None, None, None, label="G=14 trim")
r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, TAU, GAMMA)
print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}", flush=True)
results.append({"G": 14, **d, "1-R^2": float(r2), "slope": float(slope),
                "status": status, "elapsed_s": t})
mu_warm, u_warm, p_warm = mu, ug, pg

# G=16, 18, 20 with trim
for G in [16, 18, 20]:
    print(f"\n--- G = {G} (with trim) ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, t = strict_solve_trim(
        G, mu_warm, u_warm, p_warm, label=f"G={G}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, TAU, GAMMA)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    results.append({"G": G, **d, "1-R^2": float(r2), "slope": float(slope),
                    "status": status, "elapsed_s": t})
    if status.startswith("strict") or status == "non_strict_NK_monotone":
        np.savez(f"results/full_ree/posterior_v3_trim_G{G}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg

print(f"\n=== TRIM RESULTS ===")
print(f"{'G':>3} {'max':>10} {'med':>10} {'u/p':>5} {'1-R²':>11} "
      f"{'slope':>7} {'status':>22}")
for r in results:
    print(f"{r['G']:>3d} {r['max']:>10.2e} {r['med']:>10.2e} "
          f"{r['u_viol']}/{r['p_viol']} {r['1-R^2']:>11.4e} "
          f"{r['slope']:>7.4f} {r['status']:>22}")
with open("results/full_ree/trim_pgrid_test.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/full_ree/trim_pgrid_test.json")
