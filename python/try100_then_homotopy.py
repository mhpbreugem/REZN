"""For each G in {14, 15, 16, 17, 18, 19} (and high-G as available):
  1. Try 100% coverage (trim=0) directly. If strict: done.
  2. Else: homotopy from 90% to 100% in 1% steps.

Replaces the always-homotopy cutoff_homotopy_ladder.py.
"""
import time, json, warnings, os
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; TAU = 2.0; GAMMA = 0.5; TOL_MAX = 1e-14
GS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
HOMOTOPY_TRIMS = [0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015,
                   0.010, 0.005, 0.0]


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


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, n, na, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi(x, shape, u_grid, p_grid, p_lo, p_hi):
    mu = np.clip(x.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def measure(mu, u_grid, p_grid, p_lo, p_hi):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def solve(G, trim, mu_warm, u_warm, p_warm, label="",
           picard_iters=(3000, 3000, 5000)):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G, trim=trim)
    mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[0], picard_iters[0]//2, 0.05)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[1], picard_iters[1]//2, 0.01)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[2], picard_iters[2]//2, 0.003)
    d_p = measure(mu, u_grid, p_grid, p_lo, p_hi)
    if d_p["max"] < TOL_MAX and d_p["u_viol"] == 0 and d_p["p_viol"] == 0:
        return mu, d_p, u_grid, p_grid, p_lo, p_hi, "strict_picard", time.time()-t0
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi),
            mu.ravel(), f_tol=TOL_MAX, maxiter=200, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu; nk_status = f"err"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi)
    if d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_NK", time.time()-t0
    if d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "non_strict_monotone", time.time()-t0
    return mu, d_p, u_grid, p_grid, p_lo, p_hi, "fallback_picard", time.time()-t0


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Try 100% first; fall back to homotopy ===\n", flush=True)
all_results = {}

for G in GS:
    seed_path = f"results/full_ree/posterior_v3_trim90_G{G}.npz"
    if not os.path.exists(seed_path):
        print(f"\nG={G}: no trim90 seed yet, skipping", flush=True)
        continue
    print(f"\n=== G = {G} ===", flush=True)
    ck = np.load(seed_path)
    mu_w, u_w, p_w = ck["mu"], ck["u_grid"], ck["p_grid"]

    # First: try 100%
    print(f"  trying c=100% directly...", flush=True)
    mu, d, ug, pg, plo, phi_, status, t = solve(
        G, 0.0, mu_w, u_w, p_w, label=f"G={G}, c=100")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, TAU, GAMMA)
    print(f"    c=100%: max={d['max']:.2e}, 1-R²={r2:.4e}, "
          f"slope={slope:.4f}, status={status}", flush=True)
    if status.startswith("strict"):
        np.savez(f"results/full_ree/posterior_v3_G{G}_c100.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        all_results[str(G)] = [{"trim": 0.0, "coverage_pct": 100,
                                  **d, "1-R^2": float(r2),
                                  "slope": float(slope), "status": status,
                                  "t": t}]
        with open("results/full_ree/try100_then_homotopy.json", "w") as f:
            json.dump(all_results, f, indent=2)
        continue

    # Else: homotopy 90→100
    print(f"  c=100% failed; doing homotopy 90→100", flush=True)
    all_results[str(G)] = []
    for trim in HOMOTOPY_TRIMS:
        coverage_pct = 100 * (1 - 2 * trim)
        mu, d, ug, pg, plo, phi_, status, t = solve(
            G, trim, mu_w, u_w, p_w, label=f"G={G},c={coverage_pct:.0f}")
        r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, TAU, GAMMA)
        rec = {"trim": trim, "coverage_pct": coverage_pct, **d,
               "1-R^2": float(r2), "slope": float(slope),
               "status": status, "t": t}
        all_results[str(G)].append(rec)
        print(f"    c={coverage_pct:>3.0f}%: max={d['max']:.2e}, "
              f"1-R²={r2:.4e}, slope={slope:.4f}, {status}",
              flush=True)
        if status.startswith("strict") or status == "non_strict_monotone":
            np.savez(f"results/full_ree/posterior_v3_G{G}_c{coverage_pct:.0f}.npz",
                     mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
            mu_w = mu; u_w = ug; p_w = pg
        else:
            print(f"    failed at c={coverage_pct:.0f}%; stopping", flush=True)
            break
        with open("results/full_ree/try100_then_homotopy.json", "w") as f:
            json.dump(all_results, f, indent=2)

print(f"\n=== DONE ===")
print(f"\n{'G':>3} {'cov':>4} {'max':>9} {'1-R²':>10} {'slope':>7} status")
for G_str, recs in all_results.items():
    for r in recs:
        print(f"{G_str:>3} {r['coverage_pct']:>4.0f} {r['max']:>9.2e} "
              f"{r['1-R^2']:>10.4e} {r['slope']:>7.4f} {r['status']}")
