"""Repair G=12 trim90 (was non-strict max=1.2e-3).

Try multiple strategies:
1. Cold start (no warm)
2. Warm from G=13 (trim90 strict)
3. Warm from G=11 + longer picard
4. Coverage homotopy (start at 92%, widen to 90%)
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; G = 12; TAU = 2.0; GAMMA = 0.5; TOL_MAX = 1e-14


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


def solve(mu_init, u_grid, p_grid, p_lo, p_hi, picard_iters=(5000, 5000, 10000)):
    mu = pava_2d(mu_init)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[0], picard_iters[0]//2, 0.05)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[1], picard_iters[1]//2, 0.01)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                       picard_iters[2], picard_iters[2]//2, 0.003)
    d_p = measure(mu, u_grid, p_grid, p_lo, p_hi)
    print(f"  picard: max={d_p['max']:.3e}, med={d_p['med']:.3e}, "
          f"u/p={d_p['u_viol']}/{d_p['p_viol']}, t={time.time()-t0:.0f}s",
          flush=True)
    if d_p["max"] < TOL_MAX and d_p["u_viol"] == 0 and d_p["p_viol"] == 0:
        return mu, d_p, "strict_picard"
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi),
            mu.ravel(), f_tol=TOL_MAX, maxiter=300, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu; nk_status = f"err"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi)
    print(f"  NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
          f"u/p={d_nk['u_viol']}/{d_nk['p_viol']}, NK={nk_status}",
          flush=True)
    if d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, "strict_NK"
    if d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, "non_strict_monotone"
    return mu, d_p, "fallback_picard"


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Repair G={G} trim90 ===\n", flush=True)
TRIM = 0.05  # 90% coverage
u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G, trim=TRIM)
candidates = []

# Strategy 1: cold start
print("--- Strategy 1: cold start ---", flush=True)
mu0 = np.zeros((G, G))
for i, u in enumerate(u_grid):
    mu0[i, :] = Lam(TAU * u)
mu, d, status = solve(mu0, u_grid, p_grid, p_lo, p_hi)
candidates.append(("cold", mu, d, status))

# Strategy 2: warm from G=13 trim90 (smaller G->larger G interp downward)
print("\n--- Strategy 2: warm from G=13 trim90 ---", flush=True)
ck = np.load("results/full_ree/posterior_v3_trim90_G13.npz")
mu0 = interp_mu(ck["mu"], ck["u_grid"], ck["p_grid"], u_grid, p_grid)
mu, d, status = solve(mu0, u_grid, p_grid, p_lo, p_hi)
candidates.append(("warm_G13", mu, d, status))

# Strategy 3: warm from G=11 trim90 with longer picard
print("\n--- Strategy 3: warm from G=11 trim90, longer picard ---",
      flush=True)
ck = np.load("results/full_ree/posterior_v3_trim90_G11.npz")
mu0 = interp_mu(ck["mu"], ck["u_grid"], ck["p_grid"], u_grid, p_grid)
mu, d, status = solve(mu0, u_grid, p_grid, p_lo, p_hi,
                        picard_iters=(8000, 8000, 15000))
candidates.append(("warm_G11_long", mu, d, status))

# Strategy 4: coverage homotopy 92% → 90%
print("\n--- Strategy 4: homotopy 92→90% ---", flush=True)
# First solve at 92%
TRIM_94 = 0.04  # 92% coverage
p_lo94, p_hi94, p_grid94 = init_p_grid(u_grid, TAU, GAMMA, G, trim=TRIM_94)
ck = np.load("results/full_ree/posterior_v3_trim90_G13.npz")
mu0 = interp_mu(ck["mu"], ck["u_grid"], ck["p_grid"], u_grid, p_grid94)
print("  step 1: at 92%", flush=True)
mu_92, d_92, status_92 = solve(mu0, u_grid, p_grid94, p_lo94, p_hi94)
# Then expand to 90%
mu_at90 = interp_mu(mu_92, u_grid, p_grid94, u_grid, p_grid)
print("  step 2: expand to 90%", flush=True)
mu, d, status = solve(mu_at90, u_grid, p_grid, p_lo, p_hi)
candidates.append(("homotopy_92_to_90", mu, d, status))

# Pick best monotone strict result
strict = [c for c in candidates if c[3].startswith("strict")]
print(f"\n=== SUMMARY ===")
print(f"{'strategy':>22} {'status':>22} {'max':>10} {'1-R²':>11} {'slope':>7}")
for name, mu_c, d_c, status_c in candidates:
    r2, slope, _ = measure_R2(mu_c, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"{name:>22} {status_c:>22} {d_c['max']:>10.2e} "
          f"{r2:>11.4e} {slope:>7.4f}")
if strict:
    best = min(strict, key=lambda c: c[2]["max"])
    print(f"\n*** BEST STRICT: {best[0]} max={best[2]['max']:.2e} ***")
    np.savez("results/full_ree/posterior_v3_trim90_G12.npz",
             mu=best[1], u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
    r2, slope, _ = measure_R2(best[1], u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"Saved: 1-R²={r2:.4e}, slope={slope:.4f}")
else:
    print(f"\nNo strict found. Best monotone:")
    monotone = [c for c in candidates if c[2]["u_viol"] == 0
                 and c[2]["p_viol"] == 0]
    if monotone:
        best = min(monotone, key=lambda c: c[2]["max"])
        print(f"  {best[0]}: max={best[2]['max']:.2e}")
        np.savez("results/full_ree/posterior_v3_trim90_G12.npz",
                 mu=best[1], u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
