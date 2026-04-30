"""Explicit CARA test: γ=100, 500, 1000 at G=14, τ=2.

Tests whether 1-R² → 0 as γ→∞ (CRRA approaching CARA limit).
Previous γ=50 stuck at 1-R²=0.037; want to see if it ever goes to zero.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 14
TAU = 2.0
TOL_MAX = 1e-14


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


def measure_full(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


warnings.filterwarnings("ignore", category=RuntimeWarning)
results = []
# Warm-start from γ=2 strict
ck = np.load("results/full_ree/posterior_v3_strict_G14_gamma2.npz")
mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]

for gamma in [50.0, 100.0, 500.0, 1000.0, 5000.0]:
    print(f"\n=== γ={gamma} ===", flush=True)
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, gamma, G)
    mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    t0 = time.time()
    # 3 picard rounds
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma, 5000, 2500, 0.05)
    d1 = measure_full(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    print(f"  picard r1: max={d1['max']:.3e}, med={d1['med']:.3e}",
          flush=True)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma, 5000, 2500, 0.01)
    d2 = measure_full(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    print(f"  picard r2: max={d2['max']:.3e}, med={d2['med']:.3e}",
          flush=True)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma, 10000, 5000, 0.003)
    d3 = measure_full(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    print(f"  picard r3: max={d3['max']:.3e}, med={d3['med']:.3e}",
          flush=True)
    # NK polish
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, TAU, gamma),
            mu.ravel(), f_tol=TOL_MAX, maxiter=300,
            method="lgmres", verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu
        nk_status = f"err:{type(exc).__name__}"
    d_nk = measure_full(mu_nk, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    use_nk = (d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0
               and d_nk["max"] <= d3["max"])
    mu_use = mu_nk if use_nk else mu
    d = d_nk if use_nk else d3
    elapsed = time.time() - t0
    r2, slope, _ = measure_R2(mu_use, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
    print(f"  post-NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
          f"u={d_nk['u_viol']}, p={d_nk['p_viol']}, NK={nk_status}",
          flush=True)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, t={elapsed:.0f}s",
          flush=True)
    results.append({
        "gamma": gamma, "G": G, "tau": TAU,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "use_nk": use_nk,
    })
    # Update warm
    if d["u_viol"] == 0 and d["p_viol"] == 0:
        mu_warm = mu_use; u_warm = u_grid; p_warm = p_grid
    with open("results/full_ree/cara_explicit_test.json", "w") as f:
        json.dump(results, f, indent=2)

print("\n=== SUMMARY: 1-R² toward CARA limit ===")
print(f"{'γ':>6} {'max':>10} {'med':>10} {'u/p':>5} {'1-R²':>11} {'slope':>7}")
for r in results:
    print(f"{r['gamma']:>6.0f} {r['max']:>10.2e} {r['med']:>10.2e} "
          f"{r['u_viol']:>2}/{r['p_viol']:<2} {r['1-R^2']:>11.4e} "
          f"{r['slope']:>7.4f}")
print("\nNo-learning 1-R² floor at G=14 (cf. p0_results.json):")
print(f"  γ=50: ~2.8e-5  γ=100,500,1000: even smaller (essentially 0)")
print(f"\nIf REE 1-R² stays ~0.04 (floor) → method has G=14 floor")
print(f"If REE 1-R² → 0 → method is correct, just needed γ very large")
