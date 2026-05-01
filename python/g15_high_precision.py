"""G=15 posterior_method_v3 high-precision polish.

Many slow Picard rounds (100k iters) + multiple NK polish passes.
Goal: drive max-residual to absolute machine epsilon (1e-16).
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; G = 15; TAU = 2.0; GAMMA = 0.5
TOL_ULTRA = 1e-15
RESULTS_DIR = "results/full_ree"


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


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


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== G={G} HIGH-PRECISION polish, target 1e-15+ ===\n", flush=True)
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu = ck["mu"]; u_grid = ck["u_grid"]; p_grid = ck["p_grid"]
p_lo = ck["p_lo"]; p_hi = ck["p_hi"]
d = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"Initial (G=15 strict): max={d['max']:.3e}, med={d['med']:.3e}",
      flush=True)

# Multiple NK polish passes (each starts from previous best)
for pass_idx in range(1, 11):
    t0 = time.time()
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA),
            mu.ravel(), f_tol=TOL_ULTRA, maxiter=100, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    if d_nk["u_viol"] > 0 or d_nk["p_viol"] > 0:
        print(f"  Pass {pass_idx}: NK drifted (u={d_nk['u_viol']},"
              f" p={d_nk['p_viol']}); reverting", flush=True)
        # Revert and try slow Picard polish instead
        mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi,
                            10000, 5000, 0.001, TAU, GAMMA)
        d_now = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        print(f"  Pass {pass_idx} slow-Picard: max={d_now['max']:.3e}, "
              f"med={d_now['med']:.3e}, t={time.time()-t0:.0f}s",
              flush=True)
    else:
        if d_nk["max"] < d["max"]:
            mu = mu_nk
            d = d_nk
        print(f"  Pass {pass_idx} NK: max={d_nk['max']:.3e}, "
              f"med={d_nk['med']:.3e}, status={nk_status}, "
              f"t={time.time()-t0:.0f}s", flush=True)
    # Early stop
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_now = float(np.max(np.abs(cand - mu)[active]))
    if F_now < TOL_ULTRA:
        print(f"  Reached target (max={F_now:.3e}), stopping early.",
              flush=True)
        break

# Final report
d = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
r2, slope, n = measure_R2(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\nFINAL: max={d['max']:.3e}, med={d['med']:.3e}, "
      f"u/p={d['u_viol']}/{d['p_viol']}", flush=True)
print(f"      1-R²={r2:.6e}, slope={slope:.6f}, samples={n}", flush=True)

np.savez(f"{RESULTS_DIR}/posterior_v3_G15_ultra.npz",
         mu=mu, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
print(f"\nSaved {RESULTS_DIR}/posterior_v3_G15_ultra.npz", flush=True)
