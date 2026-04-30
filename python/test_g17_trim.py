"""Test G=17 with trim, warm from G=16 strict."""
import time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; G = 17; TAU = 2.0; GAMMA = 0.5; TOL_MAX = 1e-14; TRIM = 0.025


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


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== G={G} with trim, warm from G=16 trim strict ===\n", flush=True)
ck = np.load("results/full_ree/posterior_v3_trim_G16.npz")
mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]

u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G, trim=TRIM)
mu = interp_mu(mu_w, u_w, p_w, u_grid, p_grid)
mu = pava_2d(mu)

t0 = time.time()
mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.05)
mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.01)
mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 10000, 5000, 0.003)
d = measure(mu, u_grid, p_grid, p_lo, p_hi)
print(f"Picard: max={d['max']:.3e}, med={d['med']:.3e}, "
      f"u/p={d['u_viol']}/{d['p_viol']}, t={time.time()-t0:.0f}s", flush=True)

print("NK polish...", flush=True)
try:
    sol = newton_krylov(
        lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi),
        mu.ravel(), f_tol=TOL_MAX, maxiter=300, method="lgmres",
        verbose=False)
    mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "ok"
except NoConvergence as e:
    mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "noconv_kept"
d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi)
print(f"Post-NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
      f"u/p={d_nk['u_viol']}/{d_nk['p_viol']}, NK={nk_status}", flush=True)

if d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
    mu_use = mu_nk
    status = "strict_NK"
elif d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
    mu_use = mu_nk
    status = "non_strict_NK_monotone"
else:
    mu_use = mu
    status = "fallback_picard"

r2, slope, _ = measure_R2(mu_use, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
d_final = measure(mu_use, u_grid, p_grid, p_lo, p_hi)
print(f"\nFinal: max={d_final['max']:.3e}, 1-R²={r2:.4e}, slope={slope:.4f}, "
      f"status={status}", flush=True)
if status.startswith("strict") or "monotone" in status:
    np.savez(f"results/full_ree/posterior_v3_trim_G{G}.npz",
             mu=mu_use, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
    print(f"Saved checkpoint")
