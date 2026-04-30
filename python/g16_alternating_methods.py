"""G=16 breakthrough: alternating methods.

Strategies tried (in order):
  A. Linear extrapolation from G=14, G=15 strict for warm-start
  B. Projected Newton: NK step, then PAVA-2D project, repeat
  C. Alternating Picard ↔ NK: a few NK iters, re-PAVA, then Picard, alternate
  D. Genetic-style search: many perturbations, keep best monotone candidate
  E. Constrained least-squares via LM: minimize ||Φ-I||² with bounds
"""
import time, json, warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
from scipy.optimize import least_squares

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 16
TAU = 2.0
GAMMA = 0.5
TOL_MAX = 1e-14


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def F_phi(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
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


def projected_newton(mu_init, F_func, shape, u_grid, p_grid, p_lo, p_hi,
                      tau, gamma, alpha=0.5, maxiter=200, f_tol=1e-14,
                      eps_fd=1e-7):
    """Newton step + PAVA-2D projection at each iteration."""
    mu = mu_init.copy()
    n = mu.size
    for it in range(maxiter):
        F_curr = F_func(mu.ravel())
        norm_F = float(np.max(np.abs(F_curr)))
        if norm_F < f_tol:
            return mu, it, "ok"

        def Jv(v):
            ev = np.linalg.norm(v)
            if ev < eps_fd: ev = eps_fd
            mu_p = mu.ravel() + (eps_fd / ev) * v
            mu_p = np.clip(mu_p, EPS, 1-EPS)
            F_pert = F_func(mu_p)
            return (F_pert - F_curr) / (eps_fd / ev)

        Jop = LinearOperator((n, n), matvec=Jv)
        try:
            dx, info = lgmres(Jop, -F_curr, maxiter=50, rtol=1e-6)
        except Exception:
            return mu, it, "gmres_fail"
        # Damped step
        mu_new = mu.ravel() + alpha * dx
        mu_new = np.clip(mu_new.reshape(shape), EPS, 1 - EPS)
        # Project to monotone
        mu_new = pava_2d(mu_new)
        if it % 30 == 0:
            d = measure(mu_new, u_grid, p_grid, p_lo, p_hi, tau, gamma)
            print(f"    proj-N iter {it}: max={d['max']:.3e}, "
                  f"u/p={d['u_viol']}/{d['p_viol']}", flush=True)
        mu = mu_new
    return mu, maxiter, "noconv"


def alternating_picard_nk(mu_init, F_func, shape, u_grid, p_grid, p_lo, p_hi,
                           tau, gamma, n_alt=5, picard_iter=2000,
                           picard_alpha=0.005, nk_iter=20, f_tol=1e-14):
    """Alternate Picard-PAVA blocks with NK steps."""
    mu = mu_init.copy()
    for round_idx in range(n_alt):
        # Picard block
        mu_sum = np.zeros_like(mu); n_collected = 0
        for it in range(picard_iter):
            cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
            cand = pava_2d(cand)
            mu = picard_alpha * cand + (1 - picard_alpha) * mu
            mu = np.clip(mu, EPS, 1 - EPS)
            if it >= picard_iter // 2:
                mu_sum += mu; n_collected += 1
        mu = pava_2d(mu_sum / n_collected)
        d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        print(f"  alt round {round_idx+1} after picard: max={d['max']:.3e}, "
              f"med={d['med']:.3e}, u/p={d['u_viol']}/{d['p_viol']}",
              flush=True)
        if d["max"] < f_tol and d["u_viol"] == 0 and d["p_viol"] == 0:
            return mu, round_idx, "ok"
        # NK block — projected
        mu, _, _ = projected_newton(mu, F_func, shape, u_grid, p_grid,
                                      p_lo, p_hi, tau, gamma,
                                      alpha=0.3, maxiter=nk_iter, f_tol=f_tol)
        d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        print(f"  alt round {round_idx+1} after proj-NK: max={d['max']:.3e}, "
              f"med={d['med']:.3e}, u/p={d['u_viol']}/{d['p_viol']}",
              flush=True)
        if d["max"] < f_tol and d["u_viol"] == 0 and d["p_viol"] == 0:
            return mu, round_idx, "ok"
    return mu, n_alt, "noconv"


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== G={G} alternating-methods breakthrough ===\n", flush=True)

# Strategy A: Linear extrapolation G=14, G=15 → G=16
ck14 = np.load("results/full_ree/posterior_v3_strict_G14.npz")
ck15 = np.load("results/full_ree/posterior_v3_strict_G15.npz")
mu14, u14, p14 = ck14["mu"], ck14["u_grid"], ck14["p_grid"]
mu15, u15, p15 = ck15["mu"], ck15["u_grid"], ck15["p_grid"]

u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)


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


# A: extrapolate
mu14_at16 = interp_mu(mu14, u14, p14, u_grid, p_grid)
mu15_at16 = interp_mu(mu15, u15, p15, u_grid, p_grid)
# Linear extrap: μ_16 ≈ 2*μ_15 - μ_14 (G-step extrap)
mu_extrap = 2 * mu15_at16 - mu14_at16
mu_extrap = np.clip(mu_extrap, EPS, 1 - EPS)
mu_extrap = pava_2d(mu_extrap)
print("Strategy A: Linear extrap from G=14, G=15", flush=True)
d_A = measure(mu_extrap, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  before: max={d_A['max']:.3e}, med={d_A['med']:.3e}",
      flush=True)


def picard_round(mu, n, na, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


# Burn-in with picard from extrap
mu_A = picard_round(mu_extrap, 5000, 2500, 0.05)
mu_A = picard_round(mu_A, 5000, 2500, 0.01)
mu_A = picard_round(mu_A, 10000, 5000, 0.003)
d_A = measure(mu_A, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  after picard: max={d_A['max']:.3e}, med={d_A['med']:.3e}, "
      f"u/p={d_A['u_viol']}/{d_A['p_viol']}", flush=True)

F_func = lambda x: F_phi(x, mu_A.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)

# Strategy B: Projected Newton from picard
print("\nStrategy B: Projected Newton (PAVA after each NK step)", flush=True)
mu_B, iters, status = projected_newton(mu_A, F_func, mu_A.shape, u_grid,
                                         p_grid, p_lo, p_hi, TAU, GAMMA,
                                         alpha=0.5, maxiter=200, f_tol=TOL_MAX)
d_B = measure(mu_B, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  result: max={d_B['max']:.3e}, med={d_B['med']:.3e}, "
      f"u/p={d_B['u_viol']}/{d_B['p_viol']}, iters={iters}, status={status}",
      flush=True)

# Strategy C: Alternating Picard ↔ Projected NK
print("\nStrategy C: Alternating Picard ↔ projected NK (5 rounds)",
      flush=True)
mu_C, alt_rounds, status_C = alternating_picard_nk(
    mu_A, F_func, mu_A.shape, u_grid, p_grid, p_lo, p_hi,
    TAU, GAMMA, n_alt=5, picard_iter=3000, picard_alpha=0.003,
    nk_iter=30, f_tol=TOL_MAX)
d_C = measure(mu_C, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  result: max={d_C['max']:.3e}, med={d_C['med']:.3e}, "
      f"u/p={d_C['u_viol']}/{d_C['p_viol']}", flush=True)

# Strategy E: LM with bounds
print("\nStrategy E: Levenberg-Marquardt with bounds",
      flush=True)
n = mu_A.size

def F_for_lm(x):
    mu_c = pava_2d(np.clip(x.reshape(mu_A.shape), EPS, 1 - EPS))
    return F_phi(mu_c.ravel(), mu_A.shape, u_grid, p_grid, p_lo, p_hi,
                  TAU, GAMMA)

try:
    res = least_squares(F_for_lm, mu_A.ravel(), method="trf",
                          bounds=(EPS, 1 - EPS), max_nfev=2000,
                          ftol=1e-15, xtol=1e-15, gtol=1e-15)
    mu_E = pava_2d(res.x.reshape(mu_A.shape))
    d_E = measure(mu_E, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  result: max={d_E['max']:.3e}, med={d_E['med']:.3e}, "
          f"u/p={d_E['u_viol']}/{d_E['p_viol']}, "
          f"cost={res.cost:.3e}, status={res.status}", flush=True)
except Exception as e:
    print(f"  LM failed: {e}", flush=True)
    d_E = {"max": float("nan"), "med": float("nan"),
           "u_viol": -1, "p_viol": -1}
    mu_E = mu_A

# Pick best monotone result
candidates = []
if d_A["u_viol"] == 0 and d_A["p_viol"] == 0:
    candidates.append((mu_A, d_A, "A_extrap_picard"))
if d_B["u_viol"] == 0 and d_B["p_viol"] == 0:
    candidates.append((mu_B, d_B, "B_proj_NK"))
if d_C["u_viol"] == 0 and d_C["p_viol"] == 0:
    candidates.append((mu_C, d_C, "C_alternate"))
if d_E.get("u_viol", -1) == 0 and d_E.get("p_viol", -1) == 0:
    candidates.append((mu_E, d_E, "E_LM"))

print(f"\n=== SUMMARY ===")
print(f"{'strat':>22} {'max':>10} {'med':>10} {'u/p':>5}")
for label, d_x in [("A_extrap_picard", d_A), ("B_proj_NK", d_B),
                    ("C_alternate", d_C), ("E_LM", d_E)]:
    print(f"{label:>22} {d_x['max']:>10.2e} {d_x['med']:>10.2e} "
          f"{d_x['u_viol']}/{d_x['p_viol']}")

if candidates:
    best = min(candidates, key=lambda x: x[1]["max"])
    print(f"\nBest monotone: {best[2]} with max={best[1]['max']:.2e}",
          flush=True)
    mu_best = best[0]
    if best[1]["max"] < TOL_MAX:
        print(f"*** STRICT ACHIEVED at G={G}! ***")
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}_alt.npz",
                 mu=mu_best, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
    r2, slope, _ = measure_R2(mu_best, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  1-R²={r2:.4e}, slope={slope:.4f}")
else:
    print("\nNo monotone candidate found.")

results = {
    "G": G, "tau": TAU, "gamma": GAMMA,
    "A_extrap_picard": dict(d_A),
    "B_proj_NK": dict(d_B),
    "C_alternate": dict(d_C),
    "E_LM": dict(d_E) if "max" in d_E else {},
}
with open("results/full_ree/g16_alternating.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/full_ree/g16_alternating.json")
