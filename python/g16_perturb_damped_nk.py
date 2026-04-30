"""Try to break through G=16 strict via:
  - Random perturbations of warm seed
  - Damped Newton-Krylov (custom, with α<1 step size)

Tested combinations:
  - Perturb scales [0, 1e-5, 1e-4, 1e-3, 1e-2]
  - NK damping α [1.0, 0.5, 0.3, 0.1] (1.0 = standard, lower = damped)

Strict criterion: max < 1e-14 with both monotonicities.
"""
import time, json, warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

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


def F_phi_flat(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
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


def damped_newton_krylov(F_func, x0, alpha, maxiter, f_tol, inner_tol=1e-6):
    """Custom damped Newton-Krylov. x_{n+1} = x_n + α * Δx where J Δx = -F.
    Uses finite-difference for J·v."""
    x = x0.copy()
    n = len(x)
    eps_fd = 1e-7

    def Jv(v, x_curr, F_curr):
        # FD approx: J·v ≈ (F(x + ε v) - F(x)) / ε
        ev = np.linalg.norm(v) * eps_fd
        if ev < eps_fd: ev = eps_fd
        F_pert = F_func(x_curr + ev * v / np.linalg.norm(v))
        return (F_pert - F_curr) / (ev / np.linalg.norm(v))

    history = []
    for it in range(maxiter):
        F_curr = F_func(x)
        norm_F = float(np.max(np.abs(F_curr)))
        history.append(norm_F)
        if norm_F < f_tol:
            return x, history, "ok"
        # Solve J Δx = -F_curr
        Jop = LinearOperator((n, n), matvec=lambda v: Jv(v, x, F_curr))
        try:
            dx, info = lgmres(Jop, -F_curr, maxiter=50, rtol=inner_tol)
            if info != 0:
                # GMRES didn't converge cleanly; use partial step
                pass
        except Exception as e:
            return x, history, f"gmres_fail:{e}"
        x = x + alpha * dx
        x = np.clip(x, EPS, 1 - EPS)
    return x, history, "noconv"


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== G={G}: perturb + damped NK to break through ===\n", flush=True)

# Load warm seed: G=15 strict (the last working strict)
ck = np.load("results/full_ree/posterior_v3_strict_G15.npz")
mu_warm15 = ck["mu"]; u_warm15 = ck["u_grid"]; p_warm15 = ck["p_grid"]

# Build G=16 grid
u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)

# Interpolate G=15 → G=16
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


mu0 = interp_mu(mu_warm15, u_warm15, p_warm15, u_grid, p_grid)
mu0 = pava_2d(mu0)


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


# Slow Picard to get to bulk near-FP
print("Slow Picard rounds...", flush=True)
mu = picard_round(mu0, 5000, 2500, 0.05)
mu = picard_round(mu, 5000, 2500, 0.01)
mu = picard_round(mu, 10000, 5000, 0.003)
d_picard = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"After picard: max={d_picard['max']:.3e}, med={d_picard['med']:.3e}, "
      f"u/p={d_picard['u_viol']}/{d_picard['p_viol']}", flush=True)

F_lambda = lambda x: F_phi_flat(x, mu.shape, u_grid, p_grid, p_lo, p_hi,
                                  TAU, GAMMA)

results = []

# Strategy 1: damped Newton with various α
for alpha in [1.0, 0.5, 0.3, 0.1]:
    print(f"\n--- damped NK α={alpha} from picard ---", flush=True)
    x_final, hist, status = damped_newton_krylov(
        F_lambda, mu.ravel(), alpha=alpha, maxiter=200, f_tol=TOL_MAX)
    mu_final = np.clip(x_final.reshape(mu.shape), EPS, 1 - EPS)
    d = measure(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  status={status}, max={d['max']:.3e}, med={d['med']:.3e}, "
          f"u/p={d['u_viol']}/{d['p_viol']}, hist[0..3]={hist[:3]}",
          flush=True)
    results.append({"strat": f"damped_NK_α={alpha}", "alpha": alpha,
                     **d, "iters": len(hist)})

# Strategy 2: random perturbations + standard NK
for perturb_scale in [1e-5, 1e-4, 1e-3, 1e-2]:
    print(f"\n--- perturb scale={perturb_scale} + NK ---", flush=True)
    mu_pert = mu + np.random.RandomState(42).normal(0, perturb_scale, mu.shape)
    mu_pert = np.clip(mu_pert, EPS, 1 - EPS)
    mu_pert = pava_2d(mu_pert)
    x_final, hist, status = damped_newton_krylov(
        F_lambda, mu_pert.ravel(), alpha=1.0, maxiter=200, f_tol=TOL_MAX)
    mu_final = np.clip(x_final.reshape(mu.shape), EPS, 1 - EPS)
    d = measure(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  status={status}, max={d['max']:.3e}, med={d['med']:.3e}, "
          f"u/p={d['u_viol']}/{d['p_viol']}", flush=True)
    results.append({"strat": f"perturb_{perturb_scale}_NK",
                     "perturb": perturb_scale, **d, "iters": len(hist)})

# Strategy 3: perturb + damped NK
for perturb_scale in [1e-4, 1e-3]:
    for alpha in [0.5, 0.3]:
        print(f"\n--- perturb={perturb_scale} + damped α={alpha} ---",
              flush=True)
        mu_pert = mu + np.random.RandomState(42).normal(0, perturb_scale,
                                                          mu.shape)
        mu_pert = np.clip(mu_pert, EPS, 1 - EPS)
        mu_pert = pava_2d(mu_pert)
        x_final, hist, status = damped_newton_krylov(
            F_lambda, mu_pert.ravel(), alpha=alpha, maxiter=300,
            f_tol=TOL_MAX)
        mu_final = np.clip(x_final.reshape(mu.shape), EPS, 1 - EPS)
        d = measure(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        print(f"  status={status}, max={d['max']:.3e}, med={d['med']:.3e}, "
              f"u/p={d['u_viol']}/{d['p_viol']}", flush=True)
        results.append({"strat": f"pert_{perturb_scale}_damp_{alpha}",
                         "perturb": perturb_scale, "alpha": alpha, **d,
                         "iters": len(hist)})

print("\n=== SUMMARY ===")
print(f"{'strat':>30} {'max':>10} {'med':>10} {'u/p':>5}")
for r in results:
    print(f"{r['strat']:>30} {r['max']:>10.2e} {r['med']:>10.2e} "
          f"{r['u_viol']}/{r['p_viol']}")
strict_OK = [r for r in results if r["max"] < TOL_MAX
              and r["u_viol"] == 0 and r["p_viol"] == 0]
if strict_OK:
    print(f"\n*** {len(strict_OK)} strategies achieved strict! ***")
    for r in strict_OK:
        print(f"  {r['strat']}: max={r['max']:.2e}")
else:
    print("\nNone achieved strict.")
with open("results/full_ree/g16_breakthrough.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/full_ree/g16_breakthrough.json")
