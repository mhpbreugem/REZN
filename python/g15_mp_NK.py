"""G=15 Newton-Krylov in mpmath dps=100.

Per iteration:
  1. F = Φ(μ) - μ in mpmath
  2. Build Jacobian J by finite differences (225 evaluations)
  3. Solve J·Δμ = -F via mpmath.lu_solve
  4. Update μ ← μ + α·Δμ (α = 1.0 for Newton; reduce if max grows)

Saves checkpoint every iter. Reports timing and residual at every step.
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

# Import from earlier mp100 script
from g15_mp100 import (
    Lam_mp, logit_mp, crra_demand_mp, f_v_mp, interp_mp,
    phi_step_mp, measure_mp,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 100

RESULTS_DIR = "results/full_ree"
G = 15
TAU = mpf("2")
GAMMA = mpf("0.5")
N_NK_ITERS = 8
H_FD = mpf("1e-40")        # finite-difference perturbation
TARGET = mpf("1e-50")      # residual target


def F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """Residual: F[i][j] = Φ(μ)[i][j] - μ[i][j]."""
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = [[cand[i][j] - mu[i][j] for j in range(G)] for i in range(G)]
    return F, cand


def F_flat(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    F, _ = F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return [F[i][j] for i in range(G) for j in range(G)]


def measure_F_max(F):
    return max(abs(F[i][j]) for i in range(G) for j in range(G))


def measure_F_med(F):
    vals = sorted(abs(F[i][j]) for i in range(G) for j in range(G))
    return vals[len(vals) // 2]


# Load G=15 strict from float64
print(f"Loading G=15 strict, mp.dps={mp.dps}", flush=True)
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu = [[mpf(str(ck["mu"][i, j])) for j in range(G)] for i in range(G)]
u_grid = [mpf(str(ck["u_grid"][i])) for i in range(G)]
p_grid = [[mpf(str(ck["p_grid"][i, j])) for j in range(G)] for i in range(G)]
p_lo = [mpf(str(ck["p_lo"][i])) for i in range(G)]
p_hi = [mpf(str(ck["p_hi"][i])) for i in range(G)]

# Initial F
print("Computing initial F (one phi_step) ...", flush=True)
t0 = time.time()
F_init, _ = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F_max_init = measure_F_max(F_init)
F_med_init = measure_F_med(F_init)
print(f"Initial: ||F||_max={mpmath.nstr(F_max_init, 6)}, "
      f"||F||_med={mpmath.nstr(F_med_init, 6)}, t={time.time()-t0:.1f}s",
      flush=True)

history = [{"iter": 0,
            "F_max": mpmath.nstr(F_max_init, 30),
            "F_med": mpmath.nstr(F_med_init, 30)}]

for nk_iter in range(1, N_NK_ITERS + 1):
    iter_start = time.time()
    print(f"\n=== NK iter {nk_iter}/{N_NK_ITERS} ===", flush=True)

    # 1. Compute current F
    F_curr, _ = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_max_curr = measure_F_max(F_curr)
    F_med_curr = measure_F_med(F_curr)
    print(f"  Current ||F||_max={mpmath.nstr(F_max_curr, 6)}, "
          f"||F||_med={mpmath.nstr(F_med_curr, 6)}", flush=True)
    if F_max_curr < TARGET:
        print(f"  Target reached ({mpmath.nstr(TARGET, 3)})", flush=True)
        break

    # 2. Build Jacobian by finite differences
    print(f"  Building 225x225 Jacobian (FD, h={mpmath.nstr(H_FD, 3)})...",
          flush=True)
    n = G * G
    J = mpmath.zeros(n, n)
    F_flat_curr = [F_curr[i][j] for i in range(G) for j in range(G)]
    t_jac = time.time()
    for col in range(n):
        i, j = col // G, col % G
        # Perturb mu[i][j] by H_FD
        mu_pert = [row[:] for row in mu]
        mu_pert[i][j] = mu_pert[i][j] + H_FD
        F_pert, _ = F_mu(mu_pert, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        F_flat_pert = [F_pert[ii][jj] for ii in range(G) for jj in range(G)]
        for row_idx in range(n):
            J[row_idx, col] = (F_flat_pert[row_idx] - F_flat_curr[row_idx]) / H_FD
        if (col + 1) % 25 == 0:
            elapsed = time.time() - t_jac
            print(f"    Jacobian col {col+1}/{n}, t={elapsed:.1f}s "
                  f"(eta {(n - col - 1) * elapsed / (col + 1):.0f}s)",
                  flush=True)
    print(f"  Jacobian built in {time.time()-t_jac:.1f}s", flush=True)

    # 3. Solve J·Δ = -F
    print("  Solving 225x225 LU system in mp100...", flush=True)
    t_lu = time.time()
    rhs = mpmath.matrix([-F_flat_curr[k] for k in range(n)])
    try:
        delta = mpmath.lu_solve(J, rhs)
    except (mpmath.libmp.NoConvergence, ZeroDivisionError) as e:
        print(f"  LU failed: {e}", flush=True)
        break
    print(f"  LU solve in {time.time()-t_lu:.1f}s", flush=True)

    # 4. Compute step magnitude and update with damping if needed
    delta_max = max(abs(delta[k]) for k in range(n))
    print(f"  ||Δμ||_max = {mpmath.nstr(delta_max, 6)}", flush=True)
    # Default Newton step
    alpha = mpf("1")
    mu_trial = [row[:] for row in mu]
    for k in range(n):
        i, j = k // G, k % G
        mu_trial[i][j] = mu_trial[i][j] + alpha * delta[k]
        # Clip to [eps, 1-eps]
        if mu_trial[i][j] < mpf("1e-50"):
            mu_trial[i][j] = mpf("1e-50")
        if mu_trial[i][j] > mpf(1) - mpf("1e-50"):
            mu_trial[i][j] = mpf(1) - mpf("1e-50")
    F_trial, _ = F_mu(mu_trial, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_max_trial = measure_F_max(F_trial)

    # Backtracking if F grew
    while F_max_trial > F_max_curr * mpf("1.1") and alpha > mpf("0.01"):
        alpha = alpha * mpf("0.5")
        print(f"    Backtrack: α={mpmath.nstr(alpha, 4)}", flush=True)
        mu_trial = [row[:] for row in mu]
        for k in range(n):
            i, j = k // G, k % G
            mu_trial[i][j] = mu_trial[i][j] + alpha * delta[k]
            if mu_trial[i][j] < mpf("1e-50"):
                mu_trial[i][j] = mpf("1e-50")
            if mu_trial[i][j] > mpf(1) - mpf("1e-50"):
                mu_trial[i][j] = mpf(1) - mpf("1e-50")
        F_trial, _ = F_mu(mu_trial, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        F_max_trial = measure_F_max(F_trial)

    F_med_trial = measure_F_med(F_trial)
    mu = mu_trial
    elapsed = time.time() - iter_start
    print(f"  After step: ||F||_max={mpmath.nstr(F_max_trial, 6)}, "
          f"||F||_med={mpmath.nstr(F_med_trial, 6)}, α={mpmath.nstr(alpha, 4)}, "
          f"iter t={elapsed:.0f}s", flush=True)
    history.append({"iter": nk_iter,
                      "F_max": mpmath.nstr(F_max_trial, 30),
                      "F_med": mpmath.nstr(F_med_trial, 30),
                      "alpha": float(alpha),
                      "elapsed_s": elapsed})

    # Save checkpoint
    mu_strs = [[mpmath.nstr(mu[i][j], 100) for j in range(G)]
                for i in range(G)]
    with open(f"{RESULTS_DIR}/posterior_v3_G15_mpNK_iter{nk_iter}.json",
               "w") as f:
        json.dump({"iter": nk_iter,
                    "F_max": mpmath.nstr(F_max_trial, 50),
                    "F_med": mpmath.nstr(F_med_trial, 50),
                    "mu_strings": mu_strs,
                    "dps": mp.dps,
                    "history": history},
                   f, indent=1)

print("\n=== DONE ===")
print(f"Final ||F||_max = {mpmath.nstr(measure_F_max(F_curr), 10)}")
