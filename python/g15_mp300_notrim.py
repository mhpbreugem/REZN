"""G=15 mp300 Newton, no trim (TRIM=0, full p-grid).
Target 1e-100. Updates at every step.
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 300

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    Lam as Lam_f64, init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64,
)
from gap_reparam import pava_p_only, pava_u_only

RESULTS_DIR = "results/full_ree"
G = 15
TAU = mpf("2"); GAMMA = mpf("0.5")
H_FD = mpf("1e-100")
TARGET = mpf("1e-100")
MAX_ITERS = 6


def pava_2d_f64(mu): return pava_u_only(pava_p_only(mu))


def Lam_mp(z):
    if z >= 0: return mpf(1) / (mpf(1) + mpmath.exp(-z))
    e = mpmath.exp(z); return e / (mpf(1) + e)


def logit_mp(p):
    return mpmath.log(p / (mpf(1) - p))


def crra_demand_mp(mu, p, gamma):
    z = (logit_mp(mu) - logit_mp(p)) / gamma
    R = mpmath.exp(z)
    return (R - mpf(1)) / ((mpf(1) - p) + R * p)


def f_v_mp(u, v, tau):
    mean = mpf(v) - mpf("0.5")
    return mpmath.sqrt(tau / (mpf(2) * mpmath.pi)) * mpmath.exp(
        -tau / mpf(2) * (u - mean) ** 2)


def interp_mp(x_target, x_arr, y_arr):
    n = len(x_arr)
    if x_target <= x_arr[0]: return y_arr[0]
    if x_target >= x_arr[-1]: return y_arr[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_arr[mid] > x_target: hi = mid
        else: lo = mid
    x0, x1 = x_arr[lo], x_arr[lo + 1]
    y0, y1 = y_arr[lo], y_arr[lo + 1]
    if x1 == x0: return y0
    w = (x_target - x0) / (x1 - x0)
    return (mpf(1) - w) * y0 + w * y1


def phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    EPS_MP = mpf("1e-150")
    mu_new = [[mu[i][j] for j in range(G)] for i in range(G)]
    f1_grid = [f_v_mp(u_grid[i], 1, tau) for i in range(G)]
    f0_grid = [f_v_mp(u_grid[i], 0, tau) for i in range(G)]
    for i in range(G):
        for j in range(G):
            p0 = p_grid[i][j]
            mu_col = []
            for ii in range(G):
                if p0 < p_grid[ii][0]: val = mu[ii][0]
                elif p0 > p_grid[ii][-1]: val = mu[ii][-1]
                else: val = interp_mp(p0, p_grid[ii], mu[ii])
                if val < EPS_MP: val = EPS_MP
                if val > mpf(1) - EPS_MP: val = mpf(1) - EPS_MP
                mu_col.append(val)
            d = [crra_demand_mp(mu_col[ii], p0, gamma) for ii in range(G)]
            if abs(d[-1] - d[0]) < mpf("1e-100"): continue
            D_i = -d[i]
            targets = [D_i - d[ii] for ii in range(G)]
            d_inc = d[-1] > d[0]
            d_arr = d if d_inc else list(reversed(d))
            u_arr = u_grid if d_inc else list(reversed(u_grid))
            u3_star = []; valid_mask = []
            for ii in range(G):
                if targets[ii] < d_arr[0] or targets[ii] > d_arr[-1]:
                    u3_star.append(None); valid_mask.append(False)
                else:
                    u3 = interp_mp(targets[ii], d_arr, u_arr)
                    u3_star.append(u3)
                    valid_mask.append(u_grid[0] <= u3 <= u_grid[-1])
            valid = [k for k in range(G) if valid_mask[k]]
            if len(valid) < 2: continue
            f1_root = [f_v_mp(u3_star[ii], 1, tau) for ii in valid]
            f0_root = [f_v_mp(u3_star[ii], 0, tau) for ii in valid]
            f1_sweep = [f1_grid[ii] for ii in valid]
            f0_sweep = [f0_grid[ii] for ii in valid]
            A1 = sum(f1_sweep[k] * f1_root[k] for k in range(len(valid)))
            A0 = sum(f0_sweep[k] * f0_root[k] for k in range(len(valid)))
            f1_own = f1_grid[i]; f0_own = f0_grid[i]
            denom = f0_own * A0 + f1_own * A1
            if denom <= 0: continue
            new_val = f1_own * A1 / denom
            if new_val < EPS_MP: new_val = EPS_MP
            if new_val > mpf(1) - EPS_MP: new_val = mpf(1) - EPS_MP
            mu_new[i][j] = new_val
    return mu_new


def F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return [[cand[i][j] - mu[i][j] for j in range(G)] for i in range(G)]


def F_max(F):
    return max(abs(F[i][j]) for i in range(G) for j in range(G))


def F_med(F):
    vals = sorted(abs(F[i][j]) for i in range(G) for j in range(G))
    return vals[len(vals) // 2]


# Build NO-TRIM grid
print(f"Building G={G} no-trim grid (TRIM=0)...", flush=True)
UMAX = 4.0
u_grid_np = np.linspace(-UMAX, UMAX, G)
p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, 2.0, 0.5, G,
                                                   trim=0.0)

# Float64 polish first to get warm
print(f"Float64 picard polish (no-trim)...", flush=True)
mu_f = np.zeros((G, G))
for i, u in enumerate(u_grid_np):
    mu_f[i, :] = Lam_f64(2.0 * u)
mu_f = pava_2d_f64(mu_f)

t0 = time.time(); last_status = t0
for round_idx, (n, na, alpha) in enumerate(
        [(2000, 1000, 0.05), (3000, 1500, 0.01), (5000, 2500, 0.003)]):
    mu_sum = np.zeros_like(mu_f); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                          p_lo_np, p_hi_np, 2.0, 0.5)
        cand = pava_2d_f64(cand)
        mu_f = alpha * cand + (1 - alpha) * mu_f
        mu_f = np.clip(mu_f, EPS_F64, 1 - EPS_F64)
        if it >= n - na:
            mu_sum += mu_f; n_collected += 1
        if time.time() - last_status > 10:
            cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                              p_lo_np, p_hi_np, 2.0, 0.5)
            r = float(np.max(np.abs(cand2 - mu_f)[act2]))
            print(f"  [round {round_idx+1} α={alpha}] iter {it+1}/{n}, "
                  f"max={r:.3e}, t={time.time()-t0:.0f}s", flush=True)
            last_status = time.time()
    mu_f = pava_2d_f64(mu_sum / max(n_collected, 1))

cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                  p_lo_np, p_hi_np, 2.0, 0.5)
F_f64 = float(np.max(np.abs(cand2 - mu_f)[act2]))
print(f"Float64 polish done: max={F_f64:.3e}", flush=True)

# Cast to mp300
mu = [[mpf(str(mu_f[i, j])) for j in range(G)] for i in range(G)]
u_grid = [mpf(str(u_grid_np[i])) for i in range(G)]
p_grid = [[mpf(str(p_grid_np[i, j])) for j in range(G)] for i in range(G)]
p_lo = [mpf(str(p_lo_np[i])) for i in range(G)]
p_hi = [mpf(str(p_hi_np[i])) for i in range(G)]

print(f"\nStarting mp300 NK (target {mpmath.nstr(TARGET, 3)})...", flush=True)
history = []
for nk_iter in range(1, MAX_ITERS + 1):
    t_iter = time.time()
    F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_max_c = F_max(F_curr)
    F_med_c = F_med(F_curr)
    print(f"  NK iter {nk_iter}: F_max={mpmath.nstr(F_max_c, 8)}, "
          f"F_med={mpmath.nstr(F_med_c, 8)}", flush=True)
    if F_max_c < TARGET:
        print(f"  Target reached", flush=True)
        break
    n = G * G
    F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
    print(f"  Building {n}x{n} Jacobian...", flush=True)
    J = mpmath.zeros(n, n)
    t_jac = time.time()
    for col in range(n):
        i, j = col // G, col % G
        mu_pert = [row[:] for row in mu]
        mu_pert[i][j] = mu_pert[i][j] + H_FD
        F_pert = F_mu(mu_pert, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        F_flat_p = [F_pert[ii][jj] for ii in range(G) for jj in range(G)]
        for row_idx in range(n):
            J[row_idx, col] = (F_flat_p[row_idx] - F_flat[row_idx]) / H_FD
        if (col + 1) % 25 == 0:
            elapsed = time.time() - t_jac
            print(f"    Jacobian {col+1}/{n}, t={elapsed:.0f}s, "
                  f"eta {(n-col-1)*elapsed/(col+1):.0f}s", flush=True)
    print(f"  Jacobian done in {time.time()-t_jac:.0f}s", flush=True)
    print(f"  LU solve...", flush=True)
    t_lu = time.time()
    rhs = mpmath.matrix([-F_flat[k] for k in range(n)])
    delta = mpmath.lu_solve(J, rhs)
    print(f"  LU done in {time.time()-t_lu:.0f}s", flush=True)
    for k in range(n):
        i, j = k // G, k % G
        mu[i][j] = mu[i][j] + delta[k]
        if mu[i][j] < mpf("1e-150"): mu[i][j] = mpf("1e-150")
        if mu[i][j] > mpf(1) - mpf("1e-150"):
            mu[i][j] = mpf(1) - mpf("1e-150")
    F_after = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_max_a = F_max(F_after); F_med_a = F_med(F_after)
    elapsed = time.time() - t_iter
    print(f"  After: F_max={mpmath.nstr(F_max_a, 8)}, "
          f"F_med={mpmath.nstr(F_med_a, 8)}, iter t={elapsed:.0f}s",
          flush=True)
    history.append({"iter": nk_iter,
                      "F_max": mpmath.nstr(F_max_a, 100),
                      "F_med": mpmath.nstr(F_med_a, 100),
                      "elapsed_s": elapsed})

# Save
mu_strs = [[mpmath.nstr(mu[i][j], 300) for j in range(G)] for i in range(G)]
out = f"{RESULTS_DIR}/posterior_v3_G{G}_mp300_notrim.json"
with open(out, "w") as f:
    json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "trim": 0.0, "dps": 300,
                "F_max": mpmath.nstr(F_max(F_after), 100),
                "F_med": mpmath.nstr(F_med(F_after), 100),
                "u_grid": [str(x) for x in u_grid_np],
                "p_grid": [[str(p) for p in row] for row in p_grid_np],
                "mu_strings": mu_strs,
                "history": history}, f, indent=1)
print(f"\nSaved {out}", flush=True)
