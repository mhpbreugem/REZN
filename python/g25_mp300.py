"""G=50 Newton at mp300 precision, no trim, warm from G=17 mp200.

WARNING: estimated cost per iter:
  - Jacobian: 2500 cols × ~30s/phi_step at G=50 mp300 = ~21 hours
  - LU solve: 2500x2500 mp300 = ~hours-days
Total per iter: ~1-3 days. May need 3-5 iters.

Checkpoint every iter so progress isn't lost.
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
    init_p_grid as init_p_grid_f64,
)

RESULTS_DIR = "results/full_ree"
G = 25; UMAX = 4.0
TAU = mpf("2"); GAMMA = mpf("0.5")
H_FD = mpf("1e-100")
TARGET = mpf("1e-100")


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


# Build no-trim grid
print(f"Building G={G} no-trim grid...", flush=True)
u_grid_np = np.linspace(-UMAX, UMAX, G)
p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, 2.0, 0.5, G,
                                                   trim=0.0)

# Warm: G=15 mp300 iter 2 → G=25 interp (in mpmath)
print(f"Loading G=15 mp300 iter 2 warm...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G15_mp300_iter2.json") as f:
    state15 = json.load(f)
G17 = 15
mu_17_strs = state15["mu_strings"]
mu_17 = [[mpf(mu_17_strs[i][j]) for j in range(G17)] for i in range(G17)]

# G=15 grid: rebuild via init_p_grid in float64 then cast
u_prev_np = np.linspace(-UMAX, UMAX, G17)
p_lo_prev_np, p_hi_prev_np, p_grid_prev_np = init_p_grid_f64(
    u_prev_np, 2.0, 0.5, G17, trim=0.05)
u_17 = [mpf(str(x)) for x in u_prev_np]
p_17 = [[mpf(str(p)) for p in row] for row in p_grid_prev_np]

# Build mp grid
u_grid = [mpf(str(x)) for x in u_grid_np]
p_grid = [[mpf(str(p)) for p in row] for row in p_grid_np]
p_lo = [mpf(str(x)) for x in p_lo_np]
p_hi = [mpf(str(x)) for x in p_hi_np]

# Interp G=15 → G=25 in mp
print(f"Interpolating G=15 → G=25 in mp300...", flush=True)
mu = []
for i in range(G):
    row = []
    u = u_grid[i]
    # Find G=17 row
    if u <= u_17[0]: idx_a = idx_b = 0; w = mpf(0)
    elif u >= u_17[-1]: idx_a = idx_b = G17 - 1; w = mpf(0)
    else:
        for k in range(1, G17):
            if u_17[k] >= u:
                idx_a = k; idx_b = k - 1
                w = (u - u_17[idx_b]) / (u_17[idx_a] - u_17[idx_b])
                break
    for j in range(G):
        p = p_grid[i][j]
        p_b = p
        if p_b < p_17[idx_b][0]: p_b = p_17[idx_b][0]
        if p_b > p_17[idx_b][-1]: p_b = p_17[idx_b][-1]
        p_a = p
        if p_a < p_17[idx_a][0]: p_a = p_17[idx_a][0]
        if p_a > p_17[idx_a][-1]: p_a = p_17[idx_a][-1]
        m_b = interp_mp(p_b, p_17[idx_b], mu_17[idx_b])
        m_a = interp_mp(p_a, p_17[idx_a], mu_17[idx_a])
        val = (mpf(1) - w) * m_b + w * m_a
        if val < mpf("1e-150"): val = mpf("1e-150")
        if val > mpf(1) - mpf("1e-150"): val = mpf(1) - mpf("1e-150")
        row.append(val)
    mu.append(row)

print(f"Initial F at mp300...", flush=True)
t0 = time.time()
F_init = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F_max_i = F_max(F_init); F_med_i = F_med(F_init)
print(f"Initial: max={mpmath.nstr(F_max_i, 6)}, med={mpmath.nstr(F_med_i, 6)}, "
      f"phi_step={time.time()-t0:.1f}s", flush=True)

print(f"\nETA: Jacobian = 2500 × {time.time()-t0:.0f}s ≈ "
      f"{2500*(time.time()-t0)/3600:.1f} hours per iter\n", flush=True)

history = []
for nk_iter in range(1, 4):
    t_iter = time.time()
    F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_max_c = F_max(F_curr); F_med_c = F_med(F_curr)
    print(f"=== NK iter {nk_iter} ===", flush=True)
    print(f"  Current F_max={mpmath.nstr(F_max_c, 8)}, "
          f"F_med={mpmath.nstr(F_med_c, 8)}", flush=True)
    if F_max_c < TARGET:
        print(f"  Target reached", flush=True)
        break
    n = G * G
    F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
    print(f"  Building {n}×{n} Jacobian (each col ~"
          f"{(time.time()-t0):.0f}s)...", flush=True)
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
            print(f"    col {col+1}/{n}, t={elapsed:.0f}s, "
                  f"eta {(n-col-1)*elapsed/(col+1):.0f}s "
                  f"({(n-col-1)*elapsed/(col+1)/3600:.1f}h)",
                  flush=True)
    print(f"  Jacobian done in {(time.time()-t_jac)/3600:.1f}h", flush=True)
    print(f"  LU solve...", flush=True)
    t_lu = time.time()
    rhs = mpmath.matrix([-F_flat[k] for k in range(n)])
    delta = mpmath.lu_solve(J, rhs)
    print(f"  LU done in {(time.time()-t_lu)/60:.0f}min", flush=True)
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
          f"F_med={mpmath.nstr(F_med_a, 8)}, t={elapsed/3600:.1f}h",
          flush=True)
    history.append({"iter": nk_iter,
                      "F_max": mpmath.nstr(F_max_a, 100),
                      "F_med": mpmath.nstr(F_med_a, 100),
                      "elapsed_s": elapsed})
    # Save checkpoint
    mu_strs = [[mpmath.nstr(mu[i][j], 200) for j in range(G)]
                for i in range(G)]
    out = f"{RESULTS_DIR}/posterior_v3_G{G}_mp300_iter{nk_iter}.json"
    with open(out, "w") as f:
        json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "trim": 0.0, "dps": 300,
                    "iter": nk_iter,
                    "F_max": mpmath.nstr(F_max_a, 100),
                    "F_med": mpmath.nstr(F_med_a, 100),
                    "mu_strings": mu_strs,
                    "history": history}, f, indent=1)
    print(f"  Saved {out}", flush=True)

print("\n=== mp300 G=50 DONE ===", flush=True)
