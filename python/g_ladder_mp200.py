"""G-ladder mp200 NK from G=15 to G=25.

For each G:
  1. Build u/p grids
  2. Warm-start μ via float64 posterior_method (then cast to mp)
     OR interpolate previous G's converged μ
  3. NK iters until max-residual < 1e-100 OR convergence stalls

mp200 = 200 decimal digits.
"""
import time, json, warnings, os
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 200

RESULTS_DIR = "results/full_ree"
TAU = mpf("2")
GAMMA = mpf("0.5")
H_FD = mpf("1e-80")
TARGET = mpf("1e-100")
GS = list(range(15, 26))   # 15..25

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    Lam as Lam_f64, init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64,
)
from gap_reparam import pava_p_only, pava_u_only


def pava_2d_f64(mu): return pava_u_only(pava_p_only(mu))


def Lam_mp(z):
    if z >= 0:
        return mpf(1) / (mpf(1) + mpmath.exp(-z))
    e = mpmath.exp(z)
    return e / (mpf(1) + e)


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


def phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, G):
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
            if abs(d[-1] - d[0]) < mpf("1e-80"): continue
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


def F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, G):
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, G)
    return [[cand[i][j] - mu[i][j] for j in range(G)] for i in range(G)]


def F_max(F, G):
    return max(abs(F[i][j]) for i in range(G) for j in range(G))


def F_med(F, G):
    vals = sorted(abs(F[i][j]) for i in range(G) for j in range(G))
    return vals[len(vals) // 2]


def get_warm_start_f64(G, prev_state=None):
    """Get warm-start μ from posterior_method_v3 strict ckpt or interpolate."""
    UMAX = 4.0
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, 2.0, 0.5, G,
                                                       trim=0.05)
    if prev_state is not None:
        # Interpolate from previous G's converged μ
        mu_prev = prev_state["mu_arr"]
        u_prev = prev_state["u_grid_np"]
        p_prev = prev_state["p_grid_np"]
        G_prev = mu_prev.shape[0]
        mu_init = np.zeros((G, G))
        for i in range(G):
            u = u_grid_np[i]
            u_c = np.clip(u, u_prev[0], u_prev[-1])
            ra = np.searchsorted(u_prev, u_c)
            rb = max(ra - 1, 0); ra = min(ra, G_prev - 1)
            w = (u_c - u_prev[rb]) / (u_prev[ra] - u_prev[rb]) if ra != rb else 1.0
            for j in range(G):
                p = p_grid_np[i, j]
                p_b = np.clip(p, p_prev[rb, 0], p_prev[rb, -1])
                m_b = np.interp(p_b, p_prev[rb, :], mu_prev[rb, :])
                p_a = np.clip(p, p_prev[ra, 0], p_prev[ra, -1])
                m_a = np.interp(p_a, p_prev[ra, :], mu_prev[ra, :])
                mu_init[i, j] = (1 - w) * m_b + w * m_a
        mu_init = np.clip(mu_init, EPS_F64, 1 - EPS_F64)
    else:
        # Cold start: no learning
        mu_init = np.zeros((G, G))
        for i, u in enumerate(u_grid_np):
            mu_init[i, :] = Lam_f64(2.0 * u)

    # Polish in float64 with picard+PAVA to drive close to FP
    print(f"  Float64 picard polish at G={G}...", flush=True)
    mu = pava_2d_f64(mu_init)
    for round_idx, (n, na, alpha) in enumerate(
            [(2000, 1000, 0.05), (2000, 1000, 0.01), (5000, 2500, 0.003)]):
        mu_sum = np.zeros_like(mu); n_collected = 0
        for it in range(n):
            cand, active, _ = phi_step_f64(mu, u_grid_np, p_grid_np,
                                              p_lo_np, p_hi_np, 2.0, 0.5)
            cand = pava_2d_f64(cand)
            mu = alpha * cand + (1 - alpha) * mu
            mu = np.clip(mu, EPS_F64, 1 - EPS_F64)
            if it >= n - na:
                mu_sum += mu; n_collected += 1
        mu = pava_2d_f64(mu_sum / n_collected)
        cand2, active2, _ = phi_step_f64(mu, u_grid_np, p_grid_np,
                                            p_lo_np, p_hi_np, 2.0, 0.5)
        F_now = float(np.max(np.abs(cand2 - mu)[active2]))
        print(f"    round {round_idx+1}: max={F_now:.3e}", flush=True)
    return u_grid_np, p_grid_np, p_lo_np, p_hi_np, mu


def nk_in_mp(mu_f64, u_grid_np, p_grid_np, p_lo_np, p_hi_np, G,
              max_iters=5):
    # Convert to mp
    mu = [[mpf(str(mu_f64[i, j])) for j in range(G)] for i in range(G)]
    u_grid = [mpf(str(u_grid_np[i])) for i in range(G)]
    p_grid = [[mpf(str(p_grid_np[i, j])) for j in range(G)]
                for i in range(G)]
    p_lo = [mpf(str(p_lo_np[i])) for i in range(G)]
    p_hi = [mpf(str(p_hi_np[i])) for i in range(G)]

    history = []
    for nk in range(1, max_iters + 1):
        t_iter = time.time()
        F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
        F_max_c = F_max(F_curr, G)
        F_med_c = F_med(F_curr, G)
        print(f"  NK iter {nk}: F_max={mpmath.nstr(F_max_c, 6)}, "
              f"F_med={mpmath.nstr(F_med_c, 6)}", flush=True)
        if F_max_c < TARGET:
            print(f"  Target reached", flush=True)
            break
        n = G * G
        F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
        # Build Jacobian
        print(f"  Building {n}x{n} Jacobian...", flush=True)
        J = mpmath.zeros(n, n)
        t_jac = time.time()
        for col in range(n):
            i, j = col // G, col % G
            mu_pert = [row[:] for row in mu]
            mu_pert[i][j] = mu_pert[i][j] + H_FD
            F_pert = F_mu(mu_pert, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
            F_flat_p = [F_pert[ii][jj] for ii in range(G) for jj in range(G)]
            for row_idx in range(n):
                J[row_idx, col] = (F_flat_p[row_idx] - F_flat[row_idx]) / H_FD
            if (col + 1) % max(1, n // 8) == 0:
                elapsed = time.time() - t_jac
                eta = (n - col - 1) * elapsed / (col + 1)
                print(f"    Jacobian {col+1}/{n}, t={elapsed:.0f}s, "
                      f"eta {eta:.0f}s", flush=True)
        print(f"  Jacobian done in {time.time()-t_jac:.0f}s", flush=True)
        print(f"  LU solve...", flush=True)
        t_lu = time.time()
        rhs = mpmath.matrix([-F_flat[k] for k in range(n)])
        try:
            delta = mpmath.lu_solve(J, rhs)
        except Exception as e:
            print(f"  LU failed: {e}", flush=True)
            break
        print(f"  LU done in {time.time()-t_lu:.0f}s", flush=True)
        for k in range(n):
            i, j = k // G, k % G
            mu[i][j] = mu[i][j] + delta[k]
            if mu[i][j] < mpf("1e-150"): mu[i][j] = mpf("1e-150")
            if mu[i][j] > mpf(1) - mpf("1e-150"):
                mu[i][j] = mpf(1) - mpf("1e-150")
        F_after = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
        F_max_a = F_max(F_after, G)
        F_med_a = F_med(F_after, G)
        elapsed = time.time() - t_iter
        print(f"  After NK: F_max={mpmath.nstr(F_max_a, 6)}, "
              f"F_med={mpmath.nstr(F_med_a, 6)}, t={elapsed:.0f}s",
              flush=True)
        history.append({"iter": nk,
                          "F_max": mpmath.nstr(F_max_a, 50),
                          "F_med": mpmath.nstr(F_med_a, 50),
                          "elapsed_s": elapsed})

    return mu, F_max_c, F_med_c, history


# Main
print(f"=== G-ladder mp200, G=15..25 ===\n", flush=True)
prev_state = None
results = []

for G in GS:
    print(f"\n{'='*60}\nG = {G}\n{'='*60}", flush=True)
    t_G = time.time()
    u_grid_np, p_grid_np, p_lo_np, p_hi_np, mu_f64 = get_warm_start_f64(
        G, prev_state)
    cand2, active2, _ = phi_step_f64(mu_f64, u_grid_np, p_grid_np,
                                        p_lo_np, p_hi_np, 2.0, 0.5)
    F_f64 = float(np.max(np.abs(cand2 - mu_f64)[active2]))
    print(f"  Float64 max-residual: {F_f64:.3e}", flush=True)

    print(f"  Starting mp200 NK at G={G}...", flush=True)
    mu_mp, F_max_final, F_med_final, history = nk_in_mp(
        mu_f64, u_grid_np, p_grid_np, p_lo_np, p_hi_np, G, max_iters=4)

    elapsed_G = time.time() - t_G
    print(f"  Final F_max: {mpmath.nstr(F_max_final, 8)}, "
          f"F_med: {mpmath.nstr(F_med_final, 8)}", flush=True)
    print(f"  G={G} total time: {elapsed_G:.0f}s", flush=True)

    # Save
    mu_strs = [[mpmath.nstr(mu_mp[i][j], 200) for j in range(G)]
                for i in range(G)]
    out = f"{RESULTS_DIR}/posterior_v3_G{G}_mp200.json"
    with open(out, "w") as f:
        json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "dps": 200,
                    "F_max": mpmath.nstr(F_max_final, 100),
                    "F_med": mpmath.nstr(F_med_final, 100),
                    "u_grid": [str(x) for x in u_grid_np],
                    "p_grid": [[str(p) for p in row] for row in p_grid_np],
                    "mu_strings": mu_strs,
                    "history": history,
                    "elapsed_s": elapsed_G},
                   f, indent=1)
    results.append({"G": G,
                      "F_max": mpmath.nstr(F_max_final, 30),
                      "F_med": mpmath.nstr(F_med_final, 30),
                      "elapsed_s": elapsed_G})
    print(f"  Saved {out}", flush=True)

    # Keep for next iter as warm-start
    prev_state = {"mu_arr": np.array([[float(mu_mp[i][j]) for j in range(G)]
                                          for i in range(G)]),
                   "u_grid_np": u_grid_np, "p_grid_np": p_grid_np}

print("\n\n=== LADDER DONE ===\n", flush=True)
print(f"{'G':>3} {'F_max':>15} {'F_med':>15} {'time(s)':>10}")
for r in results:
    print(f"{r['G']:>3} {r['F_max']:>15} {r['F_med']:>15} "
          f"{r['elapsed_s']:>10.0f}")
