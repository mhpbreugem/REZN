"""Smart ladder mp300 NO trim, single G steps from G=17 (mp200 strict).

For G in [18, 19, 20, ..., 25]:
  Warm from previous G mp result.
  NK at mp300 trim=0 with LU regularization fallback.
  Save mp300 ckpt.

Reports per col every 25 cols.
"""
import time, json, warnings, os
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 300

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import init_p_grid as init_p_grid_f64

RESULTS_DIR = "results/full_ree"
TAU = mpf("2"); GAMMA = mpf("0.5"); UMAX = 4.0
H_FD = mpf("1e-100")
TARGET = mpf("1e-100")
GS = list(range(18, 26))   # 18..25


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


def F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, G):
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, G)
    return [[cand[i][j] - mu[i][j] for j in range(G)] for i in range(G)]


def F_max(F, G):
    return max(abs(F[i][j]) for i in range(G) for j in range(G))


def F_med(F, G):
    vals = sorted(abs(F[i][j]) for i in range(G) for j in range(G))
    return vals[len(vals) // 2]


def interp_grid(mu_old, u_old, p_old_grid, u_new, p_new_grid, G_new, G_old):
    """Interp mu_old (lists, mpf) onto new grid."""
    mu = []
    for i in range(G_new):
        row = []
        u = u_new[i]
        if u <= u_old[0]: idx_a = idx_b = 0; w = mpf(0)
        elif u >= u_old[-1]: idx_a = idx_b = G_old - 1; w = mpf(0)
        else:
            for k in range(1, G_old):
                if u_old[k] >= u:
                    idx_a = k; idx_b = k - 1
                    w = (u - u_old[idx_b]) / (u_old[idx_a] - u_old[idx_b])
                    break
        for j in range(G_new):
            p = p_new_grid[i][j]
            p_b = p
            if p_b < p_old_grid[idx_b][0]: p_b = p_old_grid[idx_b][0]
            if p_b > p_old_grid[idx_b][-1]: p_b = p_old_grid[idx_b][-1]
            p_a = p
            if p_a < p_old_grid[idx_a][0]: p_a = p_old_grid[idx_a][0]
            if p_a > p_old_grid[idx_a][-1]: p_a = p_old_grid[idx_a][-1]
            m_b = interp_mp(p_b, p_old_grid[idx_b], mu_old[idx_b])
            m_a = interp_mp(p_a, p_old_grid[idx_a], mu_old[idx_a])
            val = (mpf(1) - w) * m_b + w * m_a
            if val < mpf("1e-150"): val = mpf("1e-150")
            if val > mpf(1) - mpf("1e-150"): val = mpf(1) - mpf("1e-150")
            row.append(val)
        mu.append(row)
    return mu


def lm_step(J, F_flat, n, lam):
    """Levenberg-Marquardt step: (J^T J + λ I) Δx = -J^T F."""
    # JTJ = J.T * J
    JT = J.T
    JTJ = JT * J
    for k in range(n):
        JTJ[k, k] = JTJ[k, k] + lam
    JTF = JT * mpmath.matrix([-F_flat[k] for k in range(n)])
    return mpmath.lu_solve(JTJ, JTF)


def nk_solve_mp(mu, u_grid, p_grid, p_lo, p_hi, G, max_iters=4):
    history = []
    lam = mpf("1e-30")  # initial LM damping
    for nk_iter in range(1, max_iters + 1):
        t_iter = time.time()
        F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
        F_max_c = F_max(F_curr, G); F_med_c = F_med(F_curr, G)
        F_norm_c = sum(abs(F_curr[i][j])**2 for i in range(G)
                         for j in range(G))
        print(f"  NK iter {nk_iter}: F_max={mpmath.nstr(F_max_c, 6)}, "
              f"F_med={mpmath.nstr(F_med_c, 6)}", flush=True)
        if F_max_c < TARGET:
            print(f"  Target reached", flush=True)
            break
        n = G * G
        F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
        print(f"    Building {n}×{n} Jacobian...", flush=True)
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
            if (col + 1) % 50 == 0:
                elapsed = time.time() - t_jac
                print(f"      col {col+1}/{n}, t={elapsed:.0f}s, "
                      f"eta {(n-col-1)*elapsed/(col+1):.0f}s",
                      flush=True)
        print(f"    Jacobian {(time.time()-t_jac)/60:.0f}min", flush=True)

        # Levenberg-Marquardt with adaptive λ
        print(f"    LM step (λ={mpmath.nstr(lam, 3)})...", flush=True)
        t_lu = time.time()
        accepted = False
        for attempt in range(8):
            try:
                delta = lm_step(J, F_flat, n, lam)
            except ZeroDivisionError:
                print(f"    LM singular at λ={mpmath.nstr(lam, 3)}; "
                      f"increasing", flush=True)
                lam = lam * mpf(100)
                continue
            # Try the step
            mu_trial = [row[:] for row in mu]
            for k in range(n):
                i, j = k // G, k % G
                mu_trial[i][j] = mu_trial[i][j] + delta[k]
                if mu_trial[i][j] < mpf("1e-150"):
                    mu_trial[i][j] = mpf("1e-150")
                if mu_trial[i][j] > mpf(1) - mpf("1e-150"):
                    mu_trial[i][j] = mpf(1) - mpf("1e-150")
            F_try = F_mu(mu_trial, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
            F_norm_try = sum(abs(F_try[i][j])**2 for i in range(G)
                               for j in range(G))
            if F_norm_try < F_norm_c:
                # Accept, decrease λ
                mu = mu_trial
                lam = lam / mpf(10)
                accepted = True
                F_norm_c = F_norm_try
                print(f"    LM accepted, new λ={mpmath.nstr(lam, 3)}",
                      flush=True)
                break
            else:
                # Reject, increase λ
                lam = lam * mpf(10)
                print(f"    LM rejected (norm grew), λ={mpmath.nstr(lam, 3)}",
                      flush=True)
        if not accepted:
            print(f"    LM no improvement after 8 attempts; keeping mu",
                  flush=True)
        print(f"    LM step total {(time.time()-t_lu)/60:.0f}min",
              flush=True)

        F_after = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
        F_max_a = F_max(F_after, G); F_med_a = F_med(F_after, G)
        elapsed = time.time() - t_iter
        print(f"    After: max={mpmath.nstr(F_max_a, 6)}, "
              f"med={mpmath.nstr(F_med_a, 6)}, t={elapsed/60:.0f}min",
              flush=True)
        history.append({"iter": nk_iter,
                          "F_max": mpmath.nstr(F_max_a, 50),
                          "F_med": mpmath.nstr(F_med_a, 50),
                          "lambda": mpmath.nstr(lam, 10),
                          "elapsed_s": elapsed})
    return mu, F_max_a, F_med_a, history


# Load G=17 mp200 (cast up to mp300)
print(f"Loading G=17 mp200 → mp300...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G17_mp200.json") as f:
    state = json.load(f)
G_PREV = 17
mu_prev = [[mpf(state["mu_strings"][i][j]) for j in range(G_PREV)]
              for i in range(G_PREV)]
u_prev = [mpf(s) for s in state["u_grid"]]
p_grid_prev = [[mpf(p) for p in row] for row in state["p_grid"]]

for G in GS:
    print(f"\n{'='*60}\n=== G = {G} (mp300, no trim) ===\n{'='*60}",
          flush=True)
    t_G = time.time()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, 2.0, 0.5, G,
                                                       trim=0.0)
    u_grid = [mpf(str(x)) for x in u_grid_np]
    p_grid = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo = [mpf(str(x)) for x in p_lo_np]
    p_hi = [mpf(str(x)) for x in p_hi_np]

    print(f"  Interp G={G_PREV} → G={G} in mp300...", flush=True)
    mu = interp_grid(mu_prev, u_prev, p_grid_prev, u_grid, p_grid, G,
                       G_PREV)

    F0 = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, G)
    print(f"  After interp: max={mpmath.nstr(F_max(F0, G), 6)}, "
          f"med={mpmath.nstr(F_med(F0, G), 6)}", flush=True)

    mu, F_max_f, F_med_f, history = nk_solve_mp(mu, u_grid, p_grid,
                                                    p_lo, p_hi, G, max_iters=15)

    elapsed = time.time() - t_G
    print(f"\n  G={G} TOTAL t={elapsed/60:.0f}min, "
          f"final max={mpmath.nstr(F_max_f, 6)}", flush=True)

    # Save
    mu_strs = [[mpmath.nstr(mu[i][j], 200) for j in range(G)]
                for i in range(G)]
    out = f"{RESULTS_DIR}/posterior_v3_G{G}_mp300_notrim.json"
    with open(out, "w") as f:
        json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "trim": 0.0, "dps": 300,
                    "F_max": mpmath.nstr(F_max_f, 100),
                    "F_med": mpmath.nstr(F_med_f, 100),
                    "u_grid": [str(x) for x in u_grid_np],
                    "p_grid": [[str(p) for p in row] for row in p_grid_np],
                    "mu_strings": mu_strs,
                    "history": history,
                    "elapsed_s": elapsed}, f, indent=1)
    print(f"  Saved {out}", flush=True)

    # Set up for next G
    G_PREV = G
    mu_prev = mu; u_prev = u_grid; p_grid_prev = p_grid

print("\n=== LADDER 18..25 mp300 NO-TRIM DONE ===", flush=True)
