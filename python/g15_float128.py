"""G=15 polish in float128 (longdouble) to escape float64 epsilon floor.

numpy longdouble eps ~ 1e-19 vs float64's 2e-16. 1000x more precision.
Custom interp/PAVA in float128 since scipy/np.interp force float64.
"""
import time, json, warnings
import numpy as np
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
LD = np.longdouble  # 128-bit
EPS_LD = LD(1e-12)

UMAX = LD(4.0); G = 15
TAU = LD(2.0); GAMMA = LD(0.5)


def Lam_ld(z):
    """Sigmoid in longdouble."""
    z = np.asarray(z, dtype=LD)
    if z.ndim == 0:
        v = float(z)
        if v >= 0:
            return LD(1) / (LD(1) + np.exp(-z))
        e = np.exp(z)
        return e / (LD(1) + e)
    out = np.empty_like(z, dtype=LD)
    pos = z >= 0
    out[pos] = LD(1) / (LD(1) + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (LD(1) + e)
    return out


def logit_ld(p):
    p = np.clip(np.asarray(p, dtype=LD), EPS_LD, LD(1) - EPS_LD)
    out = np.log(p / (LD(1) - p))
    return out if out.ndim > 0 else LD(out)


def crra_demand_ld(mu, p, gamma):
    z = (logit_ld(mu) - logit_ld(p)) / gamma
    R = np.exp(np.clip(z, LD(-50), LD(50)))
    return (R - LD(1)) / ((LD(1) - p) + R * p)


def f_v_ld(u, v, tau):
    mean = LD(v) - LD(0.5)
    return np.sqrt(tau / (LD(2) * LD(math.pi))) * np.exp(
        LD(-0.5) * tau * (np.asarray(u, dtype=LD) - mean) ** 2)


def interp_ld_1d(x_target, x_arr, y_arr):
    """Linear interp in longdouble. x_arr must be sorted ascending."""
    n = len(x_arr)
    if x_target <= x_arr[0]: return y_arr[0]
    if x_target >= x_arr[-1]: return y_arr[-1]
    # Binary search
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_arr[mid] > x_target: hi = mid
        else: lo = mid
    x0, x1 = x_arr[lo], x_arr[lo + 1]
    y0, y1 = y_arr[lo], y_arr[lo + 1]
    if x1 == x0: return y0
    w = (x_target - x0) / (x1 - x0)
    return (LD(1) - w) * y0 + w * y1


def pava_inc(y):
    """In-place PAVA non-decreasing isotonic regression, longdouble."""
    y = y.copy()
    n = len(y)
    if n < 2: return y
    # Pool adjacent violators (use float64 for blocks list)
    means = list(y); weights = [1] * n
    while True:
        done = True; j = 0
        while j < len(means) - 1:
            if means[j] > means[j + 1]:
                w_new = weights[j] + weights[j + 1]
                m_new = (means[j] * weights[j] + means[j + 1] * weights[j + 1]) / LD(w_new)
                means[j] = m_new
                weights[j] = w_new
                means.pop(j + 1); weights.pop(j + 1)
                done = False
            else:
                j += 1
        if done: break
    out = np.empty(n, dtype=LD)
    idx = 0
    for k in range(len(means)):
        for _ in range(weights[k]):
            out[idx] = means[k]
            idx += 1
    return out


def pava_2d_ld(mu):
    """PAVA in p-direction then u-direction."""
    G_u, G_p = mu.shape
    out = mu.copy()
    # p-direction (each row)
    for i in range(G_u):
        out[i, :] = pava_inc(out[i, :])
    # u-direction (each column)
    for j in range(G_p):
        out[:, j] = pava_inc(out[:, j])
    return out


def phi_step_ld(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """One Φ-step in longdouble."""
    Gu = len(u_grid); Gp = p_grid.shape[1]
    mu_new = mu.copy()
    active = np.zeros((Gu, Gp), dtype=bool)
    f1_grid = f_v_ld(u_grid, 1, tau)
    f0_grid = f_v_ld(u_grid, 0, tau)
    for i in range(Gu):
        for j in range(Gp):
            p0 = p_grid[i, j]
            # Step A: extract μ_col
            mu_col = np.empty(Gu, dtype=LD)
            for ii in range(Gu):
                if p0 < p_grid[ii, 0]:
                    mu_col[ii] = mu[ii, 0]
                elif p0 > p_grid[ii, -1]:
                    mu_col[ii] = mu[ii, -1]
                else:
                    mu_col[ii] = interp_ld_1d(p0, p_grid[ii, :], mu[ii, :])
            mu_col = np.clip(mu_col, EPS_LD, LD(1) - EPS_LD)
            # Step B: demand
            p_arr = np.full(Gu, p0, dtype=LD)
            d = crra_demand_ld(mu_col, p_arr, gamma)
            # Monotonicity check
            if abs(d[-1] - d[0]) < LD(1e-15):
                continue
            # Step C: contour
            D_i = -d[i]
            targets = D_i - d
            # Inverse interp d → u₃*
            u3_star = np.empty(Gu, dtype=LD)
            for ii in range(Gu):
                target = targets[ii]
                if d[-1] > d[0]:
                    u3_star[ii] = interp_ld_1d(target, d, u_grid)
                else:
                    u3_star[ii] = interp_ld_1d(target, d[::-1], u_grid[::-1])
            valid = (u3_star >= u_grid[0]) & (u3_star <= u_grid[-1])
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue
            # Step D, E
            f1_root = f_v_ld(u3_star[valid], 1, tau)
            f0_root = f_v_ld(u3_star[valid], 0, tau)
            f1_sweep = f1_grid[valid]; f0_sweep = f0_grid[valid]
            A1 = np.sum(f1_sweep * f1_root)
            A0 = np.sum(f0_sweep * f0_root)
            f1_own = f1_grid[i]; f0_own = f0_grid[i]
            denom = f0_own * A0 + f1_own * A1
            if denom <= 0: continue
            mu_new[i, j] = np.clip(f1_own * A1 / denom,
                                      EPS_LD, LD(1) - EPS_LD)
            active[i, j] = True
    return mu_new, active


# Load G=15 strict ckpt and convert to longdouble
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu = LD(ck["mu"])
u_grid = LD(ck["u_grid"])
p_grid = LD(ck["p_grid"])
p_lo = LD(ck["p_lo"]); p_hi = LD(ck["p_hi"])

# Initial measure
cand, active = phi_step_ld(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F = np.abs(cand - mu)
res_max = float(F[active].max())
res_med = float(np.median(F[active]))
print(f"Initial (G=15 strict in longdouble): max={res_max:.3e}, med={res_med:.3e}",
      flush=True)

# Slow Picard in longdouble
N_PICARD = 5000
ALPHA = LD(0.001)
print(f"\nRunning {N_PICARD} Picard iters at α={float(ALPHA)} in longdouble...",
      flush=True)
t0 = time.time()
mu_sum = np.zeros_like(mu, dtype=LD); n_collected = 0
for it in range(N_PICARD):
    cand, active = phi_step_ld(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    cand = pava_2d_ld(cand)
    mu = ALPHA * cand + (LD(1) - ALPHA) * mu
    mu = np.clip(mu, EPS_LD, LD(1) - EPS_LD)
    if it >= N_PICARD // 2:
        mu_sum += mu; n_collected += 1
    if (it + 1) % 500 == 0:
        cand2, active2 = phi_step_ld(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        F2 = np.abs(cand2 - mu)
        rmax = float(F2[active2].max())
        rmed = float(np.median(F2[active2]))
        print(f"  iter {it+1}: max={rmax:.3e}, med={rmed:.3e}, "
              f"t={time.time()-t0:.0f}s", flush=True)
mu = pava_2d_ld(mu_sum / LD(n_collected))

cand, active = phi_step_ld(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F = np.abs(cand - mu)
res_max = float(F[active].max())
res_med = float(np.median(F[active]))
n_u = int((np.diff(mu, axis=0) < 0).sum())
n_p = int((np.diff(mu, axis=1) < 0).sum())
print(f"\nFINAL: max={res_max:.3e}, med={res_med:.3e}, u={n_u}, p={n_p}",
      flush=True)

# Convert back to float64 for measurement
mu_f64 = np.array(mu, dtype=np.float64)
np.savez(f"{RESULTS_DIR}/posterior_v3_G15_float128.npz",
         mu=mu_f64, u_grid=np.array(u_grid, dtype=np.float64),
         p_grid=np.array(p_grid, dtype=np.float64),
         p_lo=np.array(p_lo, dtype=np.float64),
         p_hi=np.array(p_hi, dtype=np.float64))
print(f"\nSaved float64 cast to {RESULTS_DIR}/posterior_v3_G15_float128.npz")
