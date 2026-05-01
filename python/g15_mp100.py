"""G=15 polish in arbitrary precision (dps=100) using mpmath.

Bypasses float64 epsilon floor entirely. Each phi_step is slow (~30s-2min)
because every operation (exp, log, interp) is 100-digit. Expected to run
for hours.

Saves intermediate states every 10 iterations.
"""
import time, json, os, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 100   # 100 decimal digits

RESULTS_DIR = "results/full_ree"
G = 15
TAU = mpf("2")
GAMMA = mpf("0.5")


def Lam_mp(z):
    """Sigmoid in mpmath."""
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
    """Linear interp in mpmath; x_arr sorted ascending."""
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
    """Φ-step in mpmath. mu, u_grid, p_grid are 2D/1D lists of mpf."""
    Gu = len(u_grid); Gp = len(p_grid[0])
    EPS_MP = mpf("1e-50")
    mu_new = [[mu[i][j] for j in range(Gp)] for i in range(Gu)]
    f1_grid = [f_v_mp(u_grid[i], 1, tau) for i in range(Gu)]
    f0_grid = [f_v_mp(u_grid[i], 0, tau) for i in range(Gu)]
    for i in range(Gu):
        for j in range(Gp):
            p0 = p_grid[i][j]
            mu_col = []
            for ii in range(Gu):
                if p0 < p_grid[ii][0]:
                    val = mu[ii][0]
                elif p0 > p_grid[ii][-1]:
                    val = mu[ii][-1]
                else:
                    val = interp_mp(p0, p_grid[ii], mu[ii])
                if val < EPS_MP: val = EPS_MP
                if val > mpf(1) - EPS_MP: val = mpf(1) - EPS_MP
                mu_col.append(val)
            d = [crra_demand_mp(mu_col[ii], p0, gamma) for ii in range(Gu)]
            if abs(d[-1] - d[0]) < mpf("1e-30"):
                continue
            D_i = -d[i]
            targets = [D_i - d[ii] for ii in range(Gu)]
            d_inc = d[-1] > d[0]
            d_arr = d if d_inc else list(reversed(d))
            u_arr = u_grid if d_inc else list(reversed(u_grid))
            u3_star = []
            valid_mask = []
            for ii in range(Gu):
                if targets[ii] < d_arr[0] or targets[ii] > d_arr[-1]:
                    u3_star.append(None)
                    valid_mask.append(False)
                else:
                    u3 = interp_mp(targets[ii], d_arr, u_arr)
                    u3_star.append(u3)
                    valid_mask.append(u_grid[0] <= u3 <= u_grid[-1])
            valid = [k for k in range(Gu) if valid_mask[k]]
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


def measure_mp(mu, mu_new):
    """Compute max and median |mu_new - mu|."""
    diffs = []
    for i in range(len(mu)):
        for j in range(len(mu[0])):
            diffs.append(abs(mu_new[i][j] - mu[i][j]))
    diffs.sort()
    n = len(diffs)
    return diffs[-1], diffs[n // 2]


# Load G=15 strict from float64 ckpt
print(f"Loading G=15 strict μ (float64) → mpf, dps={mp.dps}", flush=True)
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu = [[mpf(str(ck["mu"][i, j])) for j in range(G)] for i in range(G)]
u_grid = [mpf(str(ck["u_grid"][i])) for i in range(G)]
p_grid = [[mpf(str(ck["p_grid"][i, j])) for j in range(G)] for i in range(G)]
p_lo = [mpf(str(ck["p_lo"][i])) for i in range(G)]
p_hi = [mpf(str(ck["p_hi"][i])) for i in range(G)]

print("Initial measure...", flush=True)
t0 = time.time()
cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
m_max, m_med = measure_mp(mu, cand)
print(f"Initial: max={mpmath.nstr(m_max, 5)}, med={mpmath.nstr(m_med, 5)}, "
      f"phi_step took {time.time()-t0:.1f}s", flush=True)

# Picard with very small damping
ALPHA = mpf("0.05")
N_ITER = 100
print(f"\nRunning {N_ITER} Picard iters at α={float(ALPHA)}...", flush=True)
for it in range(1, N_ITER + 1):
    t_iter = time.time()
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    m_max, m_med = measure_mp(mu, cand)
    # Damped update
    mu_new = [[(mpf(1) - ALPHA) * mu[i][j] + ALPHA * cand[i][j]
                  for j in range(G)] for i in range(G)]
    mu = mu_new
    print(f"  iter {it:>3}: max={mpmath.nstr(m_max, 5)}, "
          f"med={mpmath.nstr(m_med, 5)}, t={time.time()-t_iter:.1f}s",
          flush=True)
    # Save every 10 iters
    if it % 10 == 0:
        # Save as strings (preserves precision)
        mu_strs = [[mpmath.nstr(mu[i][j], 100) for j in range(G)]
                    for i in range(G)]
        with open(f"{RESULTS_DIR}/posterior_v3_G15_mp100_iter{it}.json",
                   "w") as f:
            json.dump({"iter": it,
                        "max_residual": mpmath.nstr(m_max, 50),
                        "med_residual": mpmath.nstr(m_med, 50),
                        "mu_strings": mu_strs,
                        "dps": mp.dps},
                       f, indent=1)
        print(f"  Saved iter {it} state", flush=True)

# Final
mu_strs = [[mpmath.nstr(mu[i][j], 100) for j in range(G)]
            for i in range(G)]
with open(f"{RESULTS_DIR}/posterior_v3_G15_mp100_final.json", "w") as f:
    json.dump({"iter": N_ITER,
                "max_residual": mpmath.nstr(m_max, 50),
                "med_residual": mpmath.nstr(m_med, 50),
                "mu_strings": mu_strs,
                "dps": mp.dps}, f, indent=1)
print(f"\nFinal saved. max={mpmath.nstr(m_max, 10)}", flush=True)
