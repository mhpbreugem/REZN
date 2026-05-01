"""Compute 1-R² and slope from converged mp100 iter 2 μ.

Standalone — no imports from g15_mp100 (which has top-level code).
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 100

RESULTS_DIR = "results/full_ree"
G = 15
TAU = mpf("2")
GAMMA = mpf("0.5")


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


# Load mp100 iter 2 + grid
print("Loading mp100 iter 2...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G15_mpNK_iter2.json") as f:
    state = json.load(f)
mu_arr = [[mpf(state["mu_strings"][i][j]) for j in range(G)]
           for i in range(G)]
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
u_grid = [mpf(str(ck["u_grid"][i])) for i in range(G)]
p_grid = [[mpf(str(ck["p_grid"][i, j])) for j in range(G)] for i in range(G)]


def mu_at_uv(u, p):
    if u <= u_grid[0]: idx_b, idx_a, w = 0, 0, mpf(1)
    elif u >= u_grid[-1]: idx_b = idx_a = G - 1; w = mpf(1)
    else:
        for k in range(1, G):
            if u_grid[k] >= u:
                idx_a = k; idx_b = k - 1
                w = (u - u_grid[idx_b]) / (u_grid[idx_a] - u_grid[idx_b])
                break
    p_b = p
    if p_b < p_grid[idx_b][0]: p_b = p_grid[idx_b][0]
    if p_b > p_grid[idx_b][-1]: p_b = p_grid[idx_b][-1]
    p_a = p
    if p_a < p_grid[idx_a][0]: p_a = p_grid[idx_a][0]
    if p_a > p_grid[idx_a][-1]: p_a = p_grid[idx_a][-1]
    mu_b = interp_mp(p_b, p_grid[idx_b], mu_arr[idx_b])
    mu_a = interp_mp(p_a, p_grid[idx_a], mu_arr[idx_a])
    return (mpf(1) - w) * mu_b + w * mu_a


def market_clear(u1, u2, u3):
    def Z(p):
        return (
            crra_demand_mp(mu_at_uv(u1, p), p, GAMMA)
            + crra_demand_mp(mu_at_uv(u2, p), p, GAMMA)
            + crra_demand_mp(mu_at_uv(u3, p), p, GAMMA))
    lo = mpf("1e-10"); hi = mpf(1) - mpf("1e-10")
    Z_lo = Z(lo); Z_hi = Z(hi)
    if Z_lo * Z_hi >= 0:
        return None
    for _ in range(150):
        mid = (lo + hi) / mpf(2)
        Z_mid = Z(mid)
        if Z_mid > 0: lo = mid
        else: hi = mid
        if hi - lo < mpf("1e-80"): break
    return (lo + hi) / mpf(2)


print(f"Computing 1-R² over {G**3} triples in mp100...", flush=True)
t0 = time.time()
Y, X, W = [], [], []
n_done = 0
for i in range(G):
    for j in range(G):
        for k in range(G):
            u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]
            p_star = market_clear(u1, u2, u3)
            if p_star is None:
                continue
            T = TAU * (u1 + u2 + u3)
            f1_p = f_v_mp(u1, 1, TAU) * f_v_mp(u2, 1, TAU) * f_v_mp(u3, 1, TAU)
            f0_p = f_v_mp(u1, 0, TAU) * f_v_mp(u2, 0, TAU) * f_v_mp(u3, 0, TAU)
            w = (f1_p + f0_p) / mpf(2)
            Y.append(logit_mp(p_star)); X.append(T); W.append(w)
            n_done += 1
    if (i + 1) % 3 == 0:
        elapsed = time.time() - t0
        eta = (G - i - 1) * elapsed / (i + 1)
        print(f"  i={i+1}/{G}, done {n_done}/{G**3}, "
              f"t={elapsed:.0f}s (eta {eta:.0f}s)", flush=True)

print(f"\nTotal triples: {n_done} in {time.time()-t0:.0f}s", flush=True)
print("Weighted regression in mp100...", flush=True)
W_total = sum(W)
W_norm = [w / W_total for w in W]
Yb = sum(W_norm[k] * Y[k] for k in range(n_done))
Xb = sum(W_norm[k] * X[k] for k in range(n_done))
cov = sum(W_norm[k] * (Y[k] - Yb) * (X[k] - Xb) for k in range(n_done))
vy = sum(W_norm[k] * (Y[k] - Yb) ** 2 for k in range(n_done))
vx = sum(W_norm[k] * (X[k] - Xb) ** 2 for k in range(n_done))
R2 = cov ** 2 / (vy * vx)
slope = cov / vx
one_minus_R2 = mpf(1) - R2

print(f"\n=== HIGH-PRECISION (G=15, γ=0.5, τ=2, mp100) ===")
print(f"  1-R² = {mpmath.nstr(one_minus_R2, 50)}")
print(f"  slope = {mpmath.nstr(slope, 50)}")

with open(f"{RESULTS_DIR}/posterior_v3_G15_mp100_R2.json", "w") as f:
    json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "dps": mp.dps,
                "1-R^2": mpmath.nstr(one_minus_R2, 60),
                "slope": mpmath.nstr(slope, 60),
                "n_triples": n_done,
                "n_total": G**3}, f, indent=2)
print(f"\nSaved {RESULTS_DIR}/posterior_v3_G15_mp100_R2.json")
