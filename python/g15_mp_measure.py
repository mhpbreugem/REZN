"""Compute 1-R² and slope from converged mp100 iter 2 μ at high precision.

For each grid triple (u_i, u_j, u_l):
  1. Solve market clearing in mp100 for p*
  2. Record (logit(p*), T*=τ·Σu, weight)
Then weighted regression in mp100.
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

from g15_mp100 import (
    Lam_mp, logit_mp, crra_demand_mp, f_v_mp, interp_mp,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 100

RESULTS_DIR = "results/full_ree"
G = 15
TAU = mpf("2")
GAMMA = mpf("0.5")

# Load converged mp100 iter 2
print("Loading mp100 iter 2 result...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G15_mpNK_iter2.json") as f:
    state = json.load(f)
mu_arr = [[mpf(state["mu_strings"][i][j]) for j in range(G)]
           for i in range(G)]

# Load grid from float64
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
u_grid = [mpf(str(ck["u_grid"][i])) for i in range(G)]
p_grid = [[mpf(str(ck["p_grid"][i, j])) for j in range(G)] for i in range(G)]
p_lo = [mpf(str(ck["p_lo"][i])) for i in range(G)]
p_hi = [mpf(str(ck["p_hi"][i])) for i in range(G)]


def mu_at_uv(u, p):
    """Bivariate interp of μ at (u, p)."""
    if u <= u_grid[0]: idx_b, idx_a, w = 0, 0, mpf(1)
    elif u >= u_grid[-1]: idx_b = idx_a = G - 1; w = mpf(1)
    else:
        for k in range(1, G):
            if u_grid[k] >= u:
                idx_a = k; idx_b = k - 1
                w = (u - u_grid[idx_b]) / (u_grid[idx_a] - u_grid[idx_b])
                break
    p_clamped_b = p
    if p_clamped_b < p_grid[idx_b][0]: p_clamped_b = p_grid[idx_b][0]
    if p_clamped_b > p_grid[idx_b][-1]: p_clamped_b = p_grid[idx_b][-1]
    p_clamped_a = p
    if p_clamped_a < p_grid[idx_a][0]: p_clamped_a = p_grid[idx_a][0]
    if p_clamped_a > p_grid[idx_a][-1]: p_clamped_a = p_grid[idx_a][-1]
    mu_b = interp_mp(p_clamped_b, p_grid[idx_b], mu_arr[idx_b])
    mu_a = interp_mp(p_clamped_a, p_grid[idx_a], mu_arr[idx_a])
    return (mpf(1) - w) * mu_b + w * mu_a


def market_clear(u1, u2, u3):
    """Bisection in mpmath for market-clearing price."""
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
        if Z_mid > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < mpf("1e-80"):
            break
    return (lo + hi) / mpf(2)


print(f"Computing 1-R² over {G**3}={G**3} triples in mp100...", flush=True)
t0 = time.time()
Y = []; X = []; W = []
n_done = 0
for i in range(G):
    for j in range(G):
        for k in range(G):
            u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]
            p_star = market_clear(u1, u2, u3)
            if p_star is None:
                continue
            T = TAU * (u1 + u2 + u3)
            f1_prod = f_v_mp(u1, 1, TAU) * f_v_mp(u2, 1, TAU) * f_v_mp(u3, 1, TAU)
            f0_prod = f_v_mp(u1, 0, TAU) * f_v_mp(u2, 0, TAU) * f_v_mp(u3, 0, TAU)
            w = (f1_prod + f0_prod) / mpf(2)
            Y.append(logit_mp(p_star))
            X.append(T)
            W.append(w)
            n_done += 1
    if (i + 1) % 3 == 0:
        elapsed = time.time() - t0
        eta = (G - i - 1) * elapsed / (i + 1)
        print(f"  i={i+1}/{G}, done {n_done}/{G**3}, "
              f"t={elapsed:.0f}s (eta {eta:.0f}s)", flush=True)

print(f"\nTotal triples computed: {n_done} in {time.time()-t0:.0f}s",
      flush=True)

# Weighted regression in mp100
print("Computing weighted regression in mp100...", flush=True)
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

print(f"\n=== HIGH-PRECISION RESULTS (G=15, γ=0.5, τ=2, mp100) ===")
print(f"  1-R² = {mpmath.nstr(one_minus_R2, 50)}")
print(f"  slope = {mpmath.nstr(slope, 50)}")
print(f"  Number of triples: {n_done}/{G**3}")
print(f"  Total time: {time.time()-t0:.0f}s")

with open(f"{RESULTS_DIR}/posterior_v3_G15_mp100_R2.json", "w") as f:
    json.dump({"G": G, "tau": 2.0, "gamma": 0.5, "dps": mp.dps,
                "1-R^2": mpmath.nstr(one_minus_R2, 60),
                "slope": mpmath.nstr(slope, 60),
                "n_triples": n_done,
                "n_total": G**3}, f, indent=2)
print(f"\nSaved {RESULTS_DIR}/posterior_v3_G15_mp100_R2.json")
