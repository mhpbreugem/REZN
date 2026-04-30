"""Plot REE-clearing price vs no-learning price as functions of T* = τΣu_k.

BC20 paper style: 8cm×8cm axes, red solid (REE) and blue dashed (no-learning).
Uses the monotone Cesaro-PAVA μ tensor at G=14, γ=0.5, τ=2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from posterior_method_v3 import Lam, crra_demand_vec, logit, EPS

# BC20 colors
RED = (0.7, 0.11, 0.11)
BLUE = (0.0, 0.20, 0.42)
GREEN = (0.11, 0.35, 0.02)

TAU = 2.0
GAMMA = 0.5

# Load converged monotone μ
d = np.load("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
mu = d["mu"]; u_grid = d["u_grid"]; p_grid = d["p_grid"]
p_lo = d["p_lo"]; p_hi = d["p_hi"]
G = mu.shape[0]
print(f"G={G}, u_grid range [{u_grid[0]:.2f}, {u_grid[-1]:.2f}]")


def market_clear(mus, gamma, lo=1e-6, hi=1-1e-6):
    """Bisection on Σ x_k(μ_k, p) = 0."""
    def Z(p):
        return sum(crra_demand_vec(np.array([m]), np.array([p]), gamma)[0]
                   for m in mus)
    if Z(lo) * Z(hi) > 0:
        return None
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if Z(m) > 0: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def mu_at(i, p, mu, p_grid, p_lo, p_hi, tau, u_grid):
    """Evaluate μ(u_i, p) by interpolating row i."""
    u_i = u_grid[i]
    if p < p_grid[i, 0]:
        return float(mu[i, 0])
    if p > p_grid[i, -1]:
        return float(mu[i, -1])
    return float(np.interp(p, p_grid[i, :], mu[i, :]))


# Sample (u_1, u_2, u_3) triples on the grid; compute REE p* and no-learning p*
T_list, p_REE, p_NL, w_list = [], [], [], []

for i in range(G):
    for j in range(G):
        for k in range(G):
            u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]

            # No-learning prices: μ_k = Λ(τu_k)
            mus_NL = [Lam(TAU * u) for u in (u1, u2, u3)]
            p_nl = market_clear(mus_NL, GAMMA)
            if p_nl is None:
                continue

            # REE prices: solve self-consistently using converged μ
            def Z_REE(p):
                m1 = mu_at(i, p, mu, p_grid, p_lo, p_hi, TAU, u_grid)
                m2 = mu_at(j, p, mu, p_grid, p_lo, p_hi, TAU, u_grid)
                m3 = mu_at(k, p, mu, p_grid, p_lo, p_hi, TAU, u_grid)
                return sum(crra_demand_vec(np.array([m]), np.array([p]), GAMMA)[0]
                           for m in (m1, m2, m3))
            lo, hi = 1e-6, 1 - 1e-6
            if Z_REE(lo) * Z_REE(hi) > 0:
                continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if Z_REE(mid) > 0: lo = mid
                else: hi = mid
            p_ree = 0.5 * (lo + hi)

            # T* = τ(u1+u2+u3)
            T = TAU * (u1 + u2 + u3)
            # weight: triple ex-ante prob
            f1 = lambda u: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u-0.5)**2)
            f0 = lambda u: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u+0.5)**2)
            w = 0.5 * (f1(u1)*f1(u2)*f1(u3) + f0(u1)*f0(u2)*f0(u3))
            T_list.append(T); p_REE.append(p_ree); p_NL.append(p_nl)
            w_list.append(w)

T_arr = np.array(T_list); p_REE = np.array(p_REE); p_NL = np.array(p_NL)
w_arr = np.array(w_list)

# Sort by T for plotting smooth curves
order = np.argsort(T_arr)
T_sorted = T_arr[order]
pR_sorted = p_REE[order]
pN_sorted = p_NL[order]
w_sorted = w_arr[order]

# To get a smooth curve, bin by T and take weighted average
n_bins = 60
T_bins = np.linspace(T_sorted.min(), T_sorted.max(), n_bins + 1)
T_centers, p_REE_b, p_NL_b = [], [], []
for k in range(n_bins):
    mask = (T_sorted >= T_bins[k]) & (T_sorted < T_bins[k+1])
    if mask.sum() < 1: continue
    ww = w_sorted[mask]; wn = ww.sum()
    if wn == 0: continue
    T_centers.append(0.5 * (T_bins[k] + T_bins[k+1]))
    p_REE_b.append((pR_sorted[mask] * ww).sum() / wn)
    p_NL_b.append((pN_sorted[mask] * ww).sum() / wn)
T_centers = np.array(T_centers)
p_REE_b = np.array(p_REE_b)
p_NL_b = np.array(p_NL_b)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))   # 8cm × 8cm

# REE: red solid very thick
ax.plot(T_centers, p_REE_b, color=RED, lw=2.0, solid_capstyle="round",
        label="REE (CRRA, $\\gamma=0.5$)")
# No-learning: blue dashed very thick
ax.plot(T_centers, p_NL_b, color=BLUE, lw=2.0, ls="--", dash_capstyle="round",
        label="no learning")
# Reference lines: p = Lam(T) (FR / CARA full revelation)
T_ref = np.linspace(T_arr.min(), T_arr.max(), 200)
ax.plot(T_ref, Lam(T_ref), color="black", lw=1.0, ls=":", label="$p=\\Lambda(T^*)$ (FR)")

ax.set_xlabel("$T^* = \\tau(u_1+u_2+u_3)$")
ax.set_ylabel("price $p$")
ax.set_xlim(T_arr.min(), T_arr.max())
ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False, fontsize=8)
ax.grid(alpha=0.25, lw=0.5)
ax.set_title(f"REE vs no-learning price\n"
             f"$G={G}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, "
             f"1-$R^2$=0.097", fontsize=9)

plt.tight_layout()
out_pdf = "results/full_ree/posterior_v3_G14_REE_vs_nolearning.pdf"
out_png = "results/full_ree/posterior_v3_G14_REE_vs_nolearning.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")

# Report slopes for sanity
def weighted_slope(T, p, w):
    Y = logit(p); X = T
    W = w / w.sum()
    Yb = (W*Y).sum(); Xb = (W*X).sum()
    cov = (W*(Y-Yb)*(X-Xb)).sum()
    vx = (W*(X-Xb)**2).sum()
    return cov / vx if vx > 0 else 0

s_REE = weighted_slope(T_arr, p_REE, w_arr)
s_NL  = weighted_slope(T_arr, p_NL, w_arr)
print(f"slope of logit(p) on T*:")
print(f"  REE: {s_REE:.4f}")
print(f"  no-learning: {s_NL:.4f}")
print(f"  FR (CARA): 1.0000")
