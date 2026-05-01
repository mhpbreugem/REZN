"""Contour plot using mp100-converged μ (iter 2)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS
from convex_contour import (
    find_crossings, detect_curvature_sign, fit_convex_interpolant,
)

RESULTS_DIR = "results/full_ree"
TAU = 2.0; GAMMA = 0.5
P_LEVELS = [0.2, 0.3, 0.5, 0.7, 0.8]
P_ANNOTATE = [0.2, 0.5, 0.8]
u1_fixed = 1.0
G = 15

# Load mp100 iter 2 μ (converted to float64 for plotting)
print("Loading mp100 iter 2 μ...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G15_mpNK_iter2.json") as f:
    state = json.load(f)
mu_REE = np.array([[float(state["mu_strings"][i][j]) for j in range(G)]
                       for i in range(G)])
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
u_grid_R = ck["u_grid"]
p_grid_R = ck["p_grid"]


def mu_at(u, p):
    if u <= u_grid_R[0]: idx = 0; w = 0.0
    elif u >= u_grid_R[-1]: idx = len(u_grid_R) - 1; w = 1.0
    else:
        ia = np.searchsorted(u_grid_R, u); ib = ia - 1
        w = (u - u_grid_R[ib]) / (u_grid_R[ia] - u_grid_R[ib])
        p_b = np.clip(p, p_grid_R[ib, 0], p_grid_R[ib, -1])
        m_b = np.interp(p_b, p_grid_R[ib, :], mu_REE[ib, :])
        p_a = np.clip(p, p_grid_R[ia, 0], p_grid_R[ia, -1])
        m_a = np.interp(p_a, p_grid_R[ia, :], mu_REE[ia, :])
        return (1 - w) * m_b + w * m_a
    p_c = np.clip(p, p_grid_R[idx, 0], p_grid_R[idx, -1])
    return float(np.interp(p_c, p_grid_R[idx, :], mu_REE[idx, :]))


def market_clear(u1, u2, u3):
    def Z(p):
        return sum(crra_demand_vec(np.array([mu_at(u, p)]),
                                      np.array([p]), GAMMA)[0]
                   for u in (u1, u2, u3))
    try:
        return brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        return float("nan")


# Compute price slice at u1=1
G_eval = 30
u_test = np.linspace(-3.5, 3.5, G_eval)
print(f"Computing P(1, u2, u3) on {G_eval}×{G_eval} grid...", flush=True)
import time
t0 = time.time()
P_slice = np.full((G_eval, G_eval), np.nan)
for j, u2 in enumerate(u_test):
    for k, u3 in enumerate(u_test):
        p = market_clear(u1_fixed, u2, u3)
        if not np.isnan(p):
            P_slice[j, k] = p
print(f"  {(~np.isnan(P_slice)).sum()}/{P_slice.size} valid in "
      f"{time.time()-t0:.0f}s", flush=True)


def extract_2pass(p_target):
    u2p, u3p = [], []
    for j in range(G_eval):
        for u3_c in find_crossings(P_slice[j, :], u_test, p_target):
            u2p.append(u_test[j]); u3p.append(u3_c)
    for k in range(G_eval):
        for u2_c in find_crossings(P_slice[:, k], u_test, p_target):
            u2p.append(u2_c); u3p.append(u_test[k])
    return np.array(u2p), np.array(u3p)


contours = {}
for p_target in P_LEVELS:
    u2_raw, u3_raw = extract_2pass(p_target)
    if len(u2_raw) < 4:
        contours[f"{p_target:g}"] = []
        continue
    order = np.argsort(u2_raw)
    u2_s = u2_raw[order]; u3_s = u3_raw[order]
    sign = detect_curvature_sign(u2_s, u3_s)
    interp = fit_convex_interpolant(u2_s, u3_s, sign)
    if interp is None:
        contours[f"{p_target:g}"] = list(zip(u2_s.tolist(), u3_s.tolist()))
        continue
    u2_eval = np.linspace(u2_s[0], u2_s[-1], 80)
    u3_eval = interp(u2_eval)
    contours[f"{p_target:g}"] = list(zip(u2_eval.tolist(), u3_eval.tolist()))
    print(f"  p={p_target}: sign={sign:+d}, raw {len(u2_s)} → smooth 80",
          flush=True)


# Render
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.7)
        if p in P_ANNOTATE:
            mid = arr[len(arr) // 2]
            ax.annotate(f"$p={p}$", xy=mid, xytext=(mid[0] + 0.15, mid[1] + 0.15),
                          fontsize=9, color=color)
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"CRRA REE contours (mp100 Newton FP)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, "
              f"G=15, max-resid 1.4e-55",
              fontsize=10)
plt.tight_layout()
out = f"{RESULTS_DIR}/fig_contours_mpNK"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}.{{pdf,png}}", flush=True)

with open(f"{RESULTS_DIR}/fig_contours_mpNK_data.json", "w") as f:
    json.dump({"figure": "fig_contours_mpNK",
                "params": {"u1": u1_fixed, "tau": TAU, "gamma": GAMMA,
                           "G": 15, "source": "mp100 Newton iter 2",
                           "F_max": "1.37e-55"},
                "p_levels": P_LEVELS,
                "contours": contours}, f, indent=2)
