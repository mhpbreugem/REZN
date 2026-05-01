"""Fig 3B with convex_contour smooth interpolant (no matplotlib contour).

For each p_target:
  1. Find crossings via 2-pass sweep on price slice
  2. Detect curvature sign
  3. Fit shape-constrained PCHIP interpolant
  4. Evaluate at many u_2 to get smooth (u_2, u_3) contour
"""
import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS
from convex_contour import (
    find_crossings, detect_curvature_sign, fit_convex_interpolant,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0; GAMMA = 0.5
P_LEVELS = [0.2, 0.3, 0.5, 0.7, 0.8]
P_ANNOTATE = [0.2, 0.5, 0.8]
u1_fixed = 1.0

# Use G=15 strict (cleanest μ)
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]


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
        return sum(crra_demand_vec(np.array([mu_at(u, p)]), np.array([p]),
                                      GAMMA)[0]
                   for u in (u1, u2, u3))
    try:
        return brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        return float("nan")


# Compute price slice on a moderate grid (G_eval) — use 30 to give good coverage
G_eval = 30
u_test = np.linspace(-3.5, 3.5, G_eval)
print(f"Computing P(1, u2, u3) on {G_eval}×{G_eval} grid...", flush=True)
t0 = time.time()
P_slice = np.full((G_eval, G_eval), np.nan)
for j, u2 in enumerate(u_test):
    for k, u3 in enumerate(u_test):
        p = market_clear(u1_fixed, u2, u3)
        if not np.isnan(p):
            P_slice[j, k] = p
print(f"  done in {time.time()-t0:.0f}s, "
      f"{(~np.isnan(P_slice)).sum()}/{P_slice.size} valid", flush=True)


def extract_contour_2pass(p_target):
    """2-pass sweep: get all (u_2, u_3) crossings, return."""
    u2_pts, u3_pts = [], []
    # Pass A: sweep u_2, find u_3 crossings
    for j in range(G_eval):
        row = P_slice[j, :]   # P(u2_fixed, u3 varies)
        crossings = find_crossings(row, u_test, p_target)
        for u3_c in crossings:
            u2_pts.append(u_test[j])
            u3_pts.append(u3_c)
    # Pass B: sweep u_3, find u_2 crossings
    for k in range(G_eval):
        col = P_slice[:, k]   # P(u2 varies, u3_fixed)
        crossings = find_crossings(col, u_test, p_target)
        for u2_c in crossings:
            u2_pts.append(u2_c)
            u3_pts.append(u_test[k])
    return np.array(u2_pts), np.array(u3_pts)


contours = {}
for p_target in P_LEVELS:
    u2_raw, u3_raw = extract_contour_2pass(p_target)
    if len(u2_raw) < 4:
        contours[f"{p_target:g}"] = []
        print(f"  p={p_target}: too few crossings ({len(u2_raw)})",
              flush=True)
        continue
    # Sort by u_2
    order = np.argsort(u2_raw)
    u2_sorted = u2_raw[order]
    u3_sorted = u3_raw[order]
    sign = detect_curvature_sign(u2_sorted, u3_sorted)
    interp = fit_convex_interpolant(u2_sorted, u3_sorted, sign)
    if interp is None:
        contours[f"{p_target:g}"] = list(zip(u2_sorted.tolist(),
                                                u3_sorted.tolist()))
        print(f"  p={p_target}: no interp, using raw {len(u2_sorted)} pts",
              flush=True)
        continue
    # Evaluate on smooth grid
    u2_eval = np.linspace(u2_sorted[0], u2_sorted[-1], 60)
    u3_eval = interp(u2_eval)
    pts = list(zip(u2_eval.tolist(), u3_eval.tolist()))
    contours[f"{p_target:g}"] = pts
    print(f"  p={p_target}: sign={sign:+d}, raw {len(u2_sorted)} → "
          f"smooth {len(pts)}", flush=True)

# Save
with open(f"{RESULTS_DIR}/fig_multicontour_B_convex_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_B_convex",
                "params": {"u1": u1_fixed, "tau": TAU, "gamma": GAMMA,
                           "G": 15, "G_eval": G_eval,
                           "method": "convex_contour PCHIP"},
                "p_levels": P_LEVELS,
                "contours": contours}, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_multicontour_B_convex_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        f.write(f"\\addplot coordinates {{{pgf(contours[f'{p:g}'])}}};\n\n")
print("Saved fig_multicontour_B_convex_*.{json,tex}", flush=True)

# Render
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.5)
        if p in P_ANNOTATE:
            mid = arr[len(arr) // 2]
            ax.annotate(f"$p={p}$", xy=mid, xytext=(mid[0] + 0.15, mid[1] + 0.15),
                          fontsize=8, color=color)
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Fig 3B: convex contours (REE)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, "
              f"G=15 strict, G_eval={G_eval}",
              fontsize=10)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_convex_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_convex_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved preview", flush=True)
