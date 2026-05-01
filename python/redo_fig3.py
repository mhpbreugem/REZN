"""Smoother Fig 3A and 3B with annotations only at p=0.2, 0.5, 0.8."""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


P_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
P_ANNOTATE = [0.2, 0.5, 0.8]   # only these get labels
u1_fixed = 1.0


# =========================================================================
# Fig 3A: CARA multicontour (smooth analytical)
# =========================================================================
print("Fig 3A: CARA multicontour (smooth analytical)", flush=True)
# Analytical: u_2 + u_3 = (T*/τ) - u_1 = (3 logit(p)/τ) - u_1
# Use 200 points for smoothness
u_smooth = np.linspace(-3.5, 3.5, 200)
CARA_contours = {}
for p_target in P_LEVELS:
    T_target = 3 * np.log(p_target / (1 - p_target))
    u2u3_sum = T_target / TAU - u1_fixed
    pts = []
    for u2 in u_smooth:
        u3 = u2u3_sum - u2
        if -3.5 <= u3 <= 3.5:
            pts.append((float(u2), float(u3)))
    CARA_contours[f"{p_target:g}"] = pts

with open(f"{RESULTS_DIR}/fig_multicontour_A_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_A",
                "params": {"u1": u1_fixed, "tau": TAU,
                           "model": "CARA_no_learning"},
                "p_levels": P_LEVELS,
                "contours": CARA_contours}, f, indent=2)
with open(f"{RESULTS_DIR}/fig_multicontour_A_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        f.write(f"\\addplot coordinates {{{pgf(CARA_contours[f'{p:g}'])}}};\n\n")

# Render: only annotate p=0.2, 0.5, 0.8 with text label inside plot
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
for k, p in enumerate(P_LEVELS):
    color = cmap(k / max(len(P_LEVELS) - 1, 1))
    pts = CARA_contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.5)
        # Annotate only the special ones
        if p in P_ANNOTATE:
            # Place label at midpoint
            mid = arr[len(arr) // 2]
            ax.annotate(f"$p={p}$", xy=mid, xytext=(mid[0] + 0.15, mid[1] + 0.15),
                          fontsize=8, color=color)
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Fig 3A: CARA contour lines (no learning)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$",
              fontsize=10)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_A_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_A_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved Fig 3A", flush=True)


# =========================================================================
# Fig 3B: CRRA multicontour (smoother grid)
# =========================================================================
print("Fig 3B: CRRA multicontour (smoother grid)", flush=True)
GAMMA = 0.5
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


def market_clear_REE(u1, u2, u3):
    def Z(p):
        return sum(crra_demand_vec(np.array([mu_at(u, p)]), np.array([p]),
                                      GAMMA)[0]
                   for u in (u1, u2, u3))
    try:
        return brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        return None


u_test = np.linspace(-3.5, 3.5, 100)   # 100 vs 50 — denser grid
print(f"  Computing P(1, u2, u3) on {len(u_test)}^2 grid...", flush=True)
P_slice = np.full((len(u_test), len(u_test)), np.nan)
for j, u2 in enumerate(u_test):
    for k, u3 in enumerate(u_test):
        p = market_clear_REE(u1_fixed, u2, u3)
        if p is not None:
            P_slice[j, k] = p
print(f"  Done", flush=True)

fig_tmp, ax_tmp = plt.subplots()
cs = ax_tmp.contour(u_test, u_test, P_slice.T, levels=P_LEVELS)
plt.close(fig_tmp)

CRRA_contours = {}
for k, p in enumerate(P_LEVELS):
    pts = []
    if k < len(cs.allsegs):
        for c in cs.allsegs[k]:
            pts.extend([(float(x), float(y)) for x, y in c])
    CRRA_contours[f"{p:g}"] = pts

with open(f"{RESULTS_DIR}/fig_multicontour_B_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_B",
                "params": {"u1": u1_fixed, "tau": TAU, "gamma": GAMMA,
                           "model": "CRRA_REE", "G": 15},
                "p_levels": P_LEVELS,
                "contours": CRRA_contours}, f, indent=2)
with open(f"{RESULTS_DIR}/fig_multicontour_B_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        f.write(f"\\addplot coordinates {{{pgf(CRRA_contours[f'{p:g}'])}}};\n\n")

# Render
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
for k, p in enumerate(P_LEVELS):
    color = cmap(k / max(len(P_LEVELS) - 1, 1))
    pts = CRRA_contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        # Sort points to draw smooth lines (in case extraction returned scattered)
        # Just plot raw — matplotlib contour returns ordered segs
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.5)
        if p in P_ANNOTATE:
            mid = arr[len(arr) // 2]
            ax.annotate(f"$p={p}$", xy=mid, xytext=(mid[0] + 0.15, mid[1] + 0.15),
                          fontsize=8, color=color)
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Fig 3B: CRRA contour lines (REE)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, $G=15$",
              fontsize=10)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved Fig 3B", flush=True)
