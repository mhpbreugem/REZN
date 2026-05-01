"""Fig 3B high-res: 200x200 grid, market-clearing at each fine-grid point.
Levels: p ∈ {0.2, 0.3, 0.5, 0.7, 0.8}. Thin to ~50 points per contour.
"""
import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0; GAMMA = 0.5
P_LEVELS = [0.2, 0.3, 0.5, 0.7, 0.8]
u1_fixed = 1.0

ck = np.load(f"{RESULTS_DIR}/posterior_v3_trim90_G20.npz")
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


N = 200
u_test = np.linspace(-3.5, 3.5, N)
print(f"Computing P(1, u2, u3) on {N}×{N} grid...", flush=True)
t0 = time.time()
P_slice = np.full((N, N), np.nan)
for j, u2 in enumerate(u_test):
    for k, u3 in enumerate(u_test):
        p = market_clear(u1_fixed, u2, u3)
        if not np.isnan(p):
            P_slice[j, k] = p
    if (j + 1) % 25 == 0:
        print(f"  {j+1}/{N} rows done, t={time.time()-t0:.0f}s", flush=True)
print(f"Done: {(~np.isnan(P_slice)).sum()}/{P_slice.size} valid in "
      f"{time.time()-t0:.0f}s", flush=True)

# Extract contours
fig_tmp, ax_tmp = plt.subplots()
cs = ax_tmp.contour(u_test, u_test, P_slice.T, levels=P_LEVELS)
plt.close(fig_tmp)


def thin_contour(pts, n_target=50):
    """Thin a contour to roughly n_target points by evenly sampling arc length."""
    if len(pts) <= n_target:
        return pts
    arr = np.array(pts)
    dists = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
    cum_arc = np.concatenate([[0], np.cumsum(dists)])
    total = cum_arc[-1]
    if total == 0:
        return pts
    targets = np.linspace(0, total, n_target)
    thinned = []
    for t in targets:
        idx = np.searchsorted(cum_arc, t)
        idx = min(idx, len(arr) - 1)
        thinned.append((float(arr[idx, 0]), float(arr[idx, 1])))
    return thinned


CRRA_contours = {}
for k, p in enumerate(P_LEVELS):
    pts_all = []
    if k < len(cs.allsegs):
        for c in cs.allsegs[k]:
            for x, y in c:
                pts_all.append((float(x), float(y)))
    thinned = thin_contour(pts_all, n_target=50)
    CRRA_contours[f"{p:g}"] = thinned
    print(f"  p={p}: raw {len(pts_all)} → thinned {len(thinned)}", flush=True)

# Save
with open(f"{RESULTS_DIR}/fig_multicontour_B_hires_G20_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_B_hires_G20",
                "params": {"u1": u1_fixed, "tau": TAU, "gamma": GAMMA,
                           "model": "CRRA_REE", "G": 20, "trim": 0.05,
                           "N_grid": N},
                "p_levels": P_LEVELS,
                "contours": CRRA_contours}, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_multicontour_B_hires_G20_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        f.write(f"\\addplot coordinates "
                f"{{{pgf(CRRA_contours[f'{p:g}'])}}};\n\n")
print("Saved fig_multicontour_B_hires_G20_*.{json,tex}", flush=True)

# Render preview
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
P_ANNOTATE = [0.2, 0.5, 0.8]
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = CRRA_contours[f"{p:g}"]
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
ax.set_title(f"Fig 3B hires: CRRA contours (REE)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, "
              f"$G=20$ (90% coverage), $N={N}^2$",
              fontsize=10)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_hires_G20_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_hires_G20_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved preview", flush=True)
