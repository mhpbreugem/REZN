"""Combined contour plot PDF: CARA (straight lines) + CRRA (convex_contour fits)
side-by-side, BC20 style.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/full_ree"
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)
P_LEVELS = [0.2, 0.3, 0.5, 0.7, 0.8]
P_ANNOTATE = [0.2, 0.5, 0.8]

with open(f"{RESULTS_DIR}/fig_multicontour_A_data.json") as f:
    cara = json.load(f)
with open(f"{RESULTS_DIR}/fig_multicontour_B_convex_data.json") as f:
    crra = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(15/2.54, 8/2.54), sharex=True, sharey=True)
cmap = plt.cm.viridis

# Left: CARA
ax = axes[0]
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = cara["contours"].get(f"{p:g}", [])
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
ax.set_title(f"(a) CARA: contours are straight lines\n"
              f"$u_1={cara['params']['u1']}$, $\\tau=2$",
              fontsize=10)

# Right: CRRA
ax = axes[1]
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = crra["contours"].get(f"{p:g}", [])
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
ax.set_title(f"(b) CRRA REE: contours are curved\n"
              f"$u_1={crra['params']['u1']}$, $\\tau=2$, $\\gamma={crra['params']['gamma']}$, $G=15$",
              fontsize=10)

plt.tight_layout()
out = f"{RESULTS_DIR}/fig_contours_combined"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}.{{pdf,png}}")
