"""Plot the monotone Cesaro-averaged μ at G=14, γ=0.5, τ=2."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load monotone μ
d = np.load("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
mu = d["mu"]
u_grid = d["u_grid"]
p_grid = d["p_grid"]
p_lo = d["p_lo"]
p_hi = d["p_hi"]
G = mu.shape[0]

# Also load non-monotone NK FP for comparison
d2 = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu_nm = d2["mu"]
p_grid_nm = d2["p_grid"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# --- Left: monotone μ heatmap ---
ax = axes[0]
# Reshape mu to be plotted in (u, p)-space.
# Each row of mu has its own p-grid (per-row); show as a quadrilateral mesh.
# Use pcolormesh on the lattice (u_grid, p_grid[i,j]).
# Build coordinate arrays
U_corners = np.linspace(u_grid[0], u_grid[-1], G + 1)
# For each row, build p-corners; we'll just use pcolormesh per-row strip
for i in range(G):
    p_row = p_grid[i, :]
    p_corners = np.concatenate([
        [1.5 * p_row[0] - 0.5 * p_row[1]],
        0.5 * (p_row[:-1] + p_row[1:]),
        [1.5 * p_row[-1] - 0.5 * p_row[-2]],
    ])
    u_corners = [U_corners[i], U_corners[i+1]]
    P, U = np.meshgrid(p_corners, u_corners)
    Z = mu[i:i+1, :]
    pc = ax.pcolormesh(U, P, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="flat")
ax.set_xlabel("private signal u")
ax.set_ylabel("price p")
ax.set_title(f"monotone μ(u, p) — Cesaro-averaged Picard-PAVA\n"
             f"G={G}, γ=0.5, τ=2,  1-R²=0.0967, slope=0.337")
ax.set_ylim(0, 1)
plt.colorbar(pc, ax=ax, label="μ")

# --- Middle: per-row curves μ(u_i, p) vs p ---
ax = axes[1]
cmap = plt.cm.viridis
for i in range(G):
    color = cmap(i / max(G - 1, 1))
    ax.plot(p_grid[i, :], mu[i, :], color=color, lw=1.5,
            label=f"u={u_grid[i]:+.2f}" if i % 3 == 0 else None)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1, label="μ=p (CARA/FR)")
ax.set_xlabel("price p")
ax.set_ylabel("posterior μ(u, p)")
ax.set_title("μ-columns by signal u (monotone)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.3)

# --- Right: comparison monotone vs non-monotone (line plot of mu vs p for u=0) ---
ax = axes[2]
i_mid = G // 2
ax.plot(p_grid[i_mid, :], mu[i_mid, :], "g-", lw=2,
        label=f"monotone (Cesaro)  u={u_grid[i_mid]:+.2f}")
ax.plot(p_grid_nm[i_mid, :], mu_nm[i_mid, :], "r--", lw=2,
        label=f"non-monotone NK FP  u={u_grid[i_mid]:+.2f}")
# Pick a couple of off-center rows to illustrate
for off in (-3, +3):
    i = i_mid + off
    if 0 <= i < G:
        ax.plot(p_grid[i, :], mu[i, :], "g-", lw=1.2, alpha=0.5,
                label=f"monotone u={u_grid[i]:+.2f}")
        ax.plot(p_grid_nm[i, :], mu_nm[i, :], "r--", lw=1.2, alpha=0.5,
                label=f"non-monotone u={u_grid[i]:+.2f}")
ax.set_xlabel("price p")
ax.set_ylabel("μ")
ax.set_title("monotone vs non-monotone FP (3 rows)\n"
             "1-R²: 0.0967 (monotone) vs 0.1084 (non-monotone)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
out = "results/full_ree/posterior_v3_G14_monotone_PAVA.pdf"
plt.savefig(out, bbox_inches="tight")
out_png = "results/full_ree/posterior_v3_G14_monotone_PAVA.png"
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved {out}")
print(f"Saved {out_png}")
