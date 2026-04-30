"""Six distinct plots of the monotone μ(u, p) at G=14, γ=0.5, τ=2.

1. Heatmap of μ on the (u, p) lattice
2. 3D surface μ(u, p)
3. Per-row curves μ(u_i, p) vs p (rainbow by u)
4. Per-column curves μ(u, p_j) vs u (rainbow by p)
5. Contour / level-set plot of μ(u, p) on a regular grid
6. Demand heatmap x(u, p) = CRRA(μ(u,p), p)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from posterior_method_v3 import Lam, crra_demand_vec, EPS

RED = (0.7, 0.11, 0.11)
BLUE = (0.0, 0.20, 0.42)
GREEN = (0.11, 0.35, 0.02)

TAU = 2.0
GAMMA = 0.5

d = np.load("results/full_ree/posterior_v3_G14_PAVA_cesaro_mu.npz")
mu = d["mu"]
u_grid = d["u_grid"]
p_grid = d["p_grid"]
p_lo = d["p_lo"]; p_hi = d["p_hi"]
G = mu.shape[0]


def lattice_corners(values):
    """Build cell-corner array from cell centers."""
    return np.concatenate([
        [1.5*values[0] - 0.5*values[1]],
        0.5*(values[:-1] + values[1:]),
        [1.5*values[-1] - 0.5*values[-2]],
    ])


# ---- 1. Heatmap on (u, p) lattice (per-row p-grid) ----
fig, ax = plt.subplots(figsize=(7, 6))
U_corners = lattice_corners(u_grid)
for i in range(G):
    p_row = p_grid[i, :]
    p_corners = lattice_corners(p_row)
    u_corners = [U_corners[i], U_corners[i+1]]
    P, U = np.meshgrid(p_corners, u_corners)
    Z = mu[i:i+1, :]
    pc = ax.pcolormesh(U, P, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="flat")
ax.set_xlabel("private signal $u$", fontsize=11)
ax.set_ylabel("price $p$", fontsize=11)
ax.set_title(f"(1) μ(u, p) heatmap on the (u, p) lattice  "
             f"[G={G}, γ={GAMMA}, τ={TAU}]", fontsize=11)
ax.set_ylim(0, 1)
plt.colorbar(pc, ax=ax, label="μ(u, p)")
plt.tight_layout()
plt.savefig("results/full_ree/plot1_heatmap.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot1_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Saved plot1 heatmap")

# ---- 2. 3D surface ----
# To get a surface, we need μ on a regular (u, p) grid. Resample.
P_REG = np.linspace(0.02, 0.98, 60)
U_REG = u_grid.copy()
mu_reg = np.empty((len(U_REG), len(P_REG)))
for i, u in enumerate(U_REG):
    for j, p in enumerate(P_REG):
        if p < p_grid[i, 0]:
            mu_reg[i, j] = mu[i, 0]
        elif p > p_grid[i, -1]:
            mu_reg[i, j] = mu[i, -1]
        else:
            mu_reg[i, j] = float(np.interp(p, p_grid[i, :], mu[i, :]))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
P_mesh, U_mesh = np.meshgrid(P_REG, U_REG)
surf = ax.plot_surface(U_mesh, P_mesh, mu_reg, cmap="RdBu_r",
                        vmin=0, vmax=1, edgecolor="none", alpha=0.95)
ax.set_xlabel("private signal $u$")
ax.set_ylabel("price $p$")
ax.set_zlabel("μ(u, p)")
ax.set_title(f"(2) 3D surface μ(u, p)  [G={G}, γ={GAMMA}, τ={TAU}]")
ax.view_init(elev=24, azim=-130)
plt.colorbar(surf, ax=ax, shrink=0.6, label="μ")
plt.tight_layout()
plt.savefig("results/full_ree/plot2_surface3d.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot2_surface3d.pdf", bbox_inches="tight")
plt.close()
print("Saved plot2 3D surface")

# ---- 3. Per-row curves μ(u_i, p) vs p ----
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.cm.viridis
for i in range(G):
    color = cmap(i / max(G-1, 1))
    ax.plot(p_grid[i, :], mu[i, :], color=color, lw=1.8,
            label=f"u={u_grid[i]:+.2f}" if i % 2 == 0 else None)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1, label="μ=p (FR)")
ax.set_xlabel("price $p$", fontsize=11)
ax.set_ylabel("μ(u, p)", fontsize=11)
ax.set_title(f"(3) μ-columns: each line is fixed signal u  [G={G}]",
             fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/full_ree/plot3_rows.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot3_rows.pdf", bbox_inches="tight")
plt.close()
print("Saved plot3 rows")

# ---- 4. Per-column curves μ(u, p_j) vs u ----
# To plot at fixed p, we need to resample mu in u for several p values.
fig, ax = plt.subplots(figsize=(7, 6))
P_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cmap = plt.cm.plasma
for k, p0 in enumerate(P_LEVELS):
    # For each u_i, look up mu at p0 (with interpolation)
    mu_col = np.empty(G)
    for i in range(G):
        if p0 < p_grid[i, 0]:
            mu_col[i] = mu[i, 0]
        elif p0 > p_grid[i, -1]:
            mu_col[i] = mu[i, -1]
        else:
            mu_col[i] = float(np.interp(p0, p_grid[i, :], mu[i, :]))
    color = cmap(k / max(len(P_LEVELS)-1, 1))
    ax.plot(u_grid, mu_col, color=color, lw=1.8, marker="o", ms=3,
            label=f"p={p0:.1f}")
# No-learning reference: μ = Λ(τ u)
u_fine = np.linspace(u_grid[0], u_grid[-1], 200)
ax.plot(u_fine, Lam(TAU * u_fine), "k--", lw=1.5, label="Λ(τu) (no learning)")
ax.set_xlabel("private signal $u$", fontsize=11)
ax.set_ylabel("μ(u, p)", fontsize=11)
ax.set_title(f"(4) μ-rows: each line is fixed price p  [G={G}]",
             fontsize=11)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/full_ree/plot4_cols.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot4_cols.pdf", bbox_inches="tight")
plt.close()
print("Saved plot4 cols")

# ---- 5. Contour / level-set plot ----
fig, ax = plt.subplots(figsize=(7, 6))
levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
cs = ax.contourf(U_REG, P_REG, mu_reg.T, levels=20, cmap="RdBu_r",
                  vmin=0, vmax=1)
cl = ax.contour(U_REG, P_REG, mu_reg.T, levels=levels, colors="black",
                 linewidths=0.7, alpha=0.7)
ax.clabel(cl, inline=True, fontsize=8, fmt="%.2f")
ax.set_xlabel("private signal $u$", fontsize=11)
ax.set_ylabel("price $p$", fontsize=11)
ax.set_title(f"(5) Level sets of μ(u, p): contours of constant posterior  "
             f"[γ={GAMMA}]", fontsize=11)
plt.colorbar(cs, ax=ax, label="μ")
plt.tight_layout()
plt.savefig("results/full_ree/plot5_contour.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot5_contour.pdf", bbox_inches="tight")
plt.close()
print("Saved plot5 contour")

# ---- 6. Demand x(u, p) heatmap ----
x_grid = crra_demand_vec(mu_reg.ravel().reshape(mu_reg.shape),
                           np.broadcast_to(P_REG, mu_reg.shape), GAMMA)
fig, ax = plt.subplots(figsize=(7, 6))
vmax = float(np.max(np.abs(x_grid)))
pc = ax.pcolormesh(U_REG, P_REG, x_grid.T, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, shading="auto")
# Zero-demand contour
ax.contour(U_REG, P_REG, x_grid.T, levels=[0], colors="black", linewidths=1.5)
ax.set_xlabel("private signal $u$", fontsize=11)
ax.set_ylabel("price $p$", fontsize=11)
ax.set_title(f"(6) Demand x(u, p) at converged μ  "
             f"(black line: x=0)  [γ={GAMMA}]", fontsize=11)
plt.colorbar(pc, ax=ax, label="demand x")
plt.tight_layout()
plt.savefig("results/full_ree/plot6_demand.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/plot6_demand.pdf", bbox_inches="tight")
plt.close()
print("Saved plot6 demand")

# Summary index
print("\n=== Six perspectives saved ===")
for i, name in enumerate([
    "plot1_heatmap", "plot2_surface3d", "plot3_rows",
    "plot4_cols", "plot5_contour", "plot6_demand",
]):
    print(f"  {i+1}. results/full_ree/{name}.png")
