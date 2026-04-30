"""Plot strict-1e-13 μ at G=10 and G=12 (γ=0.5, τ=2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = (0.7, 0.11, 0.11)
BLUE = (0.0, 0.20, 0.42)
GREEN = (0.11, 0.35, 0.02)


def plot_one(ax_h, ax_c, mu, u_grid, p_grid, p_lo, p_hi, label):
    G = mu.shape[0]
    # Heatmap (per-row strips because each row has its own p-grid)
    U_corners = np.linspace(u_grid[0], u_grid[-1], G + 1)
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
        pc = ax_h.pcolormesh(U, P, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="flat")
    ax_h.set_xlabel("u")
    ax_h.set_ylabel("p")
    ax_h.set_title(label)
    ax_h.set_ylim(0, 1)
    plt.colorbar(pc, ax=ax_h, label="μ")

    # μ-columns
    cmap = plt.cm.viridis
    for i in range(G):
        col = cmap(i / max(G-1, 1))
        ax_c.plot(p_grid[i, :], mu[i, :], color=col, lw=1.5,
                   label=f"u={u_grid[i]:+.2f}" if i % max(G//5, 1) == 0 else None)
    ax_c.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1, label="μ=p (FR)")
    ax_c.set_xlabel("p")
    ax_c.set_ylabel("μ(u, p)")
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
    ax_c.legend(fontsize=7, loc="lower right")
    ax_c.grid(alpha=0.3)
    ax_c.set_title(label)


fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for col_idx, G in enumerate([10, 12]):
    d = np.load(f"results/full_ree/posterior_v3_strict_G{G}.npz")
    mu = d["mu"]; u_grid = d["u_grid"]; p_grid = d["p_grid"]
    p_lo = d["p_lo"]; p_hi = d["p_hi"]
    # Compute 1-R² and slope from json
    import json
    with open("results/full_ree/posterior_v3_strict_emin13.json") as f:
        R = json.load(f)
    rec = next(r for r in R["G"] if r["G"] == G)
    label = (f"G={G}, γ=0.5, τ=2  (strict 1e-13)\n"
             f"1-R²={rec['1-R^2']:.4e}, slope={rec['slope']:.4f}, "
             f"max={rec['max']:.2e}, med={rec['med']:.2e}")
    plot_one(axes[0, col_idx], axes[1, col_idx], mu, u_grid, p_grid,
              p_lo, p_hi, label)

plt.suptitle("Strict 1e-13 monotone REE: G=10 and G=12", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/full_ree/posterior_v3_strict_G10_G12.png",
             dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/posterior_v3_strict_G10_G12.pdf",
             bbox_inches="tight")
print("Saved: results/full_ree/posterior_v3_strict_G10_G12.png")
