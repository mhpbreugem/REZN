"""Plot G=14 equilibrium price contours + no-learning baseline.

For agent 1 at signal u_i, the contour C(u_i, p_0) is the set of
(u_2, u_3) such that d(u_1; p_0) + d(u_2; p_0) + d(u_3; p_0) = 0
under whatever belief structure each agent uses.

We plot two families of contours, both at the same u_1:
  - EQUILIBRIUM contours: μ_k = μ*(u_k, p_0) from G=14 NK-converged solution
  - NO-LEARNING contours: μ_k = Λ(τ u_k) (private prior only)

Each curve corresponds to a different observed price level p_0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from posterior_method_v3 import (
    Lam, init_p_grid, crra_demand_vec, EPS,
)

TAU = 2.0
GAMMA = 0.5
UMAX = 4.0


def equilibrium_d_at_price(mu, p_grid, p_lo, p_hi, u_grid, p0, gamma):
    """Demand array d(u; p_0) using equilibrium μ(u, p_0)."""
    Gu = len(u_grid)
    mu_col = np.empty(Gu)
    for i in range(Gu):
        if p0 < p_grid[i, 0]:
            mu_col[i] = mu[i, 0]
        elif p0 > p_grid[i, -1]:
            mu_col[i] = mu[i, -1]
        else:
            mu_col[i] = float(np.interp(p0, p_grid[i, :], mu[i, :]))
    mu_col = np.clip(mu_col, EPS, 1 - EPS)
    return crra_demand_vec(mu_col, np.full_like(mu_col, p0), gamma), mu_col


def no_learning_d_at_price(u_grid, p0, tau, gamma):
    """Demand array d(u; p_0) using no-learning posterior μ = Λ(τu)."""
    mu_col = np.array([Lam(tau * u) for u in u_grid])
    return crra_demand_vec(mu_col, np.full_like(mu_col, p0), gamma), mu_col


def contour_in_u2u3(d_array, u_grid, u1):
    """Given demand array d(u; p_0) (length Gu), trace the contour
    {(u_2, u_3): d(u_1) + d(u_2) + d(u_3) = 0}.

    Sweep u_2 over a fine u-grid; invert d to find u_3*(u_2).
    """
    # d at u_1 (interpolated)
    d_u1 = float(np.interp(u1, u_grid, d_array))
    u2_fine = np.linspace(u_grid[0], u_grid[-1], 400)
    d2_fine = np.interp(u2_fine, u_grid, d_array)
    targets = -d_u1 - d2_fine
    # Invert d (assume monotone increasing)
    if d_array[-1] > d_array[0]:
        u3_star = np.interp(targets, d_array, u_grid,
                            left=np.nan, right=np.nan)
    else:
        u3_star = np.interp(targets, d_array[::-1], u_grid[::-1],
                            left=np.nan, right=np.nan)
    valid = ~np.isnan(u3_star)
    return u2_fine[valid], u3_star[valid]


# Load G=14 NK-converged checkpoint
ckpt = np.load("results/full_ree/posterior_v3_fine_G14_mu.npz")
mu = ckpt["mu"]; u_grid = ckpt["u_grid"]
p_grid = ckpt["p_grid"]; p_lo = ckpt["p_lo"]; p_hi = ckpt["p_hi"]
print(f"Loaded G=14 checkpoint: mu shape {mu.shape}, "
      f"u_range=[{u_grid[0]:.2f},{u_grid[-1]:.2f}]")

# Pick u_1 = 0 (central agent) and a fan of observed prices
u1 = 0.0
# Use prices spanning the row's lens at u_1
i_mid = np.argmin(np.abs(u_grid - u1))
p_levels = np.linspace(p_grid[i_mid, 1], p_grid[i_mid, -2], 7)

# BC20 colors
red = (0.7, 0.11, 0.11)
blue = (0.0, 0.20, 0.42)
green = (0.11, 0.35, 0.02)

fig, ax = plt.subplots(figsize=(7, 7))
for k, p0 in enumerate(p_levels):
    # Equilibrium contour
    d_eq, mu_col_eq = equilibrium_d_at_price(mu, p_grid, p_lo, p_hi,
                                               u_grid, p0, GAMMA)
    u2_eq, u3_eq = contour_in_u2u3(d_eq, u_grid, u1)
    # No-learning contour at the same p_0
    d_nl, mu_col_nl = no_learning_d_at_price(u_grid, p0, TAU, GAMMA)
    u2_nl, u3_nl = contour_in_u2u3(d_nl, u_grid, u1)
    label_eq = f"REE p={p0:.3f}" if k == 0 else None
    label_nl = f"no-learning p={p0:.3f}" if k == 0 else None
    ax.plot(u2_eq, u3_eq, color=red, lw=1.5, label=label_eq)
    ax.plot(u2_nl, u3_nl, color=blue, lw=1.0, ls="--", label=label_nl)
    # Annotate the price next to the curve
    if len(u2_eq) > 0:
        midi = len(u2_eq) // 2
        ax.text(u2_eq[midi], u3_eq[midi], f"{p0:.2f}",
                color=red, fontsize=7, ha="left", va="bottom")

ax.axhline(0, color="gray", lw=0.3)
ax.axvline(0, color="gray", lw=0.3)
ax.plot(u1, u1, "k+", markersize=10)
ax.set_xlim(u_grid[0], u_grid[-1])
ax.set_ylim(u_grid[0], u_grid[-1])
ax.set_xlabel(r"$u_2$")
ax.set_ylabel(r"$u_3$")
ax.set_title(rf"Market-clearing contours at $u_1={u1:.1f}$ (G=14, "
             rf"$\tau={TAU},\ \gamma={GAMMA}$)")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_aspect("equal")
plt.tight_layout()
out = Path("figures/G14_contours_REE_vs_nolearning.pdf")
plt.savefig(out)
plt.savefig(out.with_suffix(".png"), dpi=150)
print(f"Saved: {out} and {out.with_suffix('.png')}")

# Print numeric details
print(f"\nu_1 = {u1}")
print(f"Plotted {len(p_levels)} price levels: {[f'{p:.3f}' for p in p_levels]}")
print(f"\nμ-column at p_0 = {p_levels[len(p_levels)//2]:.3f}:")
p0_mid = p_levels[len(p_levels)//2]
d_eq, mu_eq_col = equilibrium_d_at_price(mu, p_grid, p_lo, p_hi, u_grid,
                                           p0_mid, GAMMA)
d_nl, mu_nl_col = no_learning_d_at_price(u_grid, p0_mid, TAU, GAMMA)
print(f"  {'u':>8} {'μ_REE':>8} {'μ_NL':>8} {'d_REE':>9} {'d_NL':>9}")
for i, u in enumerate(u_grid):
    print(f"  {u:>8.3f} {mu_eq_col[i]:>8.4f} {mu_nl_col[i]:>8.4f} "
          f"{d_eq[i]:>9.4f} {d_nl[i]:>9.4f}")
