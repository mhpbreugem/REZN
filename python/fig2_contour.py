"""Figure 2 — CARA vs CRRA Contour (§4.1, central figure).

Holds agent 1's signal at u₁ = 1, computes the no-learning price surface
P(1, u₂, u₃) on a fine grid, and traces the level set {P = p_obs} where
p_obs = P(1, −1, 1). The level-set points are coloured by

    T* = τ (u₁ + u₂ + u₃)

— the CARA sufficient statistic. Under CARA, the level set is a straight
line and every point has the same T* (single colour). Under CRRA, the
level set is curved and T* varies along it (gradient of colour) — that
gradient is the visual signature of partial revelation.

Parameters: τ = 2, G = 200 (1D grid in each axis at u ∈ [−2, 2]).

Output:
  plots/fig2_contour.png  — two-panel scatter (CARA | CRRA γ=0.5)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import rezn_het as rh


TAU      = 2.0
G_FINE   = 200
UMAX     = 2.0
U1_FIXED = 1.0
U2_REAL, U3_REAL = -1.0, 1.0
OUT      = os.path.join(os.path.dirname(__file__), "plots")


def _slice_no_learning(u1, u_grid, taus, gammas, Ws):
    """Return P(u1, u2, u3) on the (u_grid × u_grid) tensor, no learning."""
    Gf = len(u_grid)
    P = np.empty((Gf, Gf))
    m0 = 1.0 / (1.0 + np.exp(-taus[0] * u1))
    for j in range(Gf):
        m1 = 1.0 / (1.0 + np.exp(-taus[1] * u_grid[j]))
        for l in range(Gf):
            m2 = 1.0 / (1.0 + np.exp(-taus[2] * u_grid[l]))
            mus = np.array([m0, m1, m2])
            P[j, l] = rh._clear_price(mus, gammas, Ws)
    return P


def _trace_level_set(P_slice, u_grid, p_obs):
    """For each row j, root-find l-coords (off-grid) where the row crosses
    p_obs. Returns (u2_arr, u3_arr) of crossing coordinates (linear interp
    within segments, like the production contour method)."""
    G = len(u_grid)
    u2_list, u3_list = [], []
    # Pass A: rows
    for j in range(G):
        for l in range(G - 1):
            y0 = P_slice[j, l] - p_obs
            y1 = P_slice[j, l + 1] - p_obs
            if y0 * y1 < 0:
                t = y0 / (y0 - y1)
                u3 = u_grid[l] + t * (u_grid[l + 1] - u_grid[l])
                u2_list.append(u_grid[j])
                u3_list.append(u3)
    # Pass B: cols
    for l in range(G):
        for j in range(G - 1):
            y0 = P_slice[j, l] - p_obs
            y1 = P_slice[j + 1, l] - p_obs
            if y0 * y1 < 0:
                t = y0 / (y0 - y1)
                u2 = u_grid[j] + t * (u_grid[j + 1] - u_grid[j])
                u2_list.append(u2)
                u3_list.append(u_grid[l])
    return np.array(u2_list), np.array(u3_list)


def _plot_panel(ax, u2, u3, T_vals, T_ref, title, T_min, T_max):
    sc = ax.scatter(u2, u3, c=T_vals, cmap="viridis",
                     vmin=T_min, vmax=T_max, s=8, edgecolors="none")
    # Mark the actual realisation
    ax.scatter([U2_REAL], [U3_REAL], marker="o", s=140,
               facecolors="none", edgecolors="black", linewidths=1.6,
               label=f"realised $(u_2,u_3)=({U2_REAL},{U3_REAL})$")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlim(-UMAX, UMAX); ax.set_ylim(-UMAX, UMAX)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u_2$"); ax.set_ylabel(r"$u_3$")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    return sc


def main():
    os.makedirs(OUT, exist_ok=True)

    u_grid = np.linspace(-UMAX, UMAX, G_FINE)
    Ws  = np.array([1.0, 1.0, 1.0])
    taus = np.array([TAU, TAU, TAU])

    cases = [(100.0, "CARA  ($\\gamma=100$)"),
             (0.5,   "CRRA  ($\\gamma=0.5$)")]

    panels = []
    for gamma, title in cases:
        gammas = np.array([gamma, gamma, gamma])
        P_slice = _slice_no_learning(U1_FIXED, u_grid, taus, gammas, Ws)
        # p_obs at the actual realisation
        p_obs = float(rh._clear_price(
            np.array([1.0 / (1.0 + np.exp(-TAU * U1_FIXED)),
                       1.0 / (1.0 + np.exp(-TAU * U2_REAL)),
                       1.0 / (1.0 + np.exp(-TAU * U3_REAL))]),
            gammas, Ws))
        u2, u3 = _trace_level_set(P_slice, u_grid, p_obs)
        T = TAU * (U1_FIXED + u2 + u3)
        panels.append((title, u2, u3, T, p_obs))
        print(f"{title}: p_obs={p_obs:.6f}  contour pts={len(u2)}  "
              f"T* range=[{T.min():.3f}, {T.max():.3f}]")

    # Shared color range
    T_all = np.concatenate([p[3] for p in panels])
    T_min, T_max = float(T_all.min()), float(T_all.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    sc = None
    for ax, (title, u2, u3, T, p_obs) in zip(axes, panels):
        sc = _plot_panel(ax, u2, u3, T, p_obs, title, T_min, T_max)
        ax.set_title(title + f"\n$p_{{\\rm obs}} = {p_obs:.4f}$")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85,
                         label=r"$T^* = \tau(u_1+u_2+u_3)$")
    fig.suptitle(r"Level set $\{P(1, u_2, u_3) = p_{\rm obs}\}$, "
                  r"coloured by CARA sufficient statistic",
                  y=1.02)
    png_path = os.path.join(OUT, "fig2_contour.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
