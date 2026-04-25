"""Figure 2 — CARA vs CRRA Contour (§4.1, central figure).

Holds agent 1's signal at u₁ = 1, computes the no-learning price surface
P(1, u₂, u₃) on a fine grid, and traces the level set {P = p_obs} where
p_obs = P(1, −1, 1). The level-set points are coloured by

    T* = τ (u₁ + u₂ + u₃)

— the CARA sufficient statistic. Under CARA, the level set is a straight
line and every point has the same T* (single colour). Under CRRA, the
level set is curved and T* varies along it (gradient of colour) — that
gradient is the visual signature of partial revelation.

The level set is extracted from a G_FINE × G_FINE evaluation of the price
surface via matplotlib's contour engine, then drawn as a continuous
colour-mapped LineCollection (no scatter dots).

Output:
  plots/fig2_contour.png  — two-panel curve plot (CARA | CRRA γ=0.5)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import rezn_het as rh


TAU      = 2.0
G_FINE   = 400
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


def _extract_contour_paths(P_slice, u_grid, p_obs):
    """Use matplotlib's contour engine to get ordered (u₂, u₃) vertices on
    the level set. Returns a list of paths, each an (N, 2) array."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(u_grid, u_grid, P_slice.T, levels=[p_obs])
    paths = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) >= 2:
            paths.append(v)
    plt.close(fig_tmp)
    return paths


def _plot_curve(ax, paths, T_min, T_max, tau, u1):
    """Draw each contour path as a greyscale-mapped line where shade
    encodes T*(u_2,u_3) = τ (u_1 + u_2 + u_3) — light = low T*, dark = high.
    Adds tick marks at every Δarc-length so the geometry is readable
    in pure black-and-white print."""
    cmap = plt.get_cmap("Greys")
    # Restrict to [0.25, 1.0] so even constant-T* curves print clearly.
    norm = plt.Normalize(T_min, T_max)
    last_lc = None
    for v in paths:
        u2, u3 = v[:, 0], v[:, 1]
        T = tau * (u1 + u2 + u3)
        pts = np.array([u2, u3]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_T = 0.5 * (T[:-1] + T[1:])
        # Map shade to [0.25, 1.0] of the Greys colormap
        shade = 0.25 + 0.75 * (seg_T - T_min) / max(T_max - T_min, 1e-12)
        rgba = cmap(shade)
        lc = LineCollection(segs, colors=rgba, linewidth=2.6)
        ax.add_collection(lc)
        last_lc = lc
        # Tick marks every 0.5 arc length, perpendicular to tangent
        seg_len = np.linalg.norm(np.diff(v, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        tick_targets = np.arange(0.5, cum[-1], 0.5)
        for tk in tick_targets:
            i = int(np.searchsorted(cum, tk))
            if i <= 0 or i >= len(v):
                continue
            tang = v[i] - v[i - 1]
            n = np.array([-tang[1], tang[0]])
            n /= max(np.linalg.norm(n), 1e-12)
            mid = v[i]
            ax.plot([mid[0] - 0.04 * n[0], mid[0] + 0.04 * n[0]],
                     [mid[1] - 0.04 * n[1], mid[1] + 0.04 * n[1]],
                     color="black", lw=0.8)
    ax.scatter([U2_REAL], [U3_REAL], marker="o", s=160,
                facecolors="none", edgecolors="black", linewidths=1.8,
                zorder=5,
                label=f"realised $(u_2,u_3)=({U2_REAL:.0f},{U3_REAL:.0f})$")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_xlim(-UMAX, UMAX); ax.set_ylim(-UMAX, UMAX)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u_2$"); ax.set_ylabel(r"$u_3$")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    return last_lc


def _plot_T_along_arc(ax, paths, tau, u1, label):
    """Plot T*(arc length) for each contour path."""
    for v in paths:
        u2, u3 = v[:, 0], v[:, 1]
        T = tau * (u1 + u2 + u3)
        seg_len = np.linalg.norm(np.diff(v, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg_len)])
        ax.plot(s, T, color="black", lw=1.6, label=label)
    ax.set_xlabel("arc length along level set")
    ax.set_ylabel(r"$T^* = \tau(u_1+u_2+u_3)$")
    ax.grid(True, alpha=0.3)


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
        p_obs = float(rh._clear_price(
            np.array([1.0 / (1.0 + np.exp(-TAU * U1_FIXED)),
                       1.0 / (1.0 + np.exp(-TAU * U2_REAL)),
                       1.0 / (1.0 + np.exp(-TAU * U3_REAL))]),
            gammas, Ws))
        paths = _extract_contour_paths(P_slice, u_grid, p_obs)
        all_T = np.concatenate(
            [TAU * (U1_FIXED + p[:, 0] + p[:, 1]) for p in paths])
        panels.append((title, paths, p_obs, all_T))
        print(f"{title}: p_obs={p_obs:.6f}  paths={len(paths)}  "
              f"T* range=[{all_T.min():.3f}, {all_T.max():.3f}]")

    T_all = np.concatenate([p[3] for p in panels])
    T_min, T_max = float(T_all.min()), float(T_all.max())

    # 2x2 layout: top row = level sets in (u₂, u₃); bottom row = T*(arc)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
                              gridspec_kw={"height_ratios": [3, 1.4]})
    for col, (title, paths, p_obs, _T) in enumerate(panels):
        _plot_curve(axes[0, col], paths, T_min, T_max, TAU, U1_FIXED)
        axes[0, col].set_title(title + f"\n$p_{{\\rm obs}} = {p_obs:.4f}$")
        _plot_T_along_arc(axes[1, col], paths, TAU, U1_FIXED, title)
        axes[1, col].set_ylim(min(T_min, 0) - 0.1, T_max + 0.1)

    fig.suptitle(r"Level set $\{P(1, u_2, u_3) = p_{\rm obs}\}$  "
                  r"and $T^*$ profile along the arc",
                  y=1.005)
    fig.tight_layout()
    png_path = os.path.join(OUT, "fig2_contour.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
