"""Figure 1 — Smooth Transition (§3 No-Learning Benchmark).

Filled contour plot of 1−R² over (γ, τ). For each (γ, τ), compute the
no-learning equilibrium price tensor at G=30, then 1−R² of logit(P)
against the CARA sufficient statistic T* = τ·Σu_k. The contour map
shows that 1−R² is essentially zero along the entire CARA edge (γ
large) and rises smoothly into the CRRA interior — no phase boundary,
just a continuous transition.

Axes: γ on a log scale (γ ∈ [0.1, 100]), τ linear (τ ∈ [0.3, 3.5]).
Colormap is greyscale on a log-spaced level set so the entire
4-decade range of 1−R² is visible.

Outputs (in plots/):
  fig1_smooth_transition.csv  — table of computed values
  fig1_smooth_transition.png  — filled contour with labelled lines
"""
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

import rezn_het as rh


GAMMAS = np.logspace(-1, 2, 25)         # 0.1 .. 100, 25 points
TAUS   = np.linspace(0.3, 3.5, 17)      # 0.3 .. 3.5, 17 points
G      = 30
UMAX   = 2.0
OUT    = os.path.join(os.path.dirname(__file__), "plots")


def main():
    os.makedirs(OUT, exist_ok=True)
    u  = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])

    table = np.zeros((len(GAMMAS), len(TAUS)))
    for r, g in enumerate(GAMMAS):
        for c, t in enumerate(TAUS):
            taus   = np.array([t, t, t])
            gammas = np.array([g, g, g])
            P0 = rh._nolearning_price(u, taus, gammas, Ws)
            table[r, c] = max(rh.one_minus_R2(P0, u, taus), 1e-9)
        print(f"  γ={g:7.3f}  done  (1−R² range "
               f"{table[r,:].min():.2e} – {table[r,:].max():.2e})",
               flush=True)

    # CSV
    csv_path = os.path.join(OUT, "fig1_smooth_transition.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma\\tau"] + [f"{t:.4f}" for t in TAUS])
        for r, g in enumerate(GAMMAS):
            w.writerow([f"{g:.4f}"]
                        + [f"{table[r, c]:.6e}" for c in range(len(TAUS))])
    print(f"wrote {csv_path}")

    # Filled contour with log-spaced levels
    T_grid, G_grid = np.meshgrid(TAUS, GAMMAS)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    levels = np.logspace(-6, np.log10(0.3), 25)
    cs = ax.contourf(T_grid, G_grid, table, levels=levels,
                      cmap="Greys", norm=LogNorm(vmin=1e-6, vmax=0.3),
                      extend="both")
    # Labelled contour lines at decade ticks
    line_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    cl = ax.contour(T_grid, G_grid, table, levels=line_levels,
                     colors="black", linewidths=0.9)
    ax.clabel(cl, inline=True, fontsize=8, fmt=lambda v: f"{v:g}")
    ax.set_yscale("log")
    ax.set_xlabel(r"signal precision $\tau$")
    ax.set_ylabel(r"risk aversion $\gamma$  (log scale)")
    ax.set_title(r"$1 - R^2$ of logit$(p)$ vs $T^* = \sum_k \tau_k u_k$"
                  "  (no-learning, exact)")

    cbar = fig.colorbar(cs, ax=ax,
                         ticks=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                         format=LogFormatter(base=10),
                         label=r"$1 - R^2$  (log scale)")
    fig.tight_layout()
    png = os.path.join(OUT, "fig1_smooth_transition.png")
    fig.savefig(png, dpi=200)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
