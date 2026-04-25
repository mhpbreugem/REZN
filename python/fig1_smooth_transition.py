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
from fig_export import save_png_pdf_tex


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

    # Pure black/white contour lines, no fill — Econometrica style.
    T_grid, G_grid = np.meshgrid(TAUS, GAMMAS)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Decade contour lines
    decade_levels = [1e-4, 1e-3, 1e-2, 1e-1]
    cl_dec = ax.contour(T_grid, G_grid, table, levels=decade_levels,
                         colors="black", linewidths=1.1)
    ax.clabel(cl_dec, inline=True, fontsize=9,
               fmt=lambda v: f"{v:g}")

    # Intermediate (half-decade) lines, lighter
    half_levels = [3e-4, 3e-3, 3e-2, 0.05, 0.15, 0.2]
    cl_h = ax.contour(T_grid, G_grid, table, levels=half_levels,
                      colors="black", linewidths=0.6, linestyles=":")
    ax.clabel(cl_h, inline=True, fontsize=8,
              fmt=lambda v: f"{v:g}")

    ax.set_yscale("log")
    ax.set_xlim(TAUS.min(), TAUS.max())
    ax.set_ylim(GAMMAS.min(), GAMMAS.max())
    ax.set_xlabel(r"signal precision $\tau$")
    ax.set_ylabel(r"risk aversion $\gamma$  (log scale)")
    ax.set_title(r"Iso-curves of $1 - R^2$ for logit$(p)$ vs $T^*$"
                  "  (no-learning, exact)")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    fig.tight_layout()
    save_png_pdf_tex(fig, os.path.join(OUT, "fig1_smooth_transition"))


if __name__ == "__main__":
    main()
