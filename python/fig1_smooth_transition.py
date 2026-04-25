"""Figure 1 — Smooth Transition Table (no-learning, §3).

For each (γ, τ) on the paper grid, compute the no-learning equilibrium
price tensor and report 1−R² of logit(P) against the CARA sufficient
statistic T* = τ·Σu_k. CARA (γ→∞) is at 1−R² = 0; CRRA gives strictly
positive PR with magnitude rising as γ falls.

No iteration, no contour integration — just market clearing with private
posteriors. Exact at any G; we use G=30.

Outputs (in plots/):
  fig1_smooth_transition.csv  — 9×4 table
  fig1_smooth_transition.png  — heatmap
"""
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import rezn_het as rh


GAMMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]
TAUS   = [0.5, 1.0, 2.0, 3.0]
G      = 30
UMAX   = 2.0   # paper baseline; UMAX=4 over-saturates prices
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
            table[r, c] = rh.one_minus_R2(P0, u, taus)
            print(f"  γ={g:5.1f} τ={t:4.1f}  1−R²={table[r,c]:.4e}",
                  flush=True)

    # CSV
    csv_path = os.path.join(OUT, "fig1_smooth_transition.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma\\tau"] + [f"{t}" for t in TAUS])
        for r, g in enumerate(GAMMAS):
            w.writerow([f"{g}"] + [f"{table[r,c]:.6e}" for c in range(len(TAUS))])
    print(f"wrote {csv_path}")

    # B&W heatmap: greyscale, white→dark grey, vmax = 0.15
    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_grey", ["#ffffff", "#d9d9d9", "#969696", "#525252", "#000000"])
    im = ax.imshow(table, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.15,
                    origin="upper")
    ax.set_xticks(range(len(TAUS)));   ax.set_xticklabels([f"{t}" for t in TAUS])
    ax.set_yticks(range(len(GAMMAS))); ax.set_yticklabels([f"{g}" for g in GAMMAS])
    ax.set_xlabel(r"signal precision $\tau$")
    ax.set_ylabel(r"risk aversion $\gamma$")
    ax.set_title(r"$1 - R^2$ of logit$(p)$ vs $T^* = \sum_k \tau_k u_k$"
                  "\n(no-learning, exact)")

    for r in range(len(GAMMAS)):
        for c in range(len(TAUS)):
            v = table[r, c]
            ax.text(c, r, f"{v:.3f}" if v >= 1e-3 else f"{v:.0e}",
                     ha="center", va="center",
                     color="black" if v < 0.08 else "white",
                     fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label=r"$1-R^2$")
    fig.tight_layout()
    png_path = os.path.join(OUT, "fig1_smooth_transition.png")
    fig.savefig(png_path, dpi=180)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
