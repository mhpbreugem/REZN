"""CARA knife-edge figure.

Three-agent market:
    agent 1  has CARA utility  with absolute risk aversion  a   (x-axis)
    agent 2  has CRRA utility  with relative risk aversion  γ   (y-axis)
    agent 3  has CRRA utility  with relative risk aversion  γ

For each (a, γ) on a log×log grid, we solve the no-learning equilibrium
price tensor and report 1−R² of logit(P) against the CARA sufficient
statistic T* = τ·Σu_k. Two independent routes lead to full revelation
(1−R² → 0) on the boundary of the plot:

  • γ → ∞ (top edge):  the CRRA agents become Bernoulli-like and their
    log-odds demand approaches the CARA form. The aggregator collapses
    to a CARA-style log-linear average → FR.
  • a → 0  (left edge): the CARA agent becomes infinitely risk-tolerant
    and dominates supply/demand. Equilibrium is set by his log-linear
    demand → FR.

Anywhere strictly in the interior gives 1−R² > 0 (partial revelation).
The CARA knife-edge is the entire γ = ∞ boundary — no perturbation in
γ leaves it intact.

Output:
    figures/fig_knife_edge.{png,pdf,tex}
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import rezn_het as rh
from fig_export import save_png_pdf_tex


A_VALS = np.logspace(-1, 1.5, 28)        # 0.1 .. ~32   ARA
G_VALS = np.logspace(-1, 1.5, 28)        # 0.1 .. ~32   RRA
TAU    = 2.0
G      = 25
UMAX   = 2.0
OUT    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "figures")


@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))


@njit(cache=True, fastmath=True)
def _residual_mixed(mus, p, a, gamma):
    """1 CARA agent (coef a) + 2 CRRA agents (coef γ). W=1."""
    eps = 1e-12
    p_c = max(eps, min(1 - eps, p))
    s = 0.0
    # CARA agent
    mu0 = max(eps, min(1 - eps, mus[0]))
    s += (_logit(mu0) - _logit(p_c)) / a
    # 2 CRRA agents
    for k in (1, 2):
        muk = max(eps, min(1 - eps, mus[k]))
        R = np.exp((_logit(muk) - _logit(p_c)) / gamma)
        s += (R - 1.0) / ((1.0 - p_c) + R * p_c)
    return s


@njit(cache=True)
def _clear_price_mixed(mus, a, gamma):
    lo = 1e-9; hi = 1 - 1e-9
    f_lo = _residual_mixed(mus, lo, a, gamma)
    f_hi = _residual_mixed(mus, hi, a, gamma)
    if f_lo == 0.0: return lo
    if f_hi == 0.0: return hi
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f_m = _residual_mixed(mus, m, a, gamma)
        if (hi - lo) < 1e-12 or f_m == 0.0:
            return m
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
    return 0.5 * (lo + hi)


@njit(cache=True)
def _nolearning_mixed(u, a, gamma, tau):
    Gn = u.shape[0]
    P = np.empty((Gn, Gn, Gn))
    for i in range(Gn):
        m0 = 1.0 / (1.0 + np.exp(-tau * u[i]))
        for j in range(Gn):
            m1 = 1.0 / (1.0 + np.exp(-tau * u[j]))
            for l in range(Gn):
                m2 = 1.0 / (1.0 + np.exp(-tau * u[l]))
                mus = np.array([m0, m1, m2])
                P[i, j, l] = _clear_price_mixed(mus, a, gamma)
    return P


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)

    table = np.zeros((len(G_VALS), len(A_VALS)))
    for r, gam in enumerate(G_VALS):
        for c, a in enumerate(A_VALS):
            P = _nolearning_mixed(u, float(a), float(gam), TAU)
            table[r, c] = max(rh.one_minus_R2(P, u,
                                               np.array([TAU, TAU, TAU])),
                                1e-12)
        print(f"  γ={gam:7.3f}  done  "
               f"(1−R² range {table[r, :].min():.2e} – "
               f"{table[r, :].max():.2e})", flush=True)

    A_grid, G_grid = np.meshgrid(A_VALS, G_VALS)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    decade_levels = [1e-4, 1e-3, 1e-2, 1e-1]
    cl_d = ax.contour(A_grid, G_grid, table, levels=decade_levels,
                       colors="black", linewidths=1.1)
    ax.clabel(cl_d, inline=True, fontsize=9, fmt=lambda v: f"{v:g}")
    half_levels = [3e-4, 3e-3, 3e-2, 0.05, 0.15]
    cl_h = ax.contour(A_grid, G_grid, table, levels=half_levels,
                       colors="black", linewidths=0.6, linestyles=":")
    ax.clabel(cl_h, inline=True, fontsize=8, fmt=lambda v: f"{v:g}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(A_VALS.min(), A_VALS.max())
    ax.set_ylim(G_VALS.min(), G_VALS.max())
    ax.set_xlabel(r"CARA coefficient $a$  (ARA, log scale)")
    ax.set_ylabel(r"CRRA coefficient $\gamma$  (RRA, log scale)")
    ax.set_title(r"$1 - R^2$ in the (ARA, RRA) plane —"
                  r" CARA is a knife edge"
                  f"  ($\\tau={TAU}$, 1 CARA + 2 CRRA agents)")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    # Annotation arrows pointing at the knife edges
    ax.annotate("", xy=(A_VALS[-1] * 0.85, G_VALS[-1] * 0.95),
                 xytext=(A_VALS[-1] * 0.85, G_VALS[-1] * 0.4),
                 arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(A_VALS[-1] * 0.6, G_VALS[-1] * 0.95,
             "CARA limit\n(knife edge)",
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="black"))
    fig.tight_layout()
    save_png_pdf_tex(fig, os.path.join(OUT, "fig_knife_edge"))


if __name__ == "__main__":
    main()
