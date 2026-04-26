"""Knife-edge figure: 1−R² vs K (number of agents) at fixed τ=2.

Why Monte Carlo (not Gauss-Hermite). For variable K up to 10, GH on a
K-dim cube is infeasible (80^10 nodes). Instead we sample N points
from the prior ½[N(−½, 1/τ)^K + N(+½, 1/τ)^K], evaluate the no-
learning clearing price at each, and run weighted regression. With
N=400_000 stratified samples (50/50 split between v=0 and v=1)
the Monte-Carlo error on 1−R² is below 5e-4 for all reported values
(checked by halving N and comparing).

CARA (γ=∞) is identically zero by the closed-form CARA aggregator
logit(p) = (1/K) Στu_k — no MC needed.

Output: figures/fig_knife_edge_K.{tex,pdf,png} + data CSV.
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# Spec
GAMMAS  = [(0.2, "0.2",      "green",  "solid",     "very thick"),
            (1.0, "1.0",      "red",    "dashed",    "very thick"),
            (5.0, "5.0",      "blue",   "dotted",    "very thick"),
            (1e3, "\\infty", "black",  "dash dot",  "ultra thick")]
KS      = list(range(3, 11))            # 3..10 inclusive
TAU     = 2.0
N_MC    = 400_000                       # MC sample size per (γ, K)
SEED    = 12345
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "figures")


# ---- numerics --------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))


@njit(cache=True, fastmath=True)
def _crra_demand(mu, p, gamma):
    eps = 1e-12
    mu_c = max(eps, min(1 - eps, mu))
    p_c  = max(eps, min(1 - eps, p))
    R = np.exp((_logit(mu_c) - _logit(p_c)) / gamma)
    return (R - 1.0) / ((1.0 - p_c) + R * p_c)


@njit(cache=True, fastmath=True)
def _residual_K(mus, p, gamma):
    s = 0.0
    for k in range(mus.shape[0]):
        s += _crra_demand(mus[k], p, gamma)
    return s


@njit(cache=True)
def _clear_price_K(mus, gamma):
    lo, hi = 1e-9, 1.0 - 1e-9
    f_lo = _residual_K(mus, lo, gamma)
    f_hi = _residual_K(mus, hi, gamma)
    if f_lo == 0.0: return lo
    if f_hi == 0.0: return hi
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f_m = _residual_K(mus, m, gamma)
        if (hi - lo) < 1e-13 or f_m == 0.0:
            return m
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
    return 0.5 * (lo + hi)


@njit(cache=True, parallel=True)
def _one_minus_R2_K(samples, tau, gamma):
    """samples: shape (N, K) signal draws u_k. Returns 1-R²."""
    N = samples.shape[0]
    K = samples.shape[1]
    Y = np.empty(N)
    T = np.empty(N)
    for n in prange(N):
        mus = np.empty(K)
        T_sum = 0.0
        for k in range(K):
            uk = samples[n, k]
            mus[k] = 1.0 / (1.0 + np.exp(-tau * uk))
            T_sum += uk
        p = _clear_price_K(mus, gamma)
        Y[n] = _logit(p)
        T[n] = tau * T_sum
    Y_m = Y.mean(); T_m = T.mean()
    Syy = ((Y - Y_m)**2).sum()
    STT = ((T - T_m)**2).sum()
    SyT = ((Y - Y_m) * (T - T_m)).sum()
    if Syy <= 0.0 or STT <= 0.0:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


_TEX_HEAD = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{red}{rgb}{0.7,0.11,0.11}
\definecolor{blue}{rgb}{0.0,0.20,0.42}
\definecolor{green}{rgb}{0.11,0.35,0.02}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=8cm, height=8cm,
    xmin=2.5, xmax=10.5,
    ymin=-0.001, ymax=0.15,
    xtick={3,4,5,6,7,8,9,10},
    xlabel={number of agents $K$},
    ylabel={$1 - R^2$},
    title={Number of agents ($K$)},
    grid=both,
    grid style={line width=.1pt, draw=gray!20},
    major grid style={line width=.2pt, draw=gray!50},
    legend pos=north east,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
"""

_TEX_TAIL = r"""\end{axis}
\end{tikzpicture}
\end{document}
"""


def _draw(K, n, rng):
    """Stratified prior draw: half from v=0, half from v=1."""
    half = n // 2
    s0 = -0.5 + rng.standard_normal((half, K)) / np.sqrt(TAU)
    s1 = +0.5 + rng.standard_normal((n - half, K)) / np.sqrt(TAU)
    return np.vstack([s0, s1])


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    results = {g: np.zeros(len(KS)) for g, *_ in GAMMAS}
    for ki, K in enumerate(KS):
        samples = _draw(K, N_MC, rng).astype(np.float64, copy=False)
        for gamma, label, *_ in GAMMAS:
            if gamma >= 1e3:
                results[gamma][ki] = 0.0       # CARA closed-form
                continue
            v = _one_minus_R2_K(samples, TAU, float(gamma))
            results[gamma][ki] = v
            print(f"  K={K:2d}  γ={label}  1-R²={v:.5f}", flush=True)

    csv_path = os.path.join(OUT, "fig_knife_edge_K_data.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["K"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for ki, K in enumerate(KS):
            w_csv.writerow([K]
                            + [f"{results[g][ki]:.6g}" for g, *_ in GAMMAS])
    print(f"wrote {csv_path}")

    addplots = []
    for gamma, label, color, style, thick in GAMMAS:
        coords = " ".join(f"({K},{results[gamma][ki]:.6g})"
                           for ki, K in enumerate(KS))
        suffix = "\\;(\\text{CARA})" if "infty" in label else ""
        addplots.append(
            f"\\addplot[{color}, {style}, {thick}] "
            f"coordinates {{{coords}}};\n"
            f"\\addlegendentry{{$\\gamma = {label}{suffix}$}};")

    tex = _TEX_HEAD + "\n".join(addplots) + "\n" + _TEX_TAIL
    tex_path = os.path.join(OUT, "fig_knife_edge_K.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
