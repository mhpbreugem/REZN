"""Knife-edge figure: 1−R² vs τ at γ ∈ {0.2, 1, 5, ∞}, K = 3 agents.

Gauss-Hermite quadrature (N=80 nodes/state) for Gaussian-weighted
integration. Output is a self-contained PGFPlots tex per the BC20
specification: 8cm × 8cm, log-x [0.1, 10], custom colors, smooth
linework, legend NW footnotesize.
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numba import njit, prange


N_GH    = 80
GAMMAS  = [(0.2, "0.2",      "green",  "solid",     "very thick"),
            (1.0, "1.0",      "red",    "dashed",    "very thick"),
            (5.0, "5.0",      "blue",   "dotted",    "very thick"),
            (1e3, "\\infty", "black",  "dash dot",  "ultra thick")]
TAUS    = np.logspace(np.log10(0.1), np.log10(10.0), 30)
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "figures")


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
def _residual(mus, p, gamma):
    return _crra_demand(mus[0], p, gamma) \
            + _crra_demand(mus[1], p, gamma) \
            + _crra_demand(mus[2], p, gamma)


@njit(cache=True)
def _clear_price(mus, gamma):
    lo, hi = 1e-9, 1.0 - 1e-9
    f_lo = _residual(mus, lo, gamma)
    f_hi = _residual(mus, hi, gamma)
    if f_lo == 0.0: return lo
    if f_hi == 0.0: return hi
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f_m = _residual(mus, m, gamma)
        if (hi - lo) < 1e-13 or f_m == 0.0:
            return m
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=True, parallel=True)
def _one_minus_R2_GH(u0, u1, w_GH, tau, gamma):
    n = u0.shape[0]
    Sw    = np.zeros(n)
    Swy   = np.zeros(n)
    Swy2  = np.zeros(n)
    Swt   = np.zeros(n)
    Swt2  = np.zeros(n)
    Swty  = np.zeros(n)
    for i in prange(n):
        sw = 0.0; swy = 0.0; swy2 = 0.0
        swt = 0.0; swt2 = 0.0; swty = 0.0
        for vi in range(2):
            ui = u0[i] if vi == 0 else u1[i]
            wi = w_GH[i]
            mui = 1.0 / (1.0 + np.exp(-tau * ui))
            for j in range(n):
                uj = u0[j] if vi == 0 else u1[j]
                wj = w_GH[j]
                muj = 1.0 / (1.0 + np.exp(-tau * uj))
                for l in range(n):
                    ul = u0[l] if vi == 0 else u1[l]
                    wl = w_GH[l]
                    mul = 1.0 / (1.0 + np.exp(-tau * ul))
                    mus = np.array([mui, muj, mul])
                    p = _clear_price(mus, gamma)
                    yv = _logit(p)
                    tv = tau * (ui + uj + ul)
                    wv = 0.5 * wi * wj * wl
                    sw   += wv
                    swy  += wv * yv
                    swy2 += wv * yv * yv
                    swt  += wv * tv
                    swt2 += wv * tv * tv
                    swty += wv * tv * yv
        Sw[i]   = sw
        Swy[i]  = swy
        Swy2[i] = swy2
        Swt[i]  = swt
        Swt2[i] = swt2
        Swty[i] = swty
    W   = Sw.sum()
    if W <= 0.0:
        return 0.0
    y_m = Swy.sum() / W
    T_m = Swt.sum() / W
    Syy = Swy2.sum() - W * y_m * y_m
    STT = Swt2.sum() - W * T_m * T_m
    SyT = Swty.sum() - W * T_m * y_m
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
    xmode=log,
    xmin=0.1, xmax=10,
    ymin=-0.001, ymax=0.13,
    xtick={0.1,0.2,0.5,1,2,5,10},
    xticklabels={0.1,0.2,0.5,1,2,5,10},
    xlabel={signal precision $\tau$},
    ylabel={$1 - R^2$},
    title={Signal precision ($\tau$)},
    grid=both,
    grid style={line width=.1pt, draw=gray!20},
    major grid style={line width=.2pt, draw=gray!50},
    legend pos=north west,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
"""

_TEX_TAIL = r"""\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    xi, w_raw = hermegauss(N_GH)
    w_GH = w_raw / np.sqrt(2 * np.pi)

    results = {}
    for gamma, label, color, style, thick in GAMMAS:
        print(f"\n=== γ = {label} ===", flush=True)
        if gamma >= 1e3:
            # CARA: closed-form, 1-R² ≡ 0 across τ
            vals = np.zeros_like(TAUS)
        else:
            vals = np.empty(len(TAUS))
            for it, tau in enumerate(TAUS):
                u0 = -0.5 + xi / np.sqrt(tau)
                u1 = +0.5 + xi / np.sqrt(tau)
                vals[it] = _one_minus_R2_GH(u0, u1, w_GH,
                                              float(tau), float(gamma))
                print(f"  τ={tau:7.3f}  1-R²={vals[it]:.6f}", flush=True)
        results[gamma] = vals

    csv_path = os.path.join(OUT, "fig_knife_edge_data.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w_csv.writerow([f"{tau:.6g}"]
                            + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"\nwrote {csv_path}", flush=True)

    addplots = []
    for gamma, label, color, style, thick in GAMMAS:
        coords = " ".join(f"({tau:.6g},{v:.6g})"
                           for tau, v in zip(TAUS, results[gamma]))
        suffix = "\\;(\\text{CARA})" if "infty" in label else ""
        addplots.append(
            f"\\addplot[{color}, {style}, {thick}] "
            f"coordinates {{{coords}}};\n"
            f"\\addlegendentry{{$\\gamma = {label}{suffix}$}};")

    tex = _TEX_HEAD + "\n".join(addplots) + "\n" + _TEX_TAIL
    tex_path = os.path.join(OUT, "fig_knife_edge.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
