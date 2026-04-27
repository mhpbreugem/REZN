"""Knife-edge figure with 2-state lognormal payoff.

Payoff: v ∈ {v_L, v_H} = {e^{-Δ}, e^{+Δ}} with Δ = 1.  Both payoffs strictly
positive, so CRRA utility well-defined (no zero-wealth state).  Same prior
P(v=v_H) = ½ and same signal model as the binary case (s_k = 1{v=v_H} − ½ + ε_k,
ε_k ~ N(0, 1/τ); centered u_k drives the posterior μ_k = Λ(τu_k)).

Demand (CRRA, W=1, single risky asset paying v):
    FOC :  μ (1 + x(v_H − p))^{−γ} (v_H − p)
         + (1−μ)(1 + x(v_L − p))^{−γ} (v_L − p)  =  0
solved by Newton on x bracketed by the no-bankruptcy interval
[−1/(v_H−p), 1/(p−v_L)] when v_L < p < v_H.

Market clearing : Σ x_k(μ_k, p) = 0   ⇒  bisection on p ∈ [v_L, v_H].

CARA reference (γ = ∞ proxy via γ = 1000) plotted as a flat line at zero
(closed-form: CARA + binary state ⇒ logit-linear aggregator ⇒ FR).

Output: figures/fig_knife_edge_lognormal.{tex,pdf,png,csv}
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numba import njit, prange


N_GH    = 40
DELTA   = 1.0                                          # log payoff spread
V_L     = float(np.exp(-DELTA))
V_H     = float(np.exp(+DELTA))
GAMMAS  = [(0.3, "0.3",      "green",  "solid",     "very thick"),
            (1.0, "1.0",      "red",    "dashed",    "very thick"),
            (3.0, "3.0",      "blue",   "dotted",    "very thick"),
            (1e3, "\\infty", "black",  "dash dot",  "ultra thick")]
TAUS    = np.logspace(np.log10(0.1), np.log10(20.0), 30)
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "figures")


# ---- numerics --------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _foc(x, mu, p, gamma):
    W_H = 1.0 + x * (V_H - p)
    W_L = 1.0 + x * (V_L - p)
    if W_H <= 1e-12 or W_L <= 1e-12:
        return np.nan
    return mu * W_H**(-gamma) * (V_H - p) \
            + (1 - mu) * W_L**(-gamma) * (V_L - p)


@njit(cache=True, fastmath=True)
def _foc_prime(x, mu, p, gamma):
    W_H = 1.0 + x * (V_H - p)
    W_L = 1.0 + x * (V_L - p)
    if W_H <= 1e-12 or W_L <= 1e-12:
        return np.nan
    return -gamma * mu * W_H**(-gamma - 1) * (V_H - p)**2 \
            - gamma * (1 - mu) * W_L**(-gamma - 1) * (V_L - p)**2


@njit(cache=True)
def _demand(mu, p, gamma):
    """Find x s.t. FOC=0, bracketed by no-bankruptcy bounds."""
    if p <= V_L + 1e-12 or p >= V_H - 1e-12:
        return 0.0  # corner case, market won't clear here anyway
    x_low  = -1.0 / (V_H - p) + 1e-9
    x_high = +1.0 / (p - V_L) - 1e-9
    f_low  = _foc(x_low,  mu, p, gamma)
    f_high = _foc(x_high, mu, p, gamma)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        return 0.0
    if f_low * f_high > 0.0:
        # Same sign throughout the interval — corner solution
        return x_high if abs(f_low) > abs(f_high) else x_low
    # Newton with bisection fallback
    x = 0.0
    for _ in range(40):
        f = _foc(x, mu, p, gamma)
        fp = _foc_prime(x, mu, p, gamma)
        if not np.isfinite(f) or not np.isfinite(fp) or fp == 0.0:
            break
        x_new = x - f / fp
        if x_new <= x_low or x_new >= x_high:
            break
        if abs(x_new - x) < 1e-12:
            x = x_new
            break
        x = x_new
    if abs(_foc(x, mu, p, gamma)) > 1e-9:
        # Bisect
        a, b = x_low, x_high
        f_a, f_b = f_low, f_high
        for _ in range(80):
            m = 0.5 * (a + b)
            f_m = _foc(m, mu, p, gamma)
            if not np.isfinite(f_m):
                f_m = 0.0
            if abs(f_m) < 1e-12 or (b - a) < 1e-12:
                return m
            if f_a * f_m <= 0.0:
                b, f_b = m, f_m
            else:
                a, f_a = m, f_m
        return 0.5 * (a + b)
    return x


@njit(cache=True, fastmath=True)
def _residual(mus, p, gamma):
    return _demand(mus[0], p, gamma) \
            + _demand(mus[1], p, gamma) \
            + _demand(mus[2], p, gamma)


@njit(cache=True)
def _clear_price(mus, gamma):
    lo = V_L + 1e-9
    hi = V_H - 1e-9
    f_lo = _residual(mus, lo, gamma)
    f_hi = _residual(mus, hi, gamma)
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f_m = _residual(mus, m, gamma)
        if (hi - lo) < 1e-12 or f_m == 0.0:
            return m
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))


@njit(cache=True, fastmath=True, parallel=True)
def _one_minus_R2(u0, u1, w_GH, tau, gamma):
    """GH-quadrature 1-R² of (log p) on T*=τ·Σu, weighted by ½(f₀³+f₁³)."""
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
                    # Use log p (lognormal => log-payoff is the natural scale)
                    yv = np.log(p)
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


# ---- TeX template ---------------------------------------------------------
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
    xmin=0.1, xmax=20,
    ymin=-0.001, ymax=0.15,
    xtick={0.1,0.2,0.5,1,2,5,10,20},
    xticklabels={0.1,0.2,0.5,1,2,5,10,20},
    ytick={0,0.05,0.1,0.15},
    yticklabels={0,0.05,0.10,0.15},
    xlabel={signal precision $\tau$},
    ylabel={$1 - R^2$},
    title={Signal precision ($\tau$), 2-state lognormal payoff},
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
        print(f"=== γ = {label} ===", flush=True)
        if gamma >= 1e3:
            vals = np.zeros_like(TAUS)        # CARA: knife-edge zero
        else:
            vals = np.empty(len(TAUS))
            for it, tau in enumerate(TAUS):
                u0 = -0.5 + xi / np.sqrt(tau)
                u1 = +0.5 + xi / np.sqrt(tau)
                vals[it] = _one_minus_R2(u0, u1, w_GH, float(tau),
                                           float(gamma))
                print(f"  τ={tau:7.3f}  1-R²={vals[it]:.5f}",
                       flush=True)
        results[gamma] = vals

    csv_path = os.path.join(OUT, "fig_knife_edge_lognormal_data.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w_csv.writerow([f"{tau:.6g}"]
                            + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"wrote {csv_path}")

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
    tex_path = os.path.join(OUT, "fig_knife_edge_lognormal.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
