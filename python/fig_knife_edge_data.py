"""Compute 1−R² vs τ at three CRRA risk-aversion levels (plus CARA
reference) using Gauss-Hermite quadrature.

Why GH instead of a uniform u-grid:
  The integrand is Gaussian-weighted (weights ∝ ½(f₀³+f₁³)). On a
  uniform grid u = linspace(-4, 4, G), the relevant active region
  shrinks like 1/√τ, so the effective resolution drops as τ grows
  and the discretised 1−R² develops spurious wobbles.

  Gauss-Hermite places nodes at the optimal points for ∫ g(u) e^{-u²/2}du
  with spectral convergence — 50-80 nodes per state suffice for
  machine-precision integrals across the full τ ∈ [0.1, 10] range.

For each (γ, τ):
  1. GH nodes for state v ∈ {0,1}: u^(v)_n = (v − ½) + ξ_n / √τ,
     weights w^(v)_n from numpy.polynomial.hermite_e.hermegauss
     (probabilist Hermite, weight e^{−ξ²/2}/√(2π)).
  2. For each (i, j, l) and each combination of (v_i, v_j, v_l) ∈
     {0,1}³, evaluate the no-learning clearing price; weight by
     ⅛ · w^(v_i) w^(v_j) w^(v_l) · 4 · 1[v_i=v_j=v_l=v]   (the prior
     ½(f_0³ + f_1³) is the equal-mixture marginal, equivalent to
     summing only the diagonal terms v_i=v_j=v_l).
  3. Weighted regression of logit(P) on T*=τ·Σu_k; report 1−R².

Outputs (overwrite previous):
  figures/fig_knife_edge.tex             pgfplots standalone
  figures/fig_knife_edge.{pdf,png}       compiled previews
  figures/fig_knife_edge_data.csv        machine-friendly table
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numba import njit, prange


# ---- Setup -----------------------------------------------------------------
N_GH    = 80                              # Gauss-Hermite nodes per state
GAMMAS  = [(0.2, "0.2",      "green",  "solid"),
            (1.0, "1.0",      "red",    "dashed"),
            (5.0, "5.0",      "blue",   "dotted"),
            (1e3, "\\infty", "black",  "dash dot")]
TAUS    = np.logspace(np.log10(0.1), np.log10(10.0), 30)
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "figures")


# ---- Numerics --------------------------------------------------------------
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
    """GH-quadrature 1−R² of logit(P) on T*=τ·Σu, weighted by
    ½(f₀(u_i)f₀(u_j)f₀(u_l) + f₁(...)).

    u0[n] = (v=0)-state nodes, u1[n] = (v=1)-state nodes,
    w_GH[n] = node weights (probabilist Hermite, ∑w=1).

    The weight ½(f₀³+f₁³) corresponds to summing only the
    diagonal v_i=v_j=v_l=v=0 and v=1 contributions, each with
    pre-factor ½. So we compute two GH triple-sums and average.
    """
    n = u0.shape[0]
    # Per-i partial accumulators (parallel-safe)
    Sw    = np.zeros(n)
    Swy   = np.zeros(n)
    Swy2  = np.zeros(n)
    Swt   = np.zeros(n)
    Swt2  = np.zeros(n)
    Swty  = np.zeros(n)

    for i in prange(n):
        sw = 0.0; swy = 0.0; swy2 = 0.0
        swt = 0.0; swt2 = 0.0; swty = 0.0
        # Combine v=0 and v=1 diagonals (the ½ factor folds in below)
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


def main():
    os.makedirs(OUT, exist_ok=True)

    # Probabilist Hermite nodes & weights: ∫ g(ξ) e^{−ξ²/2} dξ ≈ Σ w_n g(ξ_n)
    xi, w_raw = hermegauss(N_GH)
    w_GH = w_raw / np.sqrt(2 * np.pi)        # normalise so Σ w = 1

    results = {}
    for gamma, label, color, style in GAMMAS:
        print(f"\n=== γ = {label} ===", flush=True)
        vals = np.empty(len(TAUS))
        for it, tau in enumerate(TAUS):
            inv_sqrt_tau = 1.0 / np.sqrt(tau)
            u0 = -0.5 + xi * inv_sqrt_tau
            u1 = +0.5 + xi * inv_sqrt_tau
            vals[it] = _one_minus_R2_GH(u0, u1, w_GH,
                                          float(tau), float(gamma))
        results[gamma] = vals
        coords = " ".join(f"({tau:.6g},{v:.6g})"
                          for tau, v in zip(TAUS, vals))
        print(f"\\addplot[{color},{style},very thick] coordinates "
               f"{{{coords}}};")
        print(f"\\addlegendentry{{$\\gamma = {label}$}};")

    # CSV
    csv_path = os.path.join(OUT, "fig_knife_edge_data.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w_csv.writerow([f"{tau:.6g}"]
                            + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"\nwrote {csv_path}", flush=True)

    # PGFPlots tex
    addplots = []
    for gamma, label, color, style in GAMMAS:
        coords = " ".join(f"({tau:.6g},{v:.6g})"
                          for tau, v in zip(TAUS, results[gamma]))
        suffix = "\\;(\\text{CARA})" if "infty" in label else ""
        addplots.append(
            f"\\addplot[{color}, {style}, very thick] "
            f"coordinates {{{coords}}};\n"
            f"\\addlegendentry{{$\\gamma = {label}{suffix}$}};")

    tex = (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "    width=10cm, height=7cm,\n"
        "    xmode=log,\n"
        "    xmin=0.08, xmax=10,\n"
        "    ymin=-0.001, ymax=0.13,\n"
        "    xlabel={signal precision $\\tau$},\n"
        "    ylabel={$1 - R^2$ of logit$(p)$ vs $T^*$},\n"
        "    grid=both,\n"
        "    grid style={line width=.1pt, draw=gray!20},\n"
        "    major grid style={line width=.2pt, draw=gray!50},\n"
        "    legend pos=north west,\n"
        "    legend style={font=\\footnotesize},\n"
        "    smooth,\n"
        "]\n"
        + "\n".join(addplots) + "\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{document}\n"
    )
    tex_path = os.path.join(OUT, "fig_knife_edge.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
