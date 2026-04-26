"""Compute 1-R² vs τ at three CRRA risk-aversion levels (plus CARA
reference) for the knife-edge figure. No-learning equilibrium at G=20,
weighted regression with prior posterior weight w = ½(f₀³ + f₁³).

For each (γ, τ):
  1. u = linspace(-4, 4, 20)
  2. Private posteriors μ_k = Λ(τ u_k) for k = i, j, l
  3. Solve Σ x_k(μ_k, p) = 0 for p (CRRA market clearing)
  4. Weighted regression of logit(P) on T* = τ Σ u_k

Outputs:
  • PGFPlots coordinate blocks printed to stdout (copy-paste into the
    other chat's fig_knife_edge.tex at /mnt/user-data/outputs/).
  • figures/fig_knife_edge.tex — full standalone PGFPlots tex updated
    with the new coords, BC20-ish styling: green solid (γ=0.2),
    red dashed (γ=1), blue dotted (γ=5), black dashdotted (CARA).
  • figures/fig_knife_edge_data.csv — same numbers, machine-friendly.
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit


# ---- Setup -----------------------------------------------------------------
G       = 20
UMAX    = 4.0
GAMMAS  = [(0.2, "0.2",          "green",  "solid"),
            (1.0, "1.0",          "red",    "dashed"),
            (5.0, "5.0",          "blue",   "dotted"),
            (1e3, "\\infty",     "black",  "dash dot")]
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


@njit(cache=True, fastmath=True)
def _f0(u, tau):
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau * 0.5 * (u + 0.5)**2)


@njit(cache=True, fastmath=True)
def _f1(u, tau):
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau * 0.5 * (u - 0.5)**2)


@njit(cache=True)
def _one_minus_R2(u, tau, gamma):
    """No-learning 1-R² of logit(P) on T*=τ·Σu, weighted by ½(f₀³+f₁³)."""
    n = u.shape[0]
    N = n * n * n
    y = np.empty(N)
    T = np.empty(N)
    w = np.empty(N)
    k = 0
    for i in range(n):
        m_i = 1.0 / (1.0 + np.exp(-tau * u[i]))
        f0i = _f0(u[i], tau);  f1i = _f1(u[i], tau)
        for j in range(n):
            m_j = 1.0 / (1.0 + np.exp(-tau * u[j]))
            f0j = _f0(u[j], tau);  f1j = _f1(u[j], tau)
            for l in range(n):
                m_l = 1.0 / (1.0 + np.exp(-tau * u[l]))
                f0l = _f0(u[l], tau);  f1l = _f1(u[l], tau)
                mus = np.array([m_i, m_j, m_l])
                p = _clear_price(mus, gamma)
                y[k] = _logit(p)
                T[k] = tau * (u[i] + u[j] + u[l])
                w[k] = 0.5 * (f0i * f0j * f0l + f1i * f1j * f1l)
                k += 1
    W   = w.sum()
    y_m = (w * y).sum() / W
    T_m = (w * T).sum() / W
    Syy = (w * (y - y_m)**2).sum()
    STT = (w * (T - T_m)**2).sum()
    SyT = (w * (T - T_m) * (y - y_m)).sum()
    if Syy <= 0.0 or STT <= 0.0:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


# ---- Driver ----------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)

    results = {}
    for gamma, label, color, style in GAMMAS:
        print(f"\n=== γ = {label} ===", flush=True)
        vals = np.empty(len(TAUS))
        for it, tau in enumerate(TAUS):
            vals[it] = _one_minus_R2(u, float(tau), float(gamma))
        results[gamma] = vals
        # PGFPlots coords block
        lines = [f"% γ = {label}"]
        for tau, v in zip(TAUS, vals):
            lines.append(f"({tau:.6g},{v:.6g})")
        coord_block = " ".join(lines[1:])
        print(f"\\addplot[{color},{style},very thick] coordinates "
               f"{{{coord_block}}};")
        print(f"\\addlegendentry{{$\\gamma = {label}$}};")

    # CSV
    csv_path = os.path.join(OUT, "fig_knife_edge_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w.writerow([f"{tau:.6g}"]
                        + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"\nwrote {csv_path}", flush=True)

    # PGFPlots standalone TeX
    addplots = []
    for gamma, label, color, style in GAMMAS:
        coords = " ".join(f"({tau:.6g},{v:.6g})"
                          for tau, v in zip(TAUS, results[gamma]))
        suffix = "\\;(\\text{CARA})" if "infty" in label else ""
        addplots.append(
            f"\\addplot[{color}, {style}, very thick] "
            f"coordinates {{{coords}}};\n"
            f"\\addlegendentry{{$\\gamma = {label}{suffix}$}};")
    addplots_str = "\n".join(addplots)

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
        f"{addplots_str}\n"
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
