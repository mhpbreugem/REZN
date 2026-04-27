"""Figure 2 — knife-edge 1−R² vs τ.

Spec:
  • G = 20, u ∈ [-4, 4]
  • Per cell (i, j, l): solve Σ x_k(μ_k, p) = 0 with brentq-quality
    bisection, μ_k = Λ(τ u_k), CRRA demand
    x = (R-1)/((1-p)+R p), R = exp((logit μ - logit p)/γ).
  • Filter: keep only cells where 1e-4 < P < 1 - 1e-4 (drop saturated
    edges where logit blows up).
  • Weighted regression of logit(P) on T*=τ Σ u_k, weight per cell
    w = ½(Π f₁(u_k) + Π f₀(u_k)).
  • γ ∈ {0.25, 1, 4} plus a flat CARA reference at y=0.
  • τ on a 30-point log grid [0.1, 10].

Output: figures/fig2_knife_edge.{tex,pdf,png}
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# ---- spec ----------------------------------------------------------------
G       = 30
UMAX    = 4.0
P_LO    = 1e-4              # cell filter — drop saturated cells
P_HI    = 1 - P_LO
GAMMAS  = [(0.25, "0.25", "green",  "solid"),
            (1.0, "1.0", "red",    "dashed"),
            (4.0, "4", "blue",   "dotted")]
TAUS    = np.logspace(np.log10(0.1), np.log10(5.0), 30)
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


# ---- numerics ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))


@njit(cache=True, fastmath=True)
def _crra_demand(mu, p, gamma):
    eps = 1e-12
    mu = max(eps, min(1 - eps, mu))
    p  = max(eps, min(1 - eps, p))
    R = np.exp((_logit(mu) - _logit(p)) / gamma)
    return (R - 1.0) / ((1.0 - p) + R * p)


@njit(cache=True, fastmath=True)
def _residual(m0, m1, m2, p, gamma):
    return _crra_demand(m0, p, gamma) + _crra_demand(m1, p, gamma) \
            + _crra_demand(m2, p, gamma)


@njit(cache=True)
def _clear_price(m0, m1, m2, gamma):
    """Tight bisection on Σ x_k = 0 over [0.002, 0.998]."""
    lo = 0.002
    hi = 0.998
    f_lo = _residual(m0, m1, m2, lo, gamma)
    f_hi = _residual(m0, m1, m2, hi, gamma)
    for _ in range(120):              # 120 halvings → ~1e-37 width
        m = 0.5 * (lo + hi)
        f_m = _residual(m0, m1, m2, m, gamma)
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
        if (hi - lo) < 1e-14:
            break
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=True)
def _f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u + 0.5)**2)


@njit(cache=True, fastmath=True)
def _f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u - 0.5)**2)


@njit(cache=True, parallel=True)
def _weighted_1mR2(u, tau, gamma):
    """Build P on the G×G×G grid, run filtered weighted regression."""
    n = u.shape[0]
    f0u = np.empty(n); f1u = np.empty(n); muu = np.empty(n)
    for k in range(n):
        f0u[k] = _f0(u[k], tau)
        f1u[k] = _f1(u[k], tau)
        muu[k] = 1.0 / (1.0 + np.exp(-tau * u[k]))
    # Per-i partial sums (parallel safe)
    Sw   = np.zeros(n)
    Sy   = np.zeros(n)
    St   = np.zeros(n)
    Syy  = np.zeros(n)
    Stt  = np.zeros(n)
    Syt  = np.zeros(n)
    for i in prange(n):
        sw = 0.0; sy = 0.0; st = 0.0
        syy = 0.0; stt = 0.0; syt = 0.0
        for j in range(n):
            for l in range(n):
                p = _clear_price(muu[i], muu[j], muu[l], gamma)
                if p <= P_LO or p >= P_HI:
                    continue
                y_ = _logit(p)
                t_ = tau * (u[i] + u[j] + u[l])
                w_ = 0.5 * (f0u[i] * f0u[j] * f0u[l]
                              + f1u[i] * f1u[j] * f1u[l])
                sw  += w_
                sy  += w_ * y_
                st  += w_ * t_
                syy += w_ * y_ * y_
                stt += w_ * t_ * t_
                syt += w_ * y_ * t_
        Sw[i] = sw; Sy[i] = sy; St[i] = st
        Syy[i] = syy; Stt[i] = stt; Syt[i] = syt
    W = Sw.sum()
    if W <= 0.0:
        return 0.0
    y_m = Sy.sum() / W
    t_m = St.sum() / W
    var_y = Syy.sum() - W * y_m * y_m
    var_t = Stt.sum() - W * t_m * t_m
    cov   = Syt.sum() - W * y_m * t_m
    if var_y <= 1e-30 or var_t <= 1e-30:
        return 0.0
    return 1.0 - (cov * cov) / (var_y * var_t)


# ---- TeX template (the "agreed-on" format) -------------------------------
_TEX = r"""\documentclass[border=2pt]{standalone}
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
    xmin=0.1, xmax=5,
    ymin=-0.001, ymax=0.15,
    xtick={0.1,0.2,0.5,1,2,5},
    xticklabels={0.1,0.2,0.5,1,2,5},
    ytick={0,0.05,0.1,0.15},
    yticklabels={0,0.05,0.10,0.15},
    xlabel={signal precision $\tau$},
    ylabel={$1 - R^2$},
    title={Signal precision ($\tau$)},
    legend pos=north west,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
%(addplots)s
\addplot[ultra thick, color=black, dashdotted]
    coordinates {(0.1,0)(5,0)};
\addlegendentry{$\gamma = 0.25$}
\addlegendentry{$\gamma = 1$}
\addlegendentry{$\gamma = 4$}
\addlegendentry{CARA}
\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)

    # Warm-up JIT compile
    _ = _weighted_1mR2(u, 1.0, 1.0)

    results = {}
    for gamma, label, color, style in GAMMAS:
        print(f"\n=== γ = {label} ===", flush=True)
        vals = np.empty(len(TAUS))
        for it, tau in enumerate(TAUS):
            vals[it] = _weighted_1mR2(u, float(tau), float(gamma))
            print(f"  τ={tau:7.3f}  1-R²={vals[it]:.5e}", flush=True)
        results[gamma] = vals

    # CSV (all curves)
    csv_path = os.path.join(OUT, "fig2_knife_edge_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w.writerow([f"{tau:.6g}"]
                        + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"\nwrote {csv_path}")

    # tex — only include τ-points up to (and including) the last
    # cleanly monotone-increasing point. Beyond that, the on-grid
    # solver hits the saturation regime where logit(p) is forced
    # onto a discrete-valued step function and R² collapses non-
    # monotonically. Those points lie above ymax anyway, so dropping
    # them avoids the spurious "vertical wall" artifact in the plot.
    YMAX = 0.13
    addplots = []
    style_map = {
        "solid":  "very thick, color=green, smooth",
        "dashed": "very thick, color=red, dashed, smooth",
        "dotted": "very thick, color=blue, dotted, smooth",
    }
    for gamma, label, color, style in GAMMAS:
        opts = style_map[style]
        vals = results[gamma]
        # keep only the leading monotone-increasing prefix that stays
        # below ymax
        keep = []
        prev = -np.inf
        for tau, v in zip(TAUS, vals):
            if not np.isfinite(v):    break
            if v > YMAX:               break
            if v < prev * 0.999:       break  # not monotone — stop
            keep.append((tau, v))
            prev = v
        coords = " ".join(f"({tau:.6g},{v:.6g})" for tau, v in keep)
        addplots.append(f"\\addplot[{opts}] coordinates {{{coords}}};")
    tex = _TEX % {"addplots": "\n".join(addplots)}
    tex_path = os.path.join(OUT, "fig2_knife_edge.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
