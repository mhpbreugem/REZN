"""Figure 3 — CARA vs CRRA Contour (§4.1, central figure).

Two side-by-side panels showing the level set {P(1, u₂, u₃) = p_obs}
that agent 1 sees when she fixes u₁=1 and observes the price at the
realisation (u₁, u₂, u₃) = (1, −1, 1). Each point on the contour is
shaded by

    T* = τ (u₁ + u₂ + u₃).

Under CARA, the level set is a straight line and every point has the
same T* (single shade). Under CRRA, the level set is curved and T*
varies along it (shade gradient) — that gradient is the visual
signature of partial revelation.

No-learning prices only — no contour iteration needed. For each u₂ on
a fine 1D grid we root-find u₃ such that the analytic market-clearing
price equals p_obs.

Output:
  figures/fig3_contour.{tex,pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
from numba import njit
from scipy.optimize import brentq

# ---- spec ---------------------------------------------------------------
TAU      = 2.0
N_TRACE  = 500             # u₂ samples to trace each contour
U1_FIXED = 1.0
U2_REAL, U3_REAL = -1.0, 1.0
GAMMA_CRRA = 0.5
A_CARA   = 1.0              # absolute risk aversion in CARA panel
UMAX     = 4.0
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


# ---- numerics -----------------------------------------------------------
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
    lo = 0.002; hi = 0.998
    f_lo = _residual(m0, m1, m2, lo, gamma)
    f_hi = _residual(m0, m1, m2, hi, gamma)
    for _ in range(120):
        m = 0.5 * (lo + hi)
        f_m = _residual(m0, m1, m2, m, gamma)
        if f_lo * f_m < 0.0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
        if (hi - lo) < 1e-14:
            break
    return 0.5 * (lo + hi)


def price_at_crra(u1, u2, u3, tau, gamma):
    sig = lambda u: 1.0 / (1.0 + np.exp(-tau * u))
    return _clear_price(sig(u1), sig(u2), sig(u3), gamma)


def price_at_cara(u1, u2, u3, tau):
    """Closed-form CARA market clearing for binary v.
    x_k = (logit μ_k − logit p)/a, equal a → logit p = (1/K)Σ logit μ_k.
    For binary v, logit μ_k = τ u_k, so logit p = τ(u1+u2+u3)/3.
    """
    return 1.0 / (1.0 + np.exp(-tau * (u1 + u2 + u3) / 3.0))


def trace_contour(price_fn, p_obs, tau, u1=U1_FIXED,
                   n_trace=N_TRACE, umax=UMAX):
    """For each u₂ on a fine 1D grid, root-find u₃ such that
    price_fn(u1, u₂, u₃) = p_obs. Returns array of (u₂, u₃, T*)."""
    u2_grid = np.linspace(-umax, umax, n_trace)
    pts = []
    for u2 in u2_grid:
        f_lo = price_fn(u1, u2, -umax) - p_obs
        f_hi = price_fn(u1, u2,  umax) - p_obs
        if f_lo * f_hi >= 0.0:
            continue
        u3_star = brentq(lambda u3: price_fn(u1, u2, u3) - p_obs,
                          -umax, umax, xtol=1e-10)
        T_star = tau * (u1 + u2 + u3_star)
        pts.append((u2, u3_star, T_star))
    return np.array(pts) if pts else np.zeros((0, 3))


# ---- TeX template -------------------------------------------------------
_TEX = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\definecolor{red}{rgb}{0.7,0.11,0.11}
\definecolor{blue}{rgb}{0.0,0.20,0.42}
\definecolor{green}{rgb}{0.11,0.35,0.02}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=2 by 1, horizontal sep=1.8cm},
    width=8cm, height=8cm,
    xmin=-2, xmax=2, ymin=-2, ymax=2,
    xtick={-2,-1,0,1,2}, ytick={-2,-1,0,1,2},
    xlabel={$u_2$}, ylabel={$u_3$},
    enlargelimits=false,
    axis on top,
    point meta min=%(tmin)s, point meta max=%(tmax)s,
    colormap={greys}{rgb255=(235,235,235); rgb255=(40,40,40)},
]

\nextgroupplot[title={CARA}]
%(cara_addplot)s
\addplot[only marks, mark=star, mark size=4pt,
         color=black, line width=1pt]
    coordinates {(%(u2real)s, %(u3real)s)};

\nextgroupplot[title={CRRA  ($\gamma=0.5$)},
                colorbar, colorbar style={
                    title={$T^*$},
                    width=5pt, font=\scriptsize}]
%(crra_addplot)s
\addplot[only marks, mark=star, mark size=4pt,
         color=black, line width=1pt]
    coordinates {(%(u2real)s, %(u3real)s)};

\end{groupplot}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)

    cara_fn = lambda u1, u2, u3: price_at_cara(u1, u2, u3, TAU)
    crra_fn = lambda u1, u2, u3: price_at_crra(u1, u2, u3, TAU,
                                                  GAMMA_CRRA)

    p_obs_cara = cara_fn(U1_FIXED, U2_REAL, U3_REAL)
    p_obs_crra = crra_fn(U1_FIXED, U2_REAL, U3_REAL)
    print(f"p_obs CARA (exact)   = {p_obs_cara:.6f}")
    print(f"p_obs CRRA γ={GAMMA_CRRA} = {p_obs_crra:.6f}")

    print("Tracing CARA contour …", flush=True)
    pts_cara = trace_contour(cara_fn, p_obs_cara, TAU)
    print(f"  {len(pts_cara)} points,  T* in "
           f"[{pts_cara[:,2].min():.3f}, {pts_cara[:,2].max():.3f}]")

    print("Tracing CRRA contour …", flush=True)
    pts_crra = trace_contour(crra_fn, p_obs_crra, TAU)
    print(f"  {len(pts_crra)} points,  T* in "
           f"[{pts_crra[:,2].min():.3f}, {pts_crra[:,2].max():.3f}]")

    # CSV
    csv_path = os.path.join(OUT, "fig3_contour_data.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["panel", "u2", "u3", "T_star"])
        for u2, u3, T in pts_cara:
            w.writerow(["CARA", f"{u2:.6g}", f"{u3:.6g}", f"{T:.6g}"])
        for u2, u3, T in pts_crra:
            w.writerow(["CRRA", f"{u2:.6g}", f"{u3:.6g}", f"{T:.6g}"])
    print(f"wrote {csv_path}")

    # tex with colour-mapped scatter on each panel
    tmin = min(pts_cara[:, 2].min(), pts_crra[:, 2].min())
    tmax = max(pts_cara[:, 2].max(), pts_crra[:, 2].max())

    def addplot_block(pts):
        lines = ["\\addplot[scatter, only marks, "
                 "scatter src=explicit, "
                 "mark=*, mark size=0.7pt]"
                 " table[meta=T,col sep=space] {"
                 "u2 u3 T"]
        for u2, u3, T in pts:
            lines.append(f"{u2:.6g} {u3:.6g} {T:.6g}")
        lines.append("};")
        return "\n".join(lines)

    tex = _TEX % {
        "tmin": f"{tmin:.4f}",
        "tmax": f"{tmax:.4f}",
        "u2real": f"{U2_REAL:.1f}",
        "u3real": f"{U3_REAL:.1f}",
        "cara_addplot": addplot_block(pts_cara),
        "crra_addplot": addplot_block(pts_crra),
    }
    tex_path = os.path.join(OUT, "fig3_contour.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
