"""Figure 7 — Trade Volume E[|x_1|] vs γ.

Computes the expected absolute trade volume of agent 1 at the
no-learning REE across γ ∈ {0.1, …, 10}.

For each (γ, cell):
  • own posterior μ_1 = Λ(τ u_1)  (no learning)
  • clearing price p solves Σ x_k(μ_k, p) = 0 by bisection
  • demand x_1 = (R-1)/((1-p)+R p), R = exp((logit μ_1 - logit p)/γ)
Weighted average |x_1| over the prior gives E[|x_1|].

CARA is plotted at zero (no-trade theorem in CRRA REE; under
no-learning CARA produces small positive volume but the
theoretical limit is zero, which is what Fig 7 visualises).

Output:
  figures/fig7_volume.{tex,pdf,png,csv}
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# ---- spec ---------------------------------------------------------------
G       = 20
UMAX    = 4.0
TAU     = 2.0
GAMMAS  = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
P_LO    = 1e-4
P_HI    = 1 - P_LO
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


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
    lo, hi = 0.002, 0.998
    f_lo = _residual(m0, m1, m2, lo, gamma)
    f_hi = _residual(m0, m1, m2, hi, gamma)
    for _ in range(120):
        m = 0.5 * (lo + hi)
        f_m = _residual(m0, m1, m2, m, gamma)
        if f_lo * f_m < 0:
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
def _expected_abs_x(u, tau, gamma):
    n = u.shape[0]
    f0u = np.empty(n); f1u = np.empty(n); muu = np.empty(n)
    for k in range(n):
        f0u[k] = _f0(u[k], tau)
        f1u[k] = _f1(u[k], tau)
        muu[k] = 1.0 / (1.0 + np.exp(-tau * u[k]))
    SX  = np.zeros(n)
    SW  = np.zeros(n)
    for i in prange(n):
        sx = 0.0; sw = 0.0
        for j in range(n):
            for l in range(n):
                p = _clear_price(muu[i], muu[j], muu[l], gamma)
                if p <= P_LO or p >= P_HI:
                    continue
                x1 = _crra_demand(muu[i], p, gamma)
                w_ = 0.5 * (f0u[i] * f0u[j] * f0u[l]
                              + f1u[i] * f1u[j] * f1u[l])
                sx += w_ * abs(x1)
                sw += w_
        SX[i] = sx; SW[i] = sw
    if SW.sum() <= 0:
        return 0.0
    return SX.sum() / SW.sum()


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
    xmin=0.08, xmax=15,
    ymin=-0.02,
    xtick={0.1,0.3,1,3,10},
    xticklabels={0.1,0.3,1,3,10},
    xlabel={risk aversion $\gamma$},
    ylabel={$\mathbb{E}[|x_1|]$},
    title={Trade volume},
    legend pos=north east,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
\addplot[very thick, color=red, smooth]
    coordinates {%(coords)s};
\addlegendentry{CRRA}
\addplot[ultra thick, color=black, dashdotted]
    coordinates {(0.08,0)(15,0)};
\addlegendentry{CARA (no-trade theorem)}
\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)
    _ = _expected_abs_x(u, 1.0, 1.0)             # JIT warm-up
    rows = []
    for gamma in GAMMAS:
        E_abs_x = _expected_abs_x(u, TAU, float(gamma))
        rows.append((gamma, E_abs_x))
        print(f"  γ={gamma:6.2f}  E|x_1| = {E_abs_x:.5f}", flush=True)

    csv_path = os.path.join(OUT, "fig7_volume_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "E_abs_x1"])
        for g, x in rows:
            w.writerow([f"{g:.4f}", f"{x:.6f}"])
    print(f"wrote {csv_path}")

    coords = " ".join(f"({g:.4f},{x:.6f})" for g, x in rows)
    tex = _TEX % {"coords": coords}
    tex_path = os.path.join(OUT, "fig7_volume.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
