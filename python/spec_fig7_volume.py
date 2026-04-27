"""Figure 7 — Trade Volume E[|x_k|] vs γ.

For each γ, solve the converged REE on the production PCHIP+contour
kernel, compute the per-cell agent-1 demand at the equilibrium price
and posterior, and integrate against the prior weight.

CARA reference: explicit demand x = (logit μ - logit p)/a aggregates
to logit p = (1/K) Σ logit μ_k = T*/K (no-learning aggregator).
Under REE all agents update to μ = Λ(T*) and the price moves to
p = Λ(T*); demand x_k = (logit Λ(T*) − logit Λ(T*))/a = 0. CARA →
no-trade theorem in the binary REE. Plotted as 0.

Output:
  figures/fig7_volume.{tex,pdf,png,csv}
"""
from __future__ import annotations
import os
import csv
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh


# ---- spec ---------------------------------------------------------------
TAU      = 2.0
G        = 15
UMAX     = 2.0
GAMMAS   = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


def _logit(p):
    return np.log(p / (1.0 - p))


def _crra_demand(mu, p, gamma):
    eps = 1e-12
    mu = max(eps, min(1 - eps, mu))
    p  = max(eps, min(1 - eps, p))
    R = np.exp((_logit(mu) - _logit(p)) / gamma)
    return (R - 1.0) / ((1.0 - p) + R * p)


def _f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u + 0.5)**2)


def _f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u - 0.5)**2)


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
    ymin=-0.005,
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
\addlegendentry{CARA (no-trade)}
\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)
    taus_arr = np.array([TAU, TAU, TAU])
    Ws = np.array([1.0, 1.0, 1.0])

    f0u = _f0(u, TAU); f1u = _f1(u, TAU)
    rows = []
    last_P = None
    # γ-homotopy from high to low (CARA-like → CRRA)
    for gamma in sorted(GAMMAS, reverse=True):
        gammas_arr = np.array([gamma, gamma, gamma])
        t0 = time.time()
        # Picard α=0.3 (slow, robust) then Anderson polish
        res_p = rp.solve_picard_pchip(
            G, taus_arr, gammas_arr, umax=UMAX,
            maxiters=600, abstol=1e-7, alpha=0.3,
            P_init=last_P)
        finf_p = float(np.abs(res_p["residual"]).max())
        P = res_p["P_star"]
        # Anderson polish if Picard didn't converge
        if finf_p > 1e-6:
            res_a = rp.solve_anderson_pchip(
                G, taus_arr, gammas_arr, umax=UMAX,
                maxiters=300, abstol=1e-7, m_window=8,
                P_init=P)
            finf_a = float(np.abs(res_a["residual"]).max())
            if finf_a < finf_p:
                P, finf_p = res_a["P_star"], finf_a
        last_P = P
        # Compute E[|x_1|] at γ
        # For each cell, look up posterior μ_1 from the contour at p=P[i,j,l]
        E_abs_x = 0.0
        W = 0.0
        for i in range(G):
            for j in range(G):
                for l in range(G):
                    p = float(P[i, j, l])
                    mus = rh.posteriors_at(i, j, l, p, P, u, taus_arr)
                    x1 = _crra_demand(mus[0], p, gamma)
                    w_ = 0.5 * (f0u[i] * f0u[j] * f0u[l]
                                  + f1u[i] * f1u[j] * f1u[l])
                    E_abs_x += w_ * abs(x1)
                    W += w_
        E_abs_x /= W
        rows.append((gamma, E_abs_x, finf_p))
        print(f"  γ={gamma:5.2f}  E|x_1|={E_abs_x:.4f}  "
              f"Finf={finf_p:.2e}  ({time.time()-t0:.1f}s)",
              flush=True)

    rows.sort(key=lambda r: r[0])
    csv_path = os.path.join(OUT, "fig7_volume_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "E_abs_x1", "Finf"])
        for g, e, finf in rows:
            w.writerow([f"{g:.4f}", f"{e:.6f}", f"{finf:.3e}"])
    print(f"wrote {csv_path}")

    coords = " ".join(f"({g:.4f},{e:.6f})" for g, e, _ in rows)
    tex = _TEX % {"coords": coords}
    tex_path = os.path.join(OUT, "fig7_volume.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
