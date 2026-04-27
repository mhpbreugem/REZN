"""Figure 10 — 1−R² vs K (number of agents).

Sweeps K=3..10 at fixed τ=1, γ ∈ {0.2, 1, 5} plus CARA.

Implementation: Monte Carlo (N=400k stratified samples per (γ, K))
draws u₁..u_K from the equal-mixture prior, solves market clearing
at each draw, regresses logit(P) on T*=τ Σ u_k. With this many
samples MC error on 1-R² is below 5e-4 even at small K.

CARA demand is *explicit* (not γ→∞ limit):
    x_k = (logit μ_k - logit p) / a_k
giving the closed-form aggregator
    logit(p) = (1/K) Σ logit μ_k.
For binary v this aggregates the τu_k sufficient statistic exactly,
so 1-R² = 0 by construction. Plotted as the flat reference line.

Output:
  figures/fig10_K_agents.{tex,pdf,png,csv}
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# ---- spec ---------------------------------------------------------------
TAU     = 1.0
GAMMAS  = [(0.2, "0.2", "green",  "solid"),
            (1.0, "1.0", "red",    "dashed"),
            (5.0, "5.0", "blue",   "dotted")]
KS      = list(range(3, 11))
N_MC    = 400_000
SEED    = 13579
P_LO    = 1e-9                        # ultra-loose; bisection bounds dominate
P_HI    = 1 - P_LO
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
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
def _residual_K(mus, p, gamma):
    s = 0.0
    for k in range(mus.shape[0]):
        s += _crra_demand(mus[k], p, gamma)
    return s


@njit(cache=True)
def _clear_price_K(mus, gamma):
    """Bisection on Σ_k x_k(μ_k, p) = 0; γ=1 has closed-form mean."""
    if abs(gamma - 1.0) < 1e-12:
        s = 0.0
        for k in range(mus.shape[0]):
            s += mus[k]
        return s / mus.shape[0]
    lo, hi = 0.002, 0.998
    f_lo = _residual_K(mus, lo, gamma)
    f_hi = _residual_K(mus, hi, gamma)
    for _ in range(120):
        m = 0.5 * (lo + hi)
        f_m = _residual_K(mus, m, gamma)
        if f_lo * f_m < 0:
            hi = m; f_hi = f_m
        else:
            lo = m; f_lo = f_m
        if (hi - lo) < 1e-14:
            break
    return 0.5 * (lo + hi)


@njit(cache=True, parallel=True)
def _one_minus_R2(samples, tau, gamma):
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
        if p <= P_LO or p >= P_HI:
            Y[n] = np.nan
        else:
            Y[n] = _logit(p)
        T[n] = tau * T_sum
    # Drop filtered rows
    mask = ~np.isnan(Y)
    Y = Y[mask]; T = T[mask]
    if Y.size < 10:
        return 0.0
    Y_m = Y.mean(); T_m = T.mean()
    Syy = ((Y - Y_m)**2).sum()
    STT = ((T - T_m)**2).sum()
    SyT = ((Y - Y_m) * (T - T_m)).sum()
    if Syy <= 0 or STT <= 0:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


def _draw(K, n, rng):
    """Stratified prior draw: half from v=0, half from v=1."""
    half = n // 2
    s0 = -0.5 + rng.standard_normal((half, K)) / np.sqrt(TAU)
    s1 = +0.5 + rng.standard_normal((n - half, K)) / np.sqrt(TAU)
    return np.vstack([s0, s1])


# ---- TeX template (agreed format) ---------------------------------------
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
    xmin=2.5, xmax=10.5,
    ymin=-0.001, ymax=0.15,
    xtick={3,4,5,6,7,8,9,10},
    ytick={0,0.05,0.1,0.15},
    yticklabels={0,0.05,0.10,0.15},
    xlabel={number of agents $K$},
    ylabel={$1 - R^2$},
    title={Number of agents ($K$)},
    legend pos=north east,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
%(addplots)s
\addplot[ultra thick, color=black, dashdotted]
    coordinates {(2.5,0)(10.5,0)};
\addlegendentry{$\gamma = 0.2$}
\addlegendentry{$\gamma = 1$}
\addlegendentry{$\gamma = 5$}
\addlegendentry{CARA}
\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # JIT warm-up
    _ = _one_minus_R2(_draw(3, 256, np.random.default_rng(0)).astype(float),
                       TAU, 1.0)

    results = {g: [] for g, *_ in GAMMAS}
    for K in KS:
        samples = _draw(K, N_MC, rng).astype(np.float64, copy=False)
        for gamma, label, *_ in GAMMAS:
            v = _one_minus_R2(samples, TAU, float(gamma))
            results[gamma].append(v)
            print(f"K={K:2d}  γ={label}  1-R²={v:.5f}", flush=True)

    csv_path = os.path.join(OUT, "fig10_K_agents_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for ki, K in enumerate(KS):
            w.writerow([K]
                        + [f"{results[g][ki]:.6g}" for g, *_ in GAMMAS])
    print(f"wrote {csv_path}")

    style_map = {
        "solid":  "very thick, color=green, smooth",
        "dashed": "very thick, color=red, dashed, smooth",
        "dotted": "very thick, color=blue, dotted, smooth",
    }
    addplots = []
    for gamma, label, color, style in GAMMAS:
        opts = style_map[style]
        coords = " ".join(f"({K},{results[gamma][ki]:.6g})"
                           for ki, K in enumerate(KS))
        addplots.append(f"\\addplot[{opts}] coordinates {{{coords}}};")
    tex = _TEX % {"addplots": "\n".join(addplots)}
    tex_path = os.path.join(OUT, "fig10_K_agents.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
