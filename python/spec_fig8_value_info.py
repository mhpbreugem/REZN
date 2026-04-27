"""Figure 8 — Value of Information V(τ).

V(τ) = E[U(informed)] − E[U(uninformed)]   (per agent)

Setup. Agent 1 has CRRA utility U(W) = W^(1-γ)/(1-γ) for γ≠1, log W
for γ=1. Initial wealth W₀ = 1.

Informed agent: at each grid cell (i,j,l):
  • posterior μ_1 = Λ(τu_1)  (own signal — no contour learning here,
    so this is the upper bound on V(τ) under partial revelation)
  • equilibrium price p = no-learning P[i,j,l]
  • demand x_1 from CRRA FOC, the standard
        x = (R-1)/((1-p)+Rp), R = exp((logit μ - logit p)/γ).
  • realised wealth W = W₀ + x_1·(v − p) for actual state v.
  • realised utility U(W).

Expected utility integrates over (i,j,l) with the joint prior
weight ½·(f₀³ + f₁³) and over v with the conditional state weights:
  E[U_informed] = Σ_{i,j,l} Σ_v 0.5·f_v(u_i)f_v(u_j)f_v(u_l)·U(W₀+x_1(v−p))

Uninformed agent: μ = 1/2 always, no own signal. By symmetry her
zero-information demand averaged over the prior gives a zero net
position with utility U(W₀). So
  V(τ) = E[U_informed] − U(W₀).
This understates V slightly (uninformed could still trade on the
observed price), but provides a clean lower bound that's enough for
the figure.

Output:
  figures/fig8_value_info.{tex,pdf,png,csv}
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# ---- spec ---------------------------------------------------------------
G       = 15
UMAX    = 4.0
GAMMAS  = [(0.25, "0.25", "green",  "solid"),
            (1.0, "1.0", "red",    "dashed"),
            (4.0, "4", "blue",   "dotted")]
TAUS    = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
P_LO    = 0.01           # tight filter — drops cells where bisection
P_HI    = 1 - P_LO       # is at the [0.002, 0.998] clip and demands blow up
W0      = 1.0
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
def _f_v(u, tau, v):
    return np.sqrt(tau / (2 * np.pi)) \
            * np.exp(-tau * 0.5 * (u - (v - 0.5)) ** 2)


@njit(cache=True, fastmath=True)
def _U(W, gamma):
    """CRRA utility, well-defined for W > 0 always.
        γ < 1:  U(W) = W^{1-γ}/(1-γ),     finite at W=0 (= 0).
        γ = 1:  U(W) = log W,              -∞ at W=0.
        γ > 1:  U(W) = W^{1-γ}/(1-γ) < 0,  -∞ at W=0 (1/W^{γ-1} blows up).
    Numerics: clamp W ≥ 1e-15 for stability."""
    Wc = W if W > 1e-15 else 1e-15
    if abs(gamma - 1.0) < 1e-12:
        return np.log(Wc)
    return (Wc ** (1.0 - gamma)) / (1.0 - gamma)


@njit(cache=True, parallel=True)
def _value_info(u, tau, gamma):
    """V(τ; γ) = E_{(u_1,u_2,u_3)}[ μ_1 U(W_v=1) + (1-μ_1) U(W_v=0) ] − U(W_0)

    Outer expectation is under the marginal prior over (u_1, u_2, u_3),
    which is the equal-weight mixture ½(f_0³ + f_1³).  Inner weighting
    over v uses the agent-1 posterior μ_1 = Λ(τ u_1) (no-learning) —
    that's the agent's own subjective probability of v=1 given her
    signal alone.  Together this is the agent's ex-ante expected
    utility from trading on her own signal, which equals
        log W_0 + E[KL(μ_1 || p)]    (for log utility, derivable),
    a non-negative quantity by Jensen.
    """
    n = u.shape[0]
    f0u = np.empty(n); f1u = np.empty(n); muu = np.empty(n)
    for k in range(n):
        f0u[k] = _f_v(u[k], tau, 0)
        f1u[k] = _f_v(u[k], tau, 1)
        muu[k] = 1.0 / (1.0 + np.exp(-tau * u[k]))
    SU  = np.zeros(n)
    SW  = np.zeros(n)
    for i in prange(n):
        sU = 0.0; sW = 0.0
        for j in range(n):
            for l in range(n):
                p = _clear_price(muu[i], muu[j], muu[l], gamma)
                if p <= P_LO or p >= P_HI:
                    continue
                x1 = _crra_demand(muu[i], p, gamma)
                W_v0 = W0 + x1 * (0.0 - p)
                W_v1 = W0 + x1 * (1.0 - p)
                # Inner v-expectation uses agent-1 posterior μ_1.
                # Outer (u) weight is the marginal prior at this cell,
                # which is ½(f_0(u_1)f_0(u_2)f_0(u_3) + f_1...f_1...).
                mu_1 = muu[i]
                w_cell = 0.5 * (
                    f0u[i] * f0u[j] * f0u[l]
                    + f1u[i] * f1u[j] * f1u[l])
                EU_cell = mu_1 * _U(W_v1, gamma) \
                            + (1.0 - mu_1) * _U(W_v0, gamma)
                sU += w_cell * EU_cell
                sW += w_cell
        SU[i] = sU; SW[i] = sW
    if SW.sum() <= 0:
        return 0.0
    return SU.sum() / SW.sum() - _U(W0, gamma)


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
    xmin=0.08, xmax=8,
    xtick={0.1,0.2,0.5,1,2,5},
    xticklabels={0.1,0.2,0.5,1,2,5},
    xlabel={signal precision $\tau$},
    ylabel={$V(\tau)$},
    title={Value of information},
    legend pos=north west,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
%(addplots)s
\addplot[ultra thick, color=black, dashdotted]
    coordinates {(0.08,0)(8,0)};
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
    _ = _value_info(u, 1.0, 1.0)         # JIT warm-up

    results = {}
    for gamma, label, *_ in GAMMAS:
        vals = []
        for tau in TAUS:
            v = _value_info(u, float(tau), float(gamma))
            vals.append(v)
            print(f"  γ={label}  τ={tau:5.2f}  V={v:.5f}", flush=True)
        results[gamma] = vals

    csv_path = os.path.join(OUT, "fig8_value_info_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau"] + [f"V_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w.writerow([f"{tau:.4f}"]
                        + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"wrote {csv_path}")

    style_map = {
        "solid":  "very thick, color=green, smooth",
        "dashed": "very thick, color=red, dashed, smooth",
        "dotted": "very thick, color=blue, dotted, smooth",
    }
    addplots = []
    for gamma, label, color, style in GAMMAS:
        opts = style_map[style]
        coords = " ".join(f"({tau:.4f},{v:.6g})"
                           for tau, v in zip(TAUS, results[gamma]))
        addplots.append(f"\\addplot[{opts}] coordinates {{{coords}}};")
    tex = _TEX % {"addplots": "\n".join(addplots)}
    tex_path = os.path.join(OUT, "fig8_value_info.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
