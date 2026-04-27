"""Figure 5 — Convergence Paths (Picard vs Anderson).

Runs the production PCHIP+contour kernel on γ=0.5, τ=2, G=20 with two
solvers, recording the per-iter ||P − Φ(P)||∞.

Output:
  figures/fig5_convergence.{tex,pdf,png}
"""
from __future__ import annotations
import os
import numpy as np
import rezn_pchip as rp


# ---- spec ---------------------------------------------------------------
TAU      = 2.0
GAMMA    = 0.5
G        = 20
UMAX     = 2.0
N_ITERS  = 80
PICARD_A = 0.3
ANDERSON_M = 8
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


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
    xmode=normal,
    ymode=log,
    xmin=0, xmax=%(xmax)d,
    ymin=1e-9, ymax=1,
    xlabel={iteration},
    ylabel={$\|P - \Phi(P)\|_\infty$  (best-so-far)},
    title={Convergence  ($\gamma=%(gamma)g$, $\tau=%(tau)g$, $G=%(g)d$)},
    legend pos=north east,
    legend style={fill=none, draw=none, font=\footnotesize},
    smooth,
]
%(addplots)s
\end{axis}
\end{tikzpicture}
\end{document}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    taus_arr   = np.array([TAU, TAU, TAU])
    gammas_arr = np.array([GAMMA, GAMMA, GAMMA])

    print(f"Picard α={PICARD_A} ...", flush=True)
    res_p = rp.solve_picard_pchip(
        G, taus_arr, gammas_arr, umax=UMAX,
        maxiters=N_ITERS, abstol=0.0, alpha=PICARD_A)
    h_p = np.minimum.accumulate(np.asarray(res_p["history"]))
    print(f"  best Finf={h_p[-1]:.3e}", flush=True)

    print(f"Anderson m={ANDERSON_M} ...", flush=True)
    res_a = rp.solve_anderson_pchip(
        G, taus_arr, gammas_arr, umax=UMAX,
        maxiters=N_ITERS, abstol=0.0, m_window=ANDERSON_M)
    h_a = np.minimum.accumulate(np.asarray(res_a["history"]))
    print(f"  best Finf={h_a[-1]:.3e}", flush=True)

    # tex
    coords_p = " ".join(f"({k},{max(v,1e-15):.6g})"
                         for k, v in enumerate(h_p))
    coords_a = " ".join(f"({k},{max(v,1e-15):.6g})"
                         for k, v in enumerate(h_a))
    addplots = (
        f"\\addplot[very thick, color=green, smooth] "
        f"coordinates {{{coords_p}}};\n"
        f"\\addlegendentry{{Picard $\\alpha\\!=\\!{PICARD_A:g}$}}\n"
        f"\\addplot[very thick, color=red, dashed, smooth] "
        f"coordinates {{{coords_a}}};\n"
        f"\\addlegendentry{{Anderson $m\\!=\\!{ANDERSON_M:d}$}}"
    )
    tex = _TEX % {"xmax": N_ITERS, "gamma": GAMMA, "tau": TAU,
                   "g": G, "addplots": addplots}
    tex_path = os.path.join(OUT, "fig5_convergence.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
