"""Figure 9 — Grossman-Stiglitz resolution: V(τ) − c.

Derived from Figure 8: at fixed τ=2, plot V(τ; γ) − c for each γ
on a horizontal cost axis c. CARA: V=0 always so the line is
y = −c, always negative — no agent ever acquires costly information.
CRRA: V > 0 so the line is positive for c < V(τ) — agents acquire
information when c is small enough, paradox resolved.

Output:
  figures/fig9_GS.{tex,pdf,png}
"""
from __future__ import annotations
import os
import csv
import numpy as np


# ---- spec ---------------------------------------------------------------
TAU      = 2.0
GAMMAS   = [(0.2, "0.2", "green",  "solid"),
            (1.0, "1.0", "red",    "dashed"),
            (5.0, "5.0", "blue",   "dotted")]
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")
DATA_CSV = os.path.join(OUT, "fig8_value_info_data.csv")


def _read_V_at_tau(csv_path, tau, gammas):
    """Pick V(τ) at the requested τ row from Fig 8's CSV."""
    out = {}
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        rows = list(r)
    # find nearest row to tau
    diffs = [(abs(float(row["tau"]) - tau), row) for row in rows]
    diffs.sort(key=lambda d: d[0])
    nearest = diffs[0][1]
    print(f"  using τ={float(nearest['tau']):.3f} from {csv_path}")
    for g, *_ in gammas:
        out[g] = float(nearest[f"V_g{g}"])
    return out


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
    xmin=0, xmax=%(cmax).4f,
    xlabel={signal cost $c$},
    ylabel={$V(\tau)-c$},
    title={Grossman--Stiglitz at $\tau\!=\!%(tau)g$},
    legend pos=north east,
    legend style={fill=none, draw=none, font=\footnotesize},
]
%(addplots)s
\addplot[thin, color=black, dotted]
    coordinates {(0,0)(%(cmax).4f,0)};
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
    if not os.path.exists(DATA_CSV):
        print(f"Fig 8 CSV missing — run spec_fig8_value_info.py first")
        return
    Vs = _read_V_at_tau(DATA_CSV, TAU, GAMMAS)
    print(f"V(τ={TAU}):  " +
           "  ".join(f"γ={g}: {Vs[g]:.4f}" for g, *_ in GAMMAS))

    cmax = max(max(Vs.values()), 0.01) * 1.4
    style_map = {
        "solid":  "very thick, color=green",
        "dashed": "very thick, color=red, dashed",
        "dotted": "very thick, color=blue, dotted",
    }
    addplots = []
    for gamma, label, color, style in GAMMAS:
        V = Vs[gamma]
        coords = f"(0,{V:.6f})({cmax:.4f},{V - cmax:.6f})"
        addplots.append(
            f"\\addplot[{style_map[style]}] coordinates {{{coords}}};")
    # CARA reference: V=0 → V−c = −c for all c.
    addplots.append(
        f"\\addplot[ultra thick, color=black, dashdotted] "
        f"coordinates {{(0,0)({cmax:.4f},{-cmax:.4f})}};")

    tex = _TEX % {"cmax": cmax, "tau": TAU,
                   "addplots": "\n".join(addplots)}
    tex_path = os.path.join(OUT, "fig9_GS.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
