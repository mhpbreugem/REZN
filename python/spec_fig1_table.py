"""Figure 1 — Smooth Transition Table.

LaTeX table of 1−R² over (γ, τ), no-learning equilibrium only.

Same algorithm as Figure 2 (per-cell brentq market clearing on the
G=20 grid u ∈ [-4, 4], weighted regression with cell filter
1e-4 < P < 1-1e-4). Reports values to 3 decimal places.

γ rows: 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0
τ cols: 0.5, 1.0, 2.0, 3.0
The γ=100 row should be all zeros — CARA baseline check.

Output:
  figures/fig1_table.tex   — standalone LaTeX table (booktabs)
  figures/fig1_table.csv   — same numbers in CSV
"""
from __future__ import annotations
import os
import csv
import numpy as np
from numba import njit, prange


# ---- spec ---------------------------------------------------------------
G       = 20
UMAX    = 4.0
P_LO    = 1e-4
P_HI    = 1 - P_LO
GAMMAS  = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]
TAUS    = [0.5, 1.0, 2.0, 3.0]
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


# ---- numerics (same as Fig 2) -------------------------------------------
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


@njit(cache=True, fastmath=True)
def _f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u + 0.5)**2)


@njit(cache=True, fastmath=True)
def _f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u - 0.5)**2)


@njit(cache=True, parallel=True)
def _weighted_1mR2(u, tau, gamma):
    n = u.shape[0]
    f0u = np.empty(n); f1u = np.empty(n); muu = np.empty(n)
    for k in range(n):
        f0u[k] = _f0(u[k], tau)
        f1u[k] = _f1(u[k], tau)
        muu[k] = 1.0 / (1.0 + np.exp(-tau * u[k]))
    Sw   = np.zeros(n); Sy   = np.zeros(n); St   = np.zeros(n)
    Syy  = np.zeros(n); Stt  = np.zeros(n); Syt  = np.zeros(n)
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
                sw  += w_; sy  += w_ * y_; st  += w_ * t_
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


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)
    _ = _weighted_1mR2(u, 1.0, 1.0)        # JIT warm-up

    table = np.zeros((len(GAMMAS), len(TAUS)))
    for r, g in enumerate(GAMMAS):
        for c, t in enumerate(TAUS):
            table[r, c] = _weighted_1mR2(u, float(t), float(g))
        print(f"  γ={g:6.2f}: " +
               "  ".join(f"τ={t:.1f}: {table[r,c]:.4f}"
                          for c, t in enumerate(TAUS)),
               flush=True)

    # CSV
    csv_path = os.path.join(OUT, "fig1_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma\\tau"] + [str(t) for t in TAUS])
        for r, g in enumerate(GAMMAS):
            w.writerow([str(g)] + [f"{table[r,c]:.6f}" for c in range(len(TAUS))])
    print(f"wrote {csv_path}")

    # LaTeX standalone table
    rows_tex = []
    for r, g in enumerate(GAMMAS):
        cells = "  &  ".join(f"{table[r, c]:.3f}" for c in range(len(TAUS)))
        rows_tex.append(f"  ${g:g}$  &  {cells} \\\\")
    body = "\n".join(rows_tex)
    cols_tex = " & ".join(f"$\\tau={t:g}$" for t in TAUS)

    tex = (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\begin{document}\n"
        "\\begin{tabular}{c" + "c" * len(TAUS) + "}\n"
        "\\toprule\n"
        "$\\gamma$  &  " + cols_tex + " \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{document}\n"
    )
    tex_path = os.path.join(OUT, "fig1_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
