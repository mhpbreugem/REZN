"""Figure 6 — Three Mechanisms Table (heterogeneous γ_k, τ_k).

LaTeX table of 1−R² for six configurations isolating different
sources of partial revelation. No-learning REE only (no contour
iteration). On the G=20 grid with weighted regression.

Configurations follow the spec, but with mild ranges (max τ=3 and
max γ=10) to keep the on-grid bisection in well-behaved territory:

  1. baseline (equal γ=1, equal τ=2): pure Jensen gap
  2. het γ=(1,3,10), equal τ=2:        + heterogeneous risk aversion
  3. equal γ=1, het τ=(1,3,10):        + heterogeneous precision
  4. het γ + het τ aligned  γ=(10,3,1), τ=(1,3,10): low-γ paired w/ high-τ
  5. het γ + het τ opposed  γ=(1,3,10), τ=(1,3,10): low-γ w/ low-τ
  6. extreme opposed        γ=(0.3,3,10), τ=(0.5,3,10)

Output:
  figures/fig6_mechanisms.tex   standalone booktabs table
  figures/fig6_mechanisms.csv   raw numbers
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
OUT     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")

CONFIGS = [
    ("baseline (equal $\\gamma=1$, equal $\\tau=2$)",
        np.array([2.0, 2.0, 2.0]),  np.array([1.0, 1.0, 1.0])),
    ("het $\\gamma=(1,3,10)$, equal $\\tau=2$",
        np.array([2.0, 2.0, 2.0]),  np.array([1.0, 3.0, 10.0])),
    ("equal $\\gamma=1$, het $\\tau=(1,3,10)$",
        np.array([1.0, 3.0, 10.0]), np.array([1.0, 1.0, 1.0])),
    ("het $\\gamma+\\tau$ aligned: $\\gamma=(10,3,1)$, $\\tau=(1,3,10)$",
        np.array([1.0, 3.0, 10.0]), np.array([10.0, 3.0, 1.0])),
    ("het $\\gamma+\\tau$ opposed: $\\gamma=(1,3,10)$, $\\tau=(1,3,10)$",
        np.array([1.0, 3.0, 10.0]), np.array([1.0, 3.0, 10.0])),
    ("extreme opposed: $\\gamma=(0.3,3,10)$, $\\tau=(0.5,3,10)$",
        np.array([0.5, 3.0, 10.0]), np.array([0.3, 3.0, 10.0])),
]


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
def _residual_het(mus, p, gammas):
    return _crra_demand(mus[0], p, gammas[0]) \
            + _crra_demand(mus[1], p, gammas[1]) \
            + _crra_demand(mus[2], p, gammas[2])


@njit(cache=True)
def _clear_price_het(mus, gammas):
    lo, hi = 0.002, 0.998
    f_lo = _residual_het(mus, lo, gammas)
    f_hi = _residual_het(mus, hi, gammas)
    for _ in range(120):
        m = 0.5 * (lo + hi)
        f_m = _residual_het(mus, m, gammas)
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


@njit(cache=True, parallel=True)
def _weighted_1mR2_het(u, taus, gammas):
    n = u.shape[0]
    f0 = np.empty((3, n)); f1 = np.empty((3, n))
    mu = np.empty((3, n))
    for k in range(3):
        for i in range(n):
            f0[k, i] = _f_v(u[i], taus[k], 0)
            f1[k, i] = _f_v(u[i], taus[k], 1)
            mu[k, i] = 1.0 / (1.0 + np.exp(-taus[k] * u[i]))
    Sw   = np.zeros(n); Sy   = np.zeros(n); St   = np.zeros(n)
    Syy  = np.zeros(n); Stt  = np.zeros(n); Syt  = np.zeros(n)
    for i in prange(n):
        sw = 0.0; sy = 0.0; st = 0.0
        syy = 0.0; stt = 0.0; syt = 0.0
        for j in range(n):
            for l in range(n):
                mus = np.array([mu[0, i], mu[1, j], mu[2, l]])
                p = _clear_price_het(mus, gammas)
                if p <= P_LO or p >= P_HI:
                    continue
                y_ = _logit(p)
                t_ = taus[0] * u[i] + taus[1] * u[j] + taus[2] * u[l]
                w_ = 0.5 * (f0[0, i] * f0[1, j] * f0[2, l]
                              + f1[0, i] * f1[1, j] * f1[2, l])
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

    # JIT warm-up
    _ = _weighted_1mR2_het(u, np.array([1.0, 1.0, 1.0]),
                              np.array([1.0, 1.0, 1.0]))

    rows = []
    for label, taus, gammas in CONFIGS:
        v = _weighted_1mR2_het(u, taus.astype(float), gammas.astype(float))
        rows.append((label, v))
        print(f"  {label}\n     1-R² = {v:.4f}", flush=True)

    csv_path = os.path.join(OUT, "fig6_mechanisms.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "1-R²"])
        for label, val in rows:
            label_plain = (label.replace("$", "")
                                  .replace("\\gamma", "γ")
                                  .replace("\\tau", "τ"))
            w.writerow([label_plain, f"{val:.6f}"])
    print(f"wrote {csv_path}")

    body = "\n".join(f"  {lbl}  &  {val:.4f} \\\\" for lbl, val in rows)
    tex = (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{amsmath,booktabs}\n"
        "\\begin{document}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "configuration  &  $1-R^2$ \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{document}\n"
    )
    tex_path = os.path.join(OUT, "fig6_mechanisms.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
