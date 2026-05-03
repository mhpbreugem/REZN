#!/usr/bin/env python3
"""Fig 5 REDO: asymmetric triples (u1=+1, u2=-1, vary u3).

For each u3 value:
  - p_FR  = Lambda(T*/3) where T* = tau*(u1+u2+u3)
  - p_NL  = market clearing with private priors mu_k = Lambda(tau*u_k)
  - p_REE = market clearing using converged posterior mu*(u_k, p)

Source seed: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json
Output: results/full_ree/fig5_G20_asymmetric_pgfplots.tex
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

SEED = Path("results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json")
OUT = Path("results/full_ree/fig5_G20_asymmetric_pgfplots.tex")


def logit(p):
    return np.log(p) - np.log1p(-p)


def crra_demand(mu, p, gamma):
    if mu < 1e-15:
        mu = 1e-15
    if mu > 1 - 1e-15:
        mu = 1 - 1e-15
    if p < 1e-15 or p > 1 - 1e-15:
        return 0.0
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0:
        e = np.exp(-z)
        return (1.0 - e) / ((1.0 - p) * e + p)
    e = np.exp(z)
    return (e - 1.0) / ((1.0 - p) + p * e)


def load_posterior(path):
    with open(path) as f:
        d = json.load(f)
    u_grid = np.array([float(x) for x in d["u_grid"]])
    p_grids = [np.array([float(x) for x in row]) for row in d["p_grid"]]
    mu_grids = [np.array([float(x) for x in row]) for row in d["mu_strings"]]
    return d, u_grid, p_grids, mu_grids


def make_mu_interpolator(u_grid, p_grids, mu_grids):
    """Bilinear interpolation of mu*(u, p) on the (varying-p) checkpoint grid."""
    G = len(u_grid)

    def mu_at_u_idx(idx, p):
        p_arr = p_grids[idx]
        mu_arr = mu_grids[idx]
        if p <= p_arr[0]:
            return float(mu_arr[0])
        if p >= p_arr[-1]:
            return float(mu_arr[-1])
        j = int(np.searchsorted(p_arr, p) - 1)
        j = max(0, min(j, len(p_arr) - 2))
        denom = p_arr[j + 1] - p_arr[j]
        frac = (p - p_arr[j]) / denom if denom > 0 else 0.0
        return float(mu_arr[j] + frac * (mu_arr[j + 1] - mu_arr[j]))

    def interp(u, p):
        if u <= u_grid[0]:
            i_lo, i_hi = 0, 1
        elif u >= u_grid[-1]:
            i_lo, i_hi = G - 2, G - 1
        else:
            i_lo = int(np.searchsorted(u_grid, u) - 1)
            i_hi = i_lo + 1
        m_lo = mu_at_u_idx(i_lo, p)
        m_hi = mu_at_u_idx(i_hi, p)
        denom = u_grid[i_hi] - u_grid[i_lo]
        f = (u - u_grid[i_lo]) / denom if denom > 0 else 0.0
        f = max(0.0, min(1.0, f))
        return m_lo + f * (m_hi - m_lo)

    return interp


def clear_market(mu_fn, gamma, eps=1e-12):
    a, b = eps, 1.0 - eps
    fa = sum(crra_demand(mu_fn(p, k), p, gamma) for k in (0, 1, 2) for p in [a]) / 3.0  # dummy
    # Direct excess function
    def excess(p):
        return sum(crra_demand(mu_fn(p, k), p, gamma) for k in range(3))
    fa, fb = excess(a), excess(b)
    if fa <= 0:
        return a
    if fb >= 0:
        return b
    return brentq(excess, a, b, xtol=1e-12, rtol=1e-12)


def main():
    d, u_grid, p_grids, mu_grids = load_posterior(SEED)
    G = d["G"]
    tau = float(d["tau"])
    gamma = float(d["gamma"])
    print(f"Loaded seed: G={G}, UMAX={d['UMAX']}, tau={tau}, gamma={gamma}, "
          f"||F||={d['F_max'][:12]}...")

    interp_mu = make_mu_interpolator(u_grid, p_grids, mu_grids)

    # Fix u1 ≈ +1, u2 ≈ -1 at nearest grid points
    i1 = int(np.argmin(np.abs(u_grid - 1.0)))
    i2 = int(np.argmin(np.abs(u_grid + 1.0)))
    u1 = float(u_grid[i1])
    u2 = float(u_grid[i2])
    print(f"u1 = {u1:.4f} (grid idx {i1}), u2 = {u2:.4f} (grid idx {i2})")

    # Vary u3 from -3 to +3, 50 points
    u3_values = np.linspace(-3.0, 3.0, 50)

    rows = []
    for u3 in u3_values:
        Tstar = tau * (u1 + u2 + u3)
        p_FR = float(expit(Tstar / 3.0))

        # No-learning: private prior is Lambda(tau * u_k)
        mu_NL = [float(expit(tau * u)) for u in (u1, u2, u3)]

        def excess_NL(p):
            return sum(crra_demand(mu_NL[k], p, gamma) for k in range(3))

        eps = 1e-12
        if excess_NL(eps) <= 0:
            p_NL = eps
        elif excess_NL(1 - eps) >= 0:
            p_NL = 1 - eps
        else:
            p_NL = float(brentq(excess_NL, eps, 1 - eps, xtol=1e-12, rtol=1e-12))

        # REE: use converged posterior mu*(u, p)
        u_vec = (u1, u2, float(u3))

        def excess_REE(p):
            return sum(crra_demand(interp_mu(u_vec[k], p), p, gamma) for k in range(3))

        if excess_REE(eps) <= 0:
            p_REE = eps
        elif excess_REE(1 - eps) >= 0:
            p_REE = 1 - eps
        else:
            p_REE = float(brentq(excess_REE, eps, 1 - eps, xtol=1e-12, rtol=1e-12))

        rows.append((float(u3), float(Tstar), p_FR, p_NL, p_REE))

    # Write pgfplots
    def fmt_curve(idx):
        return "".join(f"({Tstar:.4f},{r[idx]:.6f})" for r in [(0, T, fr, nl, re) for (_, T, fr, nl, re) in rows] for Tstar in [T])

    fr_coords = "".join(f"({T:.4f},{fr:.6f})" for (_, T, fr, _, _) in rows)
    nl_coords = "".join(f"({T:.4f},{nl:.6f})" for (_, T, _, nl, _) in rows)
    re_coords = "".join(f"({T:.4f},{re:.6f})" for (_, T, _, _, re) in rows)

    Tmin, Tmax = rows[0][1], rows[-1][1]
    lines = [
        f"% Fig 5 (asymmetric triples): price vs T*",
        f"% u1 = {u1:.4f}, u2 = {u2:.4f}, u3 in [{u3_values[0]:.2f}, {u3_values[-1]:.2f}], 50 pts",
        f"% T* = tau*(u1+u2+u3) in [{Tmin:.4f}, {Tmax:.4f}]",
        f"% G={G} UMAX={d['UMAX']} gamma={gamma} tau={tau}",
        f"% Seed F_max = {d['F_max'][:18]}... (~{float(d['F_max']):.2e})",
        "",
        f"% FR (analytical, p = Lambda(T*/3))",
        f"\\addplot coordinates {{{fr_coords}}};",
        "",
        f"% NL (no learning, private priors mu_k = Lambda(tau u_k))",
        f"\\addplot coordinates {{{nl_coords}}};",
        "",
        f"% REE (converged posterior mu*(u, p))",
        f"\\addplot coordinates {{{re_coords}}};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print(f"  T* range: [{Tmin:.3f}, {Tmax:.3f}]")
    print(f"  p_FR  range: [{min(r[2] for r in rows):.3f}, {max(r[2] for r in rows):.3f}]")
    print(f"  p_NL  range: [{min(r[3] for r in rows):.3f}, {max(r[3] for r in rows):.3f}]")
    print(f"  p_REE range: [{min(r[4] for r in rows):.3f}, {max(r[4] for r in rows):.3f}]")


if __name__ == "__main__":
    main()
