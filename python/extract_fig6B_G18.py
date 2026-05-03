#!/usr/bin/env python3
"""Fig 6B REDO from the G=18 umax=4 mp300 checkpoint.

Sweeps T* in [-3, +4] (50 points). At each T*:
  - u1 = closest grid point to +1
  - u2 = closest grid point to -1
  - u3 = T*/tau - u1 - u2
  - solve REE market clearing -> p
  - mu1 = mu*(u1, p), mu2 = mu*(u2, p)

Output: results/full_ree/fig6B_G18_pgfplots.tex
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

SEED = Path("results/full_ree/posterior_v3_G18_mp300_notrim.json")
OUT = Path("results/full_ree/fig6B_G18_pgfplots.tex")


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


def main():
    d, u_grid, p_grids, mu_grids = load_posterior(SEED)
    G = d["G"]
    tau = float(d["tau"])
    gamma = float(d["gamma"])
    print(f"Loaded G={G} (umax={u_grid[-1]:.2f}), tau={tau}, gamma={gamma}")
    print(f"  ||F||_inf = {d['F_max'][:18]}... ({float(d['F_max']):.2e})")

    interp_mu = make_mu_interpolator(u_grid, p_grids, mu_grids)

    i1 = int(np.argmin(np.abs(u_grid - 1.0)))
    i2 = int(np.argmin(np.abs(u_grid + 1.0)))
    u1 = float(u_grid[i1])
    u2 = float(u_grid[i2])
    print(f"u1 = {u1:.4f} (idx {i1}), u2 = {u2:.4f} (idx {i2})")

    Tstar_values = np.linspace(-3.0, 4.0, 50)

    rows = []
    eps = 1e-12
    for Tstar in Tstar_values:
        u3 = Tstar / tau - u1 - u2

        u_vec = (u1, u2, float(u3))

        def excess_REE(p):
            return sum(crra_demand(interp_mu(u_vec[k], p), p, gamma) for k in range(3))

        if excess_REE(eps) <= 0:
            p = eps
        elif excess_REE(1 - eps) >= 0:
            p = 1 - eps
        else:
            p = float(brentq(excess_REE, eps, 1 - eps, xtol=1e-12, rtol=1e-12))

        mu1 = interp_mu(u1, p)
        mu2 = interp_mu(u2, p)
        rows.append((float(Tstar), float(u3), p, mu1, mu2))

    mu1_coords = "".join(f"({T:.4f},{m1:.6f})" for (T, _, _, m1, _) in rows)
    mu2_coords = "".join(f"({T:.4f},{m2:.6f})" for (T, _, _, _, m2) in rows)
    p_coords = "".join(f"({T:.4f},{p:.6f})" for (T, _, p, _, _) in rows)

    lines = [
        f"% Fig 6B (CRRA posteriors vs T*) from G={G} umax={u_grid[-1]:.0f} mp300",
        f"% u1 = {u1:.4f}, u2 = {u2:.4f}, T* in [{Tstar_values[0]:.2f}, {Tstar_values[-1]:.2f}]",
        f"% u3 = T*/tau - u1 - u2 (50 points)",
        f"% G={G} tau={tau} gamma={gamma} dps={d['dps']}",
        f"% Seed F_max = {d['F_max'][:18]}... (~{float(d['F_max']):.2e})",
        "",
        f"% mu1 (agent 1, u={u1:.4f})",
        f"\\addplot coordinates {{{mu1_coords}}};",
        "",
        f"% mu2 (agent 2, u={u2:.4f})",
        f"\\addplot coordinates {{{mu2_coords}}};",
        "",
        f"% price (REE)",
        f"\\addplot coordinates {{{p_coords}}};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print(f"  mu1 range: [{min(r[3] for r in rows):.3f}, {max(r[3] for r in rows):.3f}]")
    print(f"  mu2 range: [{min(r[4] for r in rows):.3f}, {max(r[4] for r in rows):.3f}]")
    print(f"  p   range: [{min(r[2] for r in rows):.3f}, {max(r[2] for r in rows):.3f}]")


if __name__ == "__main__":
    main()
