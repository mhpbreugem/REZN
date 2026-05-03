#!/usr/bin/env python3
"""Fig 6B from G=18 umax=4 mp300 checkpoint, redone in mpmath dps=100, tol 1e-50.

Sweep T* in [-3, +4] (50 pts). At each T*:
  u1 = closest grid point to +1, u2 = closest to -1, u3 = T*/tau - u1 - u2.
  Solve REE market clearing -> p; then mu1 = mu*(u1, p), mu2 = mu*(u2, p).

Output: results/full_ree/fig6B_G18_pgfplots.tex
"""

from pathlib import Path

import mpmath as mp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _mp_helpers import (
    bisect_market_clear,
    crra_demand_binary,
    fmt_mp,
    load_posterior_mp,
    make_mu_interpolator,
)

mp.mp.dps = 100

SEED = Path("results/full_ree/posterior_v3_G18_mp300_notrim.json")
OUT = Path("results/full_ree/fig6B_G18_pgfplots.tex")
N_POINTS = 50


def main():
    d, u_grid, p_grids, mu_grids = load_posterior_mp(SEED)
    G = d["G"]
    tau = mp.mpf(d["tau"])
    gamma = mp.mpf(d["gamma"])
    umax = u_grid[-1]
    print(f"Loaded G={G} umax={fmt_mp(umax,6)}, tau={tau}, gamma={gamma}")
    print(f"  ||F||_inf = {fmt_mp(mp.mpf(d['F_max']),24)}")

    interp_mu = make_mu_interpolator(u_grid, p_grids, mu_grids)

    one = mp.mpf(1)
    i1 = min(range(G), key=lambda i: abs(u_grid[i] - one))
    i2 = min(range(G), key=lambda i: abs(u_grid[i] + one))
    u1 = u_grid[i1]
    u2 = u_grid[i2]
    print(f"u1 = {fmt_mp(u1,24)} (idx {i1})")
    print(f"u2 = {fmt_mp(u2,24)} (idx {i2})")

    Tlo, Thi = mp.mpf(-3), mp.mpf(4)
    step = (Thi - Tlo) / (N_POINTS - 1)
    rows = []

    for k in range(N_POINTS):
        Tstar = Tlo + step * k
        u3 = Tstar / tau - u1 - u2
        u_vec = (u1, u2, u3)

        def excess_REE(p):
            return sum(
                crra_demand_binary(interp_mu(u_vec[j], p), p, gamma)
                for j in range(3)
            )

        p = bisect_market_clear(excess_REE)
        mu1 = interp_mu(u1, p)
        mu2 = interp_mu(u2, p)
        rows.append((Tstar, u3, p, mu1, mu2))
        print(f"  T*={fmt_mp(Tstar,8)}  u3={fmt_mp(u3,8)}  "
              f"p={fmt_mp(p,12)}  mu1={fmt_mp(mu1,12)}  mu2={fmt_mp(mu2,12)}")

    mu1_coords = "".join(f"({fmt_mp(T,10)},{fmt_mp(m1,24)})" for (T, _, _, m1, _) in rows)
    mu2_coords = "".join(f"({fmt_mp(T,10)},{fmt_mp(m2,24)})" for (T, _, _, _, m2) in rows)
    p_coords = "".join(f"({fmt_mp(T,10)},{fmt_mp(p,24)})" for (T, _, p, _, _) in rows)

    lines = [
        f"% Fig 6B from G={G} umax={fmt_mp(umax,4)} mp300 seed  [mpmath dps=100, tol 1e-50]",
        f"% u1 = {fmt_mp(u1,24)}",
        f"% u2 = {fmt_mp(u2,24)}",
        f"% T* in [{fmt_mp(Tlo,4)}, {fmt_mp(Thi,4)}], {N_POINTS} pts; u3 = T*/tau - u1 - u2",
        f"% G={G} tau={fmt_mp(tau,8)} gamma={fmt_mp(gamma,8)} dps={d['dps']}",
        f"% Seed F_max = {fmt_mp(mp.mpf(d['F_max']),24)}",
        "",
        f"% mu1 (agent 1, u={fmt_mp(u1,12)})",
        f"\\addplot coordinates {{{mu1_coords}}};",
        "",
        f"% mu2 (agent 2, u={fmt_mp(u2,12)})",
        f"\\addplot coordinates {{{mu2_coords}}};",
        "",
        f"% price (REE)",
        f"\\addplot coordinates {{{p_coords}}};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
