#!/usr/bin/env python3
"""Fig 5 REDO: asymmetric triples in mpmath at dps=100, market-clearing tol 1e-50.

u1 = closest grid point to +1, u2 = closest to -1, vary u3 in [-3, +3] (50 pts).
T* = tau (u1 + u2 + u3).
  - p_FR  = Lambda(T* / 3) (analytical)
  - p_NL  = market clearing with mu_k = Lambda(tau u_k) (private priors)
  - p_REE = market clearing using mu*(u_k, p) interpolated from the seed.

Source: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json
Output: results/full_ree/fig5_G20_asymmetric_pgfplots.tex
"""

from pathlib import Path

import mpmath as mp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _mp_helpers import (
    bisect_market_clear,
    crra_demand_binary,
    fmt_mp,
    lam,
    load_posterior_mp,
    make_mu_interpolator,
    EPS_PRICE,
)

mp.mp.dps = 100

SEED = Path("results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json")
OUT = Path("results/full_ree/fig5_G20_asymmetric_pgfplots.tex")
N_POINTS = 50


def main():
    d, u_grid, p_grids, mu_grids = load_posterior_mp(SEED)
    G = d["G"]
    tau = mp.mpf(d["tau"])
    gamma = mp.mpf(d["gamma"])
    print(f"Loaded seed: G={G}, UMAX={d['UMAX']}, tau={tau}, gamma={gamma}")
    print(f"  ||F||_inf = {fmt_mp(mp.mpf(d['F_max']), 24)}")

    interp_mu = make_mu_interpolator(u_grid, p_grids, mu_grids)

    # Closest grid points to +1 and -1
    one = mp.mpf(1)
    i1 = min(range(G), key=lambda i: abs(u_grid[i] - one))
    i2 = min(range(G), key=lambda i: abs(u_grid[i] + one))
    u1 = u_grid[i1]
    u2 = u_grid[i2]
    print(f"u1 = {fmt_mp(u1,24)} (idx {i1})")
    print(f"u2 = {fmt_mp(u2,24)} (idx {i2})")

    # 50 points u3 in [-3, +3]
    u3_lo, u3_hi = mp.mpf(-3), mp.mpf(3)
    step = (u3_hi - u3_lo) / (N_POINTS - 1)
    rows = []

    for k in range(N_POINTS):
        u3 = u3_lo + step * k
        Tstar = tau * (u1 + u2 + u3)

        # Analytic p_FR = Lambda(T*/3)
        p_FR = lam(Tstar / 3)

        # No-learning private priors: mu_k = Lambda(tau u_k)
        mu_NL = (lam(tau * u1), lam(tau * u2), lam(tau * u3))

        def excess_NL(p):
            return sum(crra_demand_binary(m, p, gamma) for m in mu_NL)

        p_NL = bisect_market_clear(excess_NL)

        # REE: market clearing using converged mu*(u, p)
        u_vec = (u1, u2, u3)

        def excess_REE(p):
            return sum(
                crra_demand_binary(interp_mu(u_vec[j], p), p, gamma)
                for j in range(3)
            )

        p_REE = bisect_market_clear(excess_REE)

        rows.append((u3, Tstar, p_FR, p_NL, p_REE))
        print(f"  u3={fmt_mp(u3,8)}  T*={fmt_mp(Tstar,10)}  "
              f"p_FR={fmt_mp(p_FR,12)}  p_NL={fmt_mp(p_NL,12)}  p_REE={fmt_mp(p_REE,12)}")

    # Pgfplots output (~24 digits per coordinate, mp-faithful)
    fr_coords = "".join(f"({fmt_mp(T,12)},{fmt_mp(fr,24)})" for (_, T, fr, _, _) in rows)
    nl_coords = "".join(f"({fmt_mp(T,12)},{fmt_mp(nl,24)})" for (_, T, _, nl, _) in rows)
    re_coords = "".join(f"({fmt_mp(T,12)},{fmt_mp(re,24)})" for (_, T, _, _, re) in rows)

    Tmin, Tmax = rows[0][1], rows[-1][1]
    lines = [
        f"% Fig 5 (asymmetric triples): price vs T*  [mpmath dps=100, tol 1e-50]",
        f"% u1 = {fmt_mp(u1,24)}",
        f"% u2 = {fmt_mp(u2,24)}",
        f"% u3 in [{fmt_mp(u3_lo,4)}, {fmt_mp(u3_hi,4)}], {N_POINTS} pts",
        f"% T* = tau*(u1+u2+u3) in [{fmt_mp(Tmin,12)}, {fmt_mp(Tmax,12)}]",
        f"% G={G} UMAX={d['UMAX']} gamma={fmt_mp(gamma,8)} tau={fmt_mp(tau,8)}",
        f"% Seed F_max = {fmt_mp(mp.mpf(d['F_max']),24)}",
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
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
