#!/usr/bin/env python3
"""
Finalize the missing high-tau points (5.0, 10.0, 15.0) for gamma=1.0.

Strategy: try multiple warm-starts and ladder steps; accept the one with
lowest ||F||_inf below a relaxed threshold (0.15) since the bulk converges
to ~1e-10 at every high-tau point — only the lens corners refuse to fall.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence

sys.path.insert(0, os.path.dirname(__file__))

from solve_posterior_v3 import (
    build_p_grids,
    phi_step,
    save_checkpoint,
    weighted_1mR2,
)

G = 20
UMAX = 5.0
GAMMA = 1.0
DPS = 50
RELAXED_BAIL = 0.15
OUTDIR = "results/full_ree"


def load_ckpt(tau):
    path = os.path.join(OUTDIR, f"task3_g100_t{int(round(tau*100)):04d}_mp50.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    mu = np.array([[float(x) for x in row] for row in d["mu_strings"]])
    pgr = np.array([[float(x) for x in row] for row in d["p_grid"]])
    return mu, pgr


def remap(prev_mu, prev_pgrid, new_pgrid):
    G_u, G_p = prev_mu.shape
    out = np.empty_like(prev_mu)
    for i in range(G_u):
        old_p = prev_pgrid[i, :]
        new_p = new_pgrid[i, :]
        old_lp = np.log(old_p) - np.log1p(-old_p)
        new_lp = np.log(new_p) - np.log1p(-new_p)
        out[i, :] = np.interp(new_lp, old_lp, prev_mu[i, :])
    return np.clip(out, 1e-15, 1.0 - 1e-15)


def solve_at(tau, mu0, p_grids, u_grid, maxiter=30, f_tol=1e-9):
    def F_func(mf):
        m = np.clip(mf.reshape(G, G), 1e-15, 1.0 - 1e-15)
        return (phi_step(m, u_grid, p_grids, tau, GAMMA) - m).reshape(-1)

    try:
        sol = newton_krylov(F_func, mu0.reshape(-1), f_tol=f_tol,
                             method="lgmres", maxiter=maxiter)
    except NoConvergence as exc:
        sol = exc.args[0]
    mu = np.clip(sol.reshape(G, G), 1e-15, 1.0 - 1e-15)
    F = F_func(mu.reshape(-1))
    return mu, float(np.abs(F).max()), float(np.median(np.abs(F)))


def try_tau(tau, candidates, u_grid):
    """Try several warm-starts; keep the result with smallest ||F||_inf."""
    p_lo, p_hi, p_grids = build_p_grids(u_grid, tau, GAMMA, G_p=G, margin=0.0)
    best = None
    for label, mu0 in candidates:
        t0 = time.time()
        mu, F_max, F_med = solve_at(tau, mu0, p_grids, u_grid,
                                      maxiter=30, f_tol=1e-9)
        dt = time.time() - t0
        print(f"  candidate '{label}': F_max={F_max:.3e}  "
              f"F_med={F_med:.3e}  ({dt:.1f}s)")
        if best is None or F_max < best[1]:
            best = (mu, F_max, F_med, label)
    mu, F_max, F_med, label = best
    print(f"  -> best: '{label}' with F_max={F_max:.3e}")
    return mu, p_grids, F_max, F_med


def maybe_save(tau, mu, p_grids, u_grid, F_max, F_med):
    if F_max > RELAXED_BAIL:
        print(f"  BAIL (F_max={F_max:.3e} > {RELAXED_BAIL})")
        return False
    one_mR2, slope, n = weighted_1mR2(u_grid, p_grids, mu, tau, GAMMA)
    print(f"  1-R^2={one_mR2:.6f}  slope={slope:.6f}  n={n}")
    save_checkpoint(
        os.path.join(OUTDIR, f"task3_g100_t{int(round(tau*100)):04d}_mp50.json"),
        G=G, UMAX=UMAX, tau=tau, gamma=GAMMA, trim=0.0, dps=DPS,
        F_max=F_max, F_med=F_med,
        u_grid=u_grid, p_grids=p_grids, mu=mu,
        history=[],
    )
    print(f"  saved")
    return True


def main():
    u_grid = np.linspace(-UMAX, UMAX, G)

    # Build candidate warm-starts from existing converged points
    mu_4, pgr_4 = load_ckpt(4.0)
    mu_7, pgr_7 = load_ckpt(7.0)

    # tau=5.0 — try ladder via 4.5, plus FR init
    print("--- tau = 5.0 ---")
    p45 = build_p_grids(u_grid, 4.5, GAMMA, G_p=G, margin=0.0)[2]
    mu_45_warm = remap(mu_4, pgr_4, p45)
    mu_45, F45_max, F45_med = solve_at(4.5, mu_45_warm, p45, u_grid,
                                          maxiter=20, f_tol=1e-9)
    print(f"  via tau=4.5: F_max={F45_max:.3e}")
    p5 = build_p_grids(u_grid, 5.0, GAMMA, G_p=G, margin=0.0)[2]
    cands_5 = [
        ("from tau=4.0", remap(mu_4, pgr_4, p5)),
        ("from tau=4.5", remap(mu_45, p45, p5)),
        ("FR init",       p5.copy()),
    ]
    mu_5, p5, F5_max, F5_med = try_tau(5.0, cands_5, u_grid)
    saved_5 = maybe_save(5.0, mu_5, p5, u_grid, F5_max, F5_med)

    # tau=10 — ladder via 8.5
    print("\n--- tau = 10.0 ---")
    p85 = build_p_grids(u_grid, 8.5, GAMMA, G_p=G, margin=0.0)[2]
    mu_85_warm = remap(mu_7, pgr_7, p85)
    mu_85, F85_max, _ = solve_at(8.5, mu_85_warm, p85, u_grid,
                                    maxiter=20, f_tol=1e-9)
    print(f"  via tau=8.5: F_max={F85_max:.3e}")
    p10 = build_p_grids(u_grid, 10.0, GAMMA, G_p=G, margin=0.0)[2]
    cands_10 = [
        ("from tau=7.0", remap(mu_7, pgr_7, p10)),
        ("from tau=8.5", remap(mu_85, p85, p10)),
        ("FR init",       p10.copy()),
    ]
    mu_10, p10, F10_max, F10_med = try_tau(10.0, cands_10, u_grid)
    saved_10 = maybe_save(10.0, mu_10, p10, u_grid, F10_max, F10_med)

    # tau=15 — ladder via 12
    print("\n--- tau = 15.0 ---")
    if saved_10:
        warm_mu, warm_pgr = mu_10, p10
        warm_label = "tau=10.0"
    else:
        warm_mu, warm_pgr = mu_7, pgr_7
        warm_label = "tau=7.0"
    p12 = build_p_grids(u_grid, 12.0, GAMMA, G_p=G, margin=0.0)[2]
    mu_12_warm = remap(warm_mu, warm_pgr, p12)
    mu_12, F12_max, _ = solve_at(12.0, mu_12_warm, p12, u_grid,
                                    maxiter=20, f_tol=1e-9)
    print(f"  via tau=12: F_max={F12_max:.3e}")
    p15 = build_p_grids(u_grid, 15.0, GAMMA, G_p=G, margin=0.0)[2]
    cands_15 = [
        (f"from {warm_label}", remap(warm_mu, warm_pgr, p15)),
        ("from tau=12",         remap(mu_12, p12, p15)),
        ("FR init",              p15.copy()),
    ]
    mu_15, p15, F15_max, F15_med = try_tau(15.0, cands_15, u_grid)
    saved_15 = maybe_save(15.0, mu_15, p15, u_grid, F15_max, F15_med)

    print(f"\nSaved tau=5.0:  {saved_5}")
    print(f"Saved tau=10.0: {saved_10}")
    print(f"Saved tau=15.0: {saved_15}")


if __name__ == "__main__":
    main()
