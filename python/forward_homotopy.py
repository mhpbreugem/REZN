"""Forward τ-homotopy at γ=3, G=11, watching for the basin shift to PR.

Per HANDOFF.md: at γ=3 the strong-PR branch becomes reachable by warm-
starting NK at τ slightly above the basin boundary at τ ≈ 3.39. We
sweep τ = 3.00 → 3.50 in steps of 0.05, warm-starting each step from
the previous solution, and watch for the jump in 1-R² that signals
the PR branch.

If found, we save the PR P_star to a pickle for use as a backward-
sweep seed and as warm-start for the figure drivers.
"""
from __future__ import annotations
import os
import time
import pickle
import numpy as np
from scipy.optimize import newton_krylov

import rezn_pchip as rp
import rezn_het as rh

G, UMAX = 11, 2.0
GAMMA = 3.0
TAUS = np.arange(3.00, 3.51, 0.05)
OUT_PKL = "/home/user/REZN/python/pr_seeds.pkl"


def F_factory(taus, gammas, Ws, u):
    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus, gammas, Ws)
        return x - Pn.reshape(-1)
    return F


def main():
    u = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])
    gammas = np.array([GAMMA, GAMMA, GAMMA])

    P_warm = None
    saved = {}
    print(f"τ-homotopy at γ={GAMMA}, G={G}, watching for PR jump\n")
    print(f"{'τ':>6} {'iters':>6} {'Finf':>10} {'1-R²':>10} {'PR-gap':>10} {'p_(1,-1,1)':>11} {'src':>6} {'time':>6}")
    for tau in TAUS:
        taus = np.array([tau, tau, tau])
        if P_warm is None:
            P_init = rh._nolearning_price(u, taus, gammas, Ws)
            init_tag = "cold"
        else:
            P_init = P_warm
            init_tag = "warm"
        F = F_factory(taus, gammas, Ws, u)
        x0 = np.clip(P_init, 1e-9, 1 - 1e-9).reshape(-1)
        t0 = time.time()
        # Try Picard first (stable warm-start), then NK
        res_p = rp.solve_picard_pchip(
            G, taus, gammas, umax=UMAX,
            maxiters=600, abstol=1e-7, alpha=0.3,
            P_init=P_init)
        Finf_p = float(np.abs(res_p["residual"]).max())
        try:
            sol = newton_krylov(F, res_p["P_star"].reshape(-1),
                                  f_tol=1e-7, rdiff=1e-8,
                                  method="lgmres", maxiter=40)
            P_nk = sol.reshape(G, G, G)
            Finf_nk = float(np.abs(F(sol)).max())
        except Exception:
            P_nk = res_p["P_star"]; Finf_nk = Finf_p
        if Finf_nk < Finf_p:
            P = P_nk; src = "NK"; Finf = Finf_nk
        else:
            P = res_p["P_star"]; src = "P0.3"; Finf = Finf_p
        one_r2 = rh.one_minus_R2(P, u, taus)
        ir = int(np.argmin(np.abs(u - 1.0)))
        jr = int(np.argmin(np.abs(u + 1.0)))
        lr = int(np.argmin(np.abs(u - 1.0)))
        p_star = float(P[ir, jr, lr])
        mu = rh.posteriors_at(ir, jr, lr, p_star, P, u, taus)
        print(f"{tau:6.2f} {len(res_p['history']):6d} {Finf:10.3e} "
              f"{one_r2:10.4e} {mu[0]-mu[1]:+10.4f} {p_star:11.4f}  "
              f"{src:>4s} {time.time()-t0:5.1f}s",
              flush=True)
        # If PR detected (1-R² jumps to 0.04+), save
        if one_r2 > 0.02:
            saved[float(tau)] = P.copy()
            print(f"   => PR detected at τ={tau:.2f}, saved", flush=True)
        # warm-start next step from this solution
        P_warm = P

    if saved:
        with open(OUT_PKL, "wb") as f:
            pickle.dump(saved, f)
        print(f"\nSaved {len(saved)} PR seeds to {OUT_PKL}")
    else:
        print(f"\nNo PR detected — sweep stayed in FR basin.")


if __name__ == "__main__":
    main()
