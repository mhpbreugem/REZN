"""Probe whether analytic-Newton + perturbed seeds reach the strong-PR
branch of the CRRA REE at γ=0.5, τ=2.

If a seed lands on a fixed point with non-trivial 1-R² (≫1e-3) and
PR-gap μ_1 - μ_2 distinct from zero, we have the strong-PR branch
and can use that P_init for the τ/γ/K sweeps in Figs 6-10.
"""
from __future__ import annotations
import os
import time
import numpy as np

import pchip_jacobian as pj
import rezn_pchip as rp
import rezn_het as rh


G    = 11
UMAX = 2.0
TAU  = 2.0
GAMMA = 0.5


def fingerprint(P, u, taus):
    G = u.shape[0]
    finf = float(np.abs(P - rp._phi_map_pchip(P, u, taus, taus*0+GAMMA,
                                                 np.array([1.0,1.0,1.0]))).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    i = int(np.argmin(np.abs(u - 1.0)))
    j = int(np.argmin(np.abs(u + 1.0)))
    l = int(np.argmin(np.abs(u - 1.0)))
    p_star = float(P[i, j, l])
    mu = rh.posteriors_at(i, j, l, p_star, P, u, taus)
    return one_r2, p_star, mu, finf


def main():
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([TAU, TAU, TAU])
    gammas = np.array([GAMMA, GAMMA, GAMMA])
    Ws = np.array([1.0, 1.0, 1.0])

    P0 = rh._nolearning_price(u, taus, gammas, Ws)
    print(f"=== γ={GAMMA}, τ={TAU}, G={G} ===")
    one_r2, p_star, mu, finf = fingerprint(P0, u, taus)
    print(f"NL seed:     1-R²={one_r2:.4e}  p_(1,-1,1)={p_star:.4f}  "
           f"μ={tuple(round(m, 4) for m in mu)}  Finf={finf:.2e}")

    seeds = {
        "no-learning":   P0.copy(),
        "average-1/2":   np.full_like(P0, 0.5),
        "logit-amp×2":   None,        # built below
        "logit-amp×0.5": None,
        "rand-σ=0.1":    None,
        "rand-σ=0.2":    None,
    }
    eps = 1e-9
    L0 = np.log(np.clip(P0, eps, 1-eps) / (1 - np.clip(P0, eps, 1-eps)))
    L0_mean = L0.mean()
    seeds["logit-amp×2"]   = 1.0 / (1.0 + np.exp(-(L0_mean + 2.0 * (L0 - L0_mean))))
    seeds["logit-amp×0.5"] = 1.0 / (1.0 + np.exp(-(L0_mean + 0.5 * (L0 - L0_mean))))
    rng = np.random.default_rng(0)
    seeds["rand-σ=0.1"] = np.clip(P0 + 0.1 * rng.standard_normal(P0.shape),
                                     1e-9, 1 - 1e-9)
    rng = np.random.default_rng(1)
    seeds["rand-σ=0.2"] = np.clip(P0 + 0.2 * rng.standard_normal(P0.shape),
                                     1e-9, 1 - 1e-9)

    for name, P_init in seeds.items():
        t0 = time.time()
        try:
            res = pj.solve_newton(
                G, taus, gammas, umax=UMAX,
                P_init=P_init, maxiters=15, abstol=1e-9,
                lgmres_tol=1e-7, lgmres_maxiter=120)
            P = res["P_star"]
            one_r2, p_star, mu, finf = fingerprint(P, u, taus)
            tag = "PR?" if one_r2 > 1e-3 else "FR"
            print(f"  [{name:18s}]  iters={len(res['history']):2d}  "
                  f"Finf={res['best_Finf']:.1e}  1-R²={one_r2:.4e}  "
                  f"PR-gap={mu[0]-mu[1]:+.4f}  [{tag}]  "
                  f"({time.time()-t0:.1f}s)")
        except Exception as ex:
            print(f"  [{name:18s}]  FAIL: {ex}")


if __name__ == "__main__":
    main()
