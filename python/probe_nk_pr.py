"""Reproduce the documented PR-branch result at γ=3, τ=3, G=11
using scipy newton_krylov directly (the method pchip_continuation
used to find the strong-PR branch).

Expected: 1-R² ≈ 0.057, p_(1,-1,1) ≈ 0.632, μ ≈ (0.645, 0.633, 0.645).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize.nonlin import NoConvergence

import rezn_pchip as rp
import rezn_het as rh


G, UMAX = 11, 2.0


def try_nk(tau, gamma, seed_name, P_init):
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([tau, tau, tau])
    gammas = np.array([gamma, gamma, gamma])
    Ws = np.array([1.0, 1.0, 1.0])

    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus, gammas, Ws)
        return x - Pn.reshape(-1)

    x0 = np.clip(P_init, 1e-9, 1 - 1e-9).reshape(-1)
    t0 = time.time()
    try:
        sol = newton_krylov(F, x0, f_tol=1e-6, rdiff=1e-8,
                              method="lgmres", maxiter=40,
                              verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
    P = sol.reshape(G, G, G)
    Finf = float(np.abs(F(sol)).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    i = int(np.argmin(np.abs(u - 1.0)))
    j = int(np.argmin(np.abs(u + 1.0)))
    l = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[i, j, l])
    mu = rh.posteriors_at(i, j, l, p, P, u, taus)
    print(f"  [{seed_name:18s}]  γ={gamma} τ={tau}  Finf={Finf:.2e}  "
          f"1-R²={one_r2:.4e}  PR-gap={mu[0]-mu[1]:+.4f}  "
          f"p_(1,-1,1)={p:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return P, Finf, one_r2


def main():
    u = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])

    print("=== γ=3, τ=3 (PR branch documented in HANDOFF) ===")
    taus = np.array([3.0, 3.0, 3.0])
    gammas = np.array([3.0, 3.0, 3.0])
    P_NL = rh._nolearning_price(u, taus, gammas, Ws)

    # Try several seeds
    try_nk(3.0, 3.0, "no-learning", P_NL.copy())

    rng = np.random.default_rng(42)
    P_pert = np.clip(P_NL + 0.1 * rng.standard_normal(P_NL.shape),
                       1e-9, 1 - 1e-9)
    try_nk(3.0, 3.0, "NL+0.1·rand", P_pert)

    P_pert2 = np.clip(P_NL + 0.2 * rng.standard_normal(P_NL.shape),
                        1e-9, 1 - 1e-9)
    try_nk(3.0, 3.0, "NL+0.2·rand", P_pert2)

    # Try a "broken-symmetry" seed by tilting toward the PR direction
    # Reference values from backward_snap: μ=(0.645, 0.633, 0.645).
    # Tilt the no-learning P slightly so logit(P) is more sensitive to
    # the (sign of u_2) — break the agent-symmetry.
    sign_tilt = np.zeros_like(P_NL)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                # subtle bias toward giving more weight to u_j
                sign_tilt[i, j, l] = 0.05 * np.tanh(u[j])
    P_tilt = np.clip(P_NL + sign_tilt, 1e-9, 1 - 1e-9)
    try_nk(3.0, 3.0, "NL + sign-tilt", P_tilt)


if __name__ == "__main__":
    main()
