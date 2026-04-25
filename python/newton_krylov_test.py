"""Newton-Krylov solve on the PCHIP Φ map.

Picard + Anderson stalled at τ<2.37 at G=11. Here we try matrix-free
Newton via scipy.optimize.newton_krylov, which solves F(P) = P - Φ(P) = 0
using inexact Newton with GMRES for the inner linear solve. Jacobian
products J·v are obtained via a finite-difference directional derivative
of F, so no Jacobian storage is needed.

Trade-off versus Picard: each Newton-Krylov outer iteration needs roughly
20-50 Φ evaluations (for the GMRES vector products), but converges
quadratically near the fixed point, so typically 5-10 outer iterations
suffice. Ideal for stiff configs where Picard's spectral radius is close
to 1.

First we smoke-test on a known stuck config (τ=2.3, γ=3, G=11) and report.
"""
from __future__ import annotations
import sys
import time
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

import rezn_het as rh
import rezn_pchip as rp


G      = 11
UMAX   = 2.0
TAU    = 2.3
GAMMA  = 3.0
TOL    = 1e-8


def build_F(taus, gammas, Ws):
    """Return F(x) = vec(x) - vec(Phi_PCHIP(x))."""
    u = np.linspace(-UMAX, UMAX, G)
    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus, gammas, Ws)
        return x - Pn.reshape(-1)
    return F, u


def main():
    print(f"Newton-Krylov smoke test  τ=({TAU},)×3  γ=({GAMMA},)×3  G={G}")
    sys.stdout.flush()

    # Warmup JIT
    _ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

    taus   = rh._as_vec3(TAU)
    gammas = rh._as_vec3(GAMMA)
    Ws     = rh._as_vec3(1.0)

    # Seed from no-learning, then pre-iterate Picard for a few steps
    u = np.linspace(-UMAX, UMAX, G)
    P0 = rh._nolearning_price(u, taus, gammas, Ws)
    print("Pre-iterating Picard 50 steps to get close…")
    sys.stdout.flush()
    Pcur = P0.copy()
    for i in range(50):
        Pnew = rp._phi_map_pchip(Pcur, u, taus, gammas, Ws)
        Pcur = Pnew
    Fvec = (Pcur.reshape(-1) - rp._phi_map_pchip(Pcur, u, taus, gammas, Ws).reshape(-1))
    print(f"  after 50 Picard: ||F||∞ = {np.abs(Fvec).max():.3e}")

    F, _ = build_F(taus, gammas, Ws)

    print("Starting Newton-Krylov…")
    sys.stdout.flush()

    # newton_krylov solves F(x) = 0.
    t0 = time.time()
    try:
        sol = newton_krylov(
            F, Pcur.reshape(-1),
            f_tol=TOL,
            rdiff=1e-7,
            method="lgmres",
            verbose=True,
            maxiter=30,
        )
        dt = time.time() - t0
        Fvec = F(sol)
        print(f"\n[converged] time={dt:.1f}s  ||F||∞={np.abs(Fvec).max():.3e}")
    except NoConvergence as e:
        dt = time.time() - t0
        sol = e.args[0]
        Fvec = F(sol)
        print(f"\n[NoConvergence] time={dt:.1f}s  ||F||∞={np.abs(Fvec).max():.3e}")

    # Report at (1,-1,1)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i, j, l = idx(1.0), idx(-1.0), idx(1.0)
    Psol = sol.reshape(G, G, G)
    mu = rh.posteriors_at(i, j, l, float(Psol[i,j,l]), Psol, u, taus)
    print(f"\nAt cell (u1,u2,u3) ≈ ({u[i]:.3f}, {u[j]:.3f}, {u[l]:.3f}):")
    print(f"  p*  = {float(Psol[i,j,l]):.12f}")
    print(f"  μ1  = {mu[0]:.12f}")
    print(f"  μ2  = {mu[1]:.12f}")
    print(f"  μ3  = {mu[2]:.12f}")
    print(f"  PR gap μ1-μ2 = {mu[0]-mu[1]:.6e}")


if __name__ == "__main__":
    main()
