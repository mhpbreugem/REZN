"""FD using the SAME outer clip as the Newton solver would apply.

If the analytic ana matches FD-with-clip but not FD-without-clip, the bug
is solely in not respecting the outer clip in the tangent.
"""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import pchip_jacobian as pj


CLIP_LO = 1e-9
CLIP_HI = 1 - 1e-9


def phi_clipped(P, u, taus, gammas, Ws):
    """Evaluate Φ on P clipped to [1e-9, 1-1e-9]."""
    Pc = np.clip(P, CLIP_LO, CLIP_HI)
    return rp._phi_map_pchip(Pc, u,
                              np.asarray(taus, float),
                              np.asarray(gammas, float),
                              np.asarray(Ws, float))


def main():
    G = 5
    UMAX = 2.0
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([3.0, 3.0, 3.0])
    gammas = np.array([3.0, 3.0, 3.0])
    Ws = np.array([1.0, 1.0, 1.0])

    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                 maxiters=300, abstol=1e-13, alpha=1.0)
    P0 = res["P_star"]
    rng = np.random.default_rng(7)
    P = np.clip(P0 + 0.01 * rng.standard_normal(P0.shape), CLIP_LO, CLIP_HI)
    Phi_P = phi_clipped(P, u, taus, gammas, Ws)

    # bad cell: out=(4,2,3), in=(3,2,3), V=e_in
    in_ijl = (3, 2, 3); out_ijl = (4, 2, 3)
    V = np.zeros_like(P); V[in_ijl] = 1.0

    # Analytic
    JV = pj.J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)
    print(f"Analytic JV at out={out_ijl}: {JV[out_ijl]:+.4e}")

    print("\nFD with explicit clip on P+εV (Newton would clip):")
    for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        Pp = phi_clipped(P + eps * V, u, taus, gammas, Ws)
        Pm = phi_clipped(P - eps * V, u, taus, gammas, Ws)
        Fp = np.clip(P + eps * V, CLIP_LO, CLIP_HI) - Pp
        Fm = np.clip(P - eps * V, CLIP_LO, CLIP_HI) - Pm
        JV_fd = (Fp - Fm) / (2 * eps)
        print(f"  eps={eps:.0e}: fd JV[out] = {JV_fd[out_ijl]:+.4e}")

    print("\nFD without clip:")
    for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        Pp = rp._phi_map_pchip(P + eps * V, u, taus, gammas, Ws)
        Pm = rp._phi_map_pchip(P - eps * V, u, taus, gammas, Ws)
        Fp = (P + eps * V) - Pp
        Fm = (P - eps * V) - Pm
        JV_fd = (Fp - Fm) / (2 * eps)
        print(f"  eps={eps:.0e}: fd JV[out] = {JV_fd[out_ijl]:+.4e}")


if __name__ == "__main__":
    main()
