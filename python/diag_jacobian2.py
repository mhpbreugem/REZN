"""Diagnostic 2: are the J·V failures FD noise from too-small eps·V?

Strategy: scale V to O(1) magnitude and use eps=1e-7 (matching the
single-coord tests where everything passed).
"""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import pchip_jacobian as pj


def _phi(P, u, taus, gammas, Ws):
    return rp._phi_map_pchip(P, u,
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
    P = np.clip(P0 + 0.01 * rng.standard_normal(P0.shape), 1e-9, 1 - 1e-9)

    Phi_P = _phi(P, u, taus, gammas, Ws)

    # ---- Test: random V of size 1.0 -----------------------------------
    print("Random V scaled to O(1) magnitude:")
    for V_scale, eps in [(1.0, 1e-6), (1.0, 1e-7), (0.1, 1e-7), (0.01, 1e-7),
                          (0.001, 1e-7), (0.001, 1e-6), (0.001, 1e-5)]:
        V = rng.standard_normal(P.shape) * V_scale
        # analytic
        out = pj.J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)
        # FD
        Pp = _phi(P + eps * V, u, taus, gammas, Ws)
        Pm = _phi(P - eps * V, u, taus, gammas, Ws)
        F_p = (P + eps * V) - Pp
        F_m = (P - eps * V) - Pm
        JV_fd = (F_p - F_m) / (2 * eps)
        err = np.abs(out - JV_fd).max()
        rel = err / max(1.0, np.abs(JV_fd).max())
        print(f"  V_scale={V_scale:.0e} eps={eps:.0e}: max_abs_err={err:.3e} "
              f"rel={rel:.3e}  ‖JV‖∞={np.abs(out).max():.3e} ‖JVfd‖∞={np.abs(JV_fd).max():.3e}")


if __name__ == "__main__":
    main()
