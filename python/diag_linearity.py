"""Diagnostic 3: is the analytic tangent linear in V?

If yes, ana(V_a + V_b) == ana(V_a) + ana(V_b).
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

    # Two single-coord perturbations
    Va = np.zeros_like(P); Va[0, 0, 0] = 1.0
    Vb = np.zeros_like(P); Vb[1, 1, 1] = 1.0
    Vab = Va + Vb

    out_a = pj.J_dot_v_with_phi(P, Va, Phi_P, u, taus, gammas, Ws)
    out_b = pj.J_dot_v_with_phi(P, Vb, Phi_P, u, taus, gammas, Ws)
    out_ab = pj.J_dot_v_with_phi(P, Vab, Phi_P, u, taus, gammas, Ws)

    sum_ab = out_a + out_b
    err = np.abs(out_ab - sum_ab)
    print(f"Linearity (single-coord pair): max|J(V_a+V_b) - J(V_a) - J(V_b)| = {err.max():.3e}")

    # Try a few more pairs
    print("\nLinearity at random pairs:")
    for trial in range(5):
        idx_a = tuple(rng.integers(0, G, 3))
        idx_b = tuple(rng.integers(0, G, 3))
        Va = np.zeros_like(P); Va[idx_a] = 1.0
        Vb = np.zeros_like(P); Vb[idx_b] = 1.0
        Vab = Va + Vb
        out_a = pj.J_dot_v_with_phi(P, Va, Phi_P, u, taus, gammas, Ws)
        out_b = pj.J_dot_v_with_phi(P, Vb, Phi_P, u, taus, gammas, Ws)
        out_ab = pj.J_dot_v_with_phi(P, Vab, Phi_P, u, taus, gammas, Ws)
        err = np.abs(out_ab - (out_a + out_b)).max()
        # Also baseline: where do they differ?
        diff = out_ab - (out_a + out_b)
        worst = np.unravel_index(np.abs(diff).argmax(), P.shape)
        print(f"  V@{idx_a} + V@{idx_b}: max linearity err = {err:.3e}  worst@{worst}")

    # Scaling test on random V
    print("\nScaling: ana(2V) vs 2*ana(V) for random V?")
    V = rng.standard_normal(P.shape)
    out_1 = pj.J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)
    out_2 = pj.J_dot_v_with_phi(P, 2 * V, Phi_P, u, taus, gammas, Ws)
    err = np.abs(out_2 - 2 * out_1).max()
    print(f"  max|J(2V) - 2J(V)| = {err:.3e}")
    print(f"  ‖J(V)‖∞ = {np.abs(out_1).max():.3e}")
    print(f"  ‖J(2V)‖∞ = {np.abs(out_2).max():.3e}")


if __name__ == "__main__":
    main()
