"""FD-check _contour_sum_tangent at the bad cell's actual slice."""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


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

    # Bad cell: out=(4,2,3), in=(3,2,3). Agent 1 (slice = P[:,2,:]).
    out_ijl = (4, 2, 3); in_ijl = (3, 2, 3)
    i, j, l = out_ijl
    p_obs = P[i, j, l]

    V = np.zeros_like(P); V[in_ijl] = 1.0

    print(f"At cell {out_ijl}: p_obs={p_obs:.6e}")

    for ag in [1, 2]:
        slice_, slice_dot = pj._slice_for_agent(P, V, ag, i, j, l)
        if ag == 0:
            tau_A = taus[1]; tau_B = taus[2]
        elif ag == 1:
            tau_A = taus[0]; tau_B = taus[2]
        else:
            tau_A = taus[0]; tau_B = taus[1]
        p_obs_dot = V[i, j, l]  # 0 in this case (out != in)
        print(f"\n--- Agent {ag} ---")
        print(f"  p_obs_dot=V[i,j,l]={p_obs_dot}")
        print(f"  slice range: [{slice_.min():.4f}, {slice_.max():.4f}]")
        print(f"  slice_dot nonzero: {(slice_dot != 0).sum()} cells, max|slice_dot|={np.abs(slice_dot).max()}")

        A0, A1, A0d, A1d = pj._contour_sum_tangent(
            slice_, slice_dot, u, tau_A, tau_B, p_obs, p_obs_dot)
        # FD over slice perturbation
        eps = 1e-7
        Ap0_s, Ap1_s = rp._contour_sum_pchip(
            slice_ + eps * slice_dot, u, tau_A, tau_B, p_obs)
        Am0_s, Am1_s = rp._contour_sum_pchip(
            slice_ - eps * slice_dot, u, tau_A, tau_B, p_obs)
        fd_A0_s = (Ap0_s - Am0_s) / (2 * eps)
        fd_A1_s = (Ap1_s - Am1_s) / (2 * eps)
        print(f"  ana A0d={A0d:+.4e} fd A0d (slice only)={fd_A0_s:+.4e}")
        print(f"  ana A1d={A1d:+.4e} fd A1d (slice only)={fd_A1_s:+.4e}")

        # Sweep eps
        for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            Ap0, _ = rp._contour_sum_pchip(
                slice_ + eps * slice_dot, u, tau_A, tau_B, p_obs)
            Am0, _ = rp._contour_sum_pchip(
                slice_ - eps * slice_dot, u, tau_A, tau_B, p_obs)
            print(f"    eps={eps:.0e}: fd A0d = {(Ap0-Am0)/(2*eps):+.4e}")

if __name__ == "__main__":
    main()
