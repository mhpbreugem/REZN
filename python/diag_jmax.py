"""Sweep all single-coord directions, find largest entries of dΦ/dP."""
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

    # Build J = -dΦ/dP one column at a time. Use J_dot_v_with_phi which is
    # I - dΦ/dP, so dΦ_col = V - J·V.
    N = G ** 3
    dPhi = np.zeros((N, N))  # dΦ[out_idx, in_idx]
    for in_idx in range(N):
        V = np.zeros_like(P)
        i0, j0, l0 = np.unravel_index(in_idx, P.shape)
        V[i0, j0, l0] = 1.0
        JV = pj.J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)
        # dΦ·e_k = V - JV
        dPhi[:, in_idx] = (V - JV).ravel()

    print(f"max|dΦ/dP| analytic over all entries: {np.abs(dPhi).max():.3e}")
    # Compare to FD column for the same input
    in_idx_max = np.unravel_index(np.abs(dPhi).argmax(), (N, N))
    out_flat, in_flat = in_idx_max
    out_ijl = np.unravel_index(out_flat, P.shape)
    in_ijl = np.unravel_index(in_flat, P.shape)
    print(f"max@out={out_ijl} in={in_ijl}: ana={dPhi[out_flat, in_flat]:+.4e}")

    # FD for that column
    eps = 1e-7
    V = np.zeros_like(P); V[in_ijl] = 1.0
    Pp = _phi(P + eps * V, u, taus, gammas, Ws)
    Pm = _phi(P - eps * V, u, taus, gammas, Ws)
    fd_col = (Pp - Pm) / (2 * eps)
    print(f"FD column at in={in_ijl}: max|dΦ/dP_in|={np.abs(fd_col).max():.3e}")
    print(f"  fd at out={out_ijl}: {fd_col[out_ijl]:+.4e}")

    # Test linearity of Φ near P: is FD reliable at this in_idx?
    # Show FD at multiple eps
    for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        Pp = _phi(P + eps * V, u, taus, gammas, Ws)
        Pm = _phi(P - eps * V, u, taus, gammas, Ws)
        fd_val = (Pp[out_ijl] - Pm[out_ijl]) / (2 * eps)
        print(f"  eps={eps:.0e}: fd[out]={fd_val:+.4e}  ana={dPhi[out_flat, in_flat]:+.4e}")


if __name__ == "__main__":
    main()
