"""Direct test: does an analytic-Jacobian Newton step reduce ‖F‖?

If yes, the analytic is fit-for-purpose regardless of what FD says at
boundary cells. Wire it up as a LinearOperator and run the sweep.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
import rezn_pchip as rp
import pchip_jacobian as pj


CLIP_LO = 1e-9
CLIP_HI = 1 - 1e-9


def F_residual(P, u, taus, gammas, Ws):
    """F(P) = P - Φ(P)."""
    return P - rp._phi_map_pchip(P, u, taus, gammas, Ws)


def newton_step(P, u, taus, gammas, Ws, tol=1e-10, maxiter=80):
    """One Newton step using lgmres on J·dP = -F."""
    G = P.shape[0]
    N = G ** 3

    F = F_residual(P, u, taus, gammas, Ws)
    Phi_P = P - F  # = Φ(P)

    def matvec(v):
        V = v.reshape(P.shape)
        out = pj.J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)
        return out.reshape(-1)

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    dP_flat, info = lgmres(op, -F.reshape(-1), rtol=tol, atol=0.0,
                            maxiter=maxiter)
    return dP_flat.reshape(P.shape), info


def main():
    for G, taus, gammas in [(7, (3, 3, 3), (3, 3, 3)),
                              (5, (2, 2, 2), (3, 3, 3))]:
        UMAX = 2.0
        u = np.linspace(-UMAX, UMAX, G)
        taus_a = np.asarray(taus, float)
        gammas_a = np.asarray(gammas, float)
        Ws_a = np.array([1.0, 1.0, 1.0])
        print(f"\n=== G={G} τ={taus} γ={gammas} ===")
        # Warm-start with Picard
        res = rp.solve_picard_pchip(G, taus_a, gammas_a, umax=UMAX,
                                     maxiters=200, abstol=1e-9, alpha=1.0)
        P = res["P_star"]
        F0 = F_residual(P, u, taus_a, gammas_a, Ws_a)
        Finf0 = float(np.abs(F0).max())
        print(f"  After Picard: ‖F‖∞ = {Finf0:.3e}")

        # Newton step with analytic J
        dP, info = newton_step(P, u, taus_a, gammas_a, Ws_a, tol=1e-8)
        print(f"  lgmres info={info}, ‖dP‖∞={np.abs(dP).max():.3e}")

        # Try several step sizes (line search)
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            P_new = np.clip(P + alpha * dP, CLIP_LO, CLIP_HI)
            F_new = F_residual(P_new, u, taus_a, gammas_a, Ws_a)
            Finf_new = float(np.abs(F_new).max())
            print(f"  α={alpha:6.4f}: ‖F‖∞ = {Finf_new:.3e}  "
                  f"({'OK' if Finf_new < Finf0 else 'WORSE'})")


if __name__ == "__main__":
    main()
