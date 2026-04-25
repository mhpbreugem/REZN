"""Print internals of phi_tangent_at_cell at the bad cell."""
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
    Phi_P = rp._phi_map_pchip(P, u, taus, gammas, Ws)

    out_ijl = (4, 2, 3)
    in_ijl = (3, 2, 3)
    i, j, l = out_ijl
    p_star = Phi_P[i, j, l]
    p_obs = P[i, j, l]
    print(f"At out cell {out_ijl}: P[i,j,l]={p_obs:.6e} Φ[i,j,l]={p_star:.6e}")

    # Posteriors
    m0, m1, m2 = rp._posteriors_at_pchip(i, j, l, p_obs, P, u, taus)
    print(f"  μ₀={m0:.6e} μ₁={m1:.6e} μ₂={m2:.6e}")
    mus = np.array([m0, m1, m2])

    # Clearing jacobian at (mus, p_star)
    h, d0, d1, d2, dp = pj._clearing_jacobian(mus, p_star, gammas, Ws)
    print(f"  h(mus, p*)={h:.3e}")
    print(f"  ∂h/∂μ₀={d0:.6e} ∂h/∂μ₁={d1:.6e} ∂h/∂μ₂={d2:.6e}")
    print(f"  ∂h/∂p={dp:.6e}")

    # Now test V = e_{in_ijl}
    V = np.zeros_like(P); V[in_ijl] = 1.0

    # phi_tangent_at_cell decomposed
    mu_dots = []
    for ag in range(3):
        slice_, slice_dot = pj._slice_for_agent(P, V, ag, i, j, l)
        if ag == 0:
            tau_own = taus[0]; tau_A = taus[1]; tau_B = taus[2]
            u_own = u[i]
        elif ag == 1:
            tau_own = taus[1]; tau_A = taus[0]; tau_B = taus[2]
            u_own = u[j]
        else:
            tau_own = taus[2]; tau_A = taus[0]; tau_B = taus[1]
            u_own = u[l]
        A0, A1, A0d, A1d = pj._contour_sum_tangent(
            slice_, slice_dot, u, tau_A, tau_B, p_obs, V[i, j, l])
        g0 = rh._f0(u_own, tau_own); g1 = rh._f1(u_own, tau_own)
        mu, mu_dot = pj._posterior_and_tangent(A0, A1, A0d, A1d, g0, g1,
                                                 u_own, tau_own)
        print(f"  ag={ag}: A0={A0:.4e} A1={A1:.4e} A0d={A0d:.4e} A1d={A1d:.4e}")
        print(f"         μ={mu:.4e} dμ={mu_dot:.4e}")
        mu_dots.append(mu_dot)

    pdot_total = -(d0 * mu_dots[0] + d1 * mu_dots[1] + d2 * mu_dots[2]) / dp
    print(f"  Σ ∂h/∂μk · dμk = {d0*mu_dots[0] + d1*mu_dots[1] + d2*mu_dots[2]:.6e}")
    print(f"  dΦ_analytic = {pdot_total:.6e}")

    # FD Φ check
    eps = 1e-7
    Phi_p = rp._phi_map_pchip(P + eps * V, u, taus, gammas, Ws)
    Phi_m = rp._phi_map_pchip(P - eps * V, u, taus, gammas, Ws)
    fd = (Phi_p[i, j, l] - Phi_m[i, j, l]) / (2 * eps)
    print(f"  dΦ_fd (eps=1e-7) = {fd:.6e}")
    # Show Φ_p, Φ_m, and Φ_0 at this cell
    print(f"    Φ(P-εV)[i,j,l]={Phi_m[i,j,l]:.10e}")
    print(f"    Φ(P+0)[i,j,l] ={p_star:.10e}")
    print(f"    Φ(P+εV)[i,j,l]={Phi_p[i,j,l]:.10e}")

    # Check if posterior changes are consistent (FD μ vs ana μ_dot)
    print("\nFD-check μ_dot per agent (eps=1e-7):")
    eps = 1e-7
    P_p = P + eps * V; P_m = P - eps * V
    p_obs_p = P_p[i, j, l]; p_obs_m = P_m[i, j, l]
    for ag in range(3):
        m_p = rp._agent_posterior_pchip(ag, i, j, l, p_obs_p, P_p, u, taus)
        m_m = rp._agent_posterior_pchip(ag, i, j, l, p_obs_m, P_m, u, taus)
        fd_mu = (m_p - m_m) / (2 * eps)
        print(f"  ag={ag}: ana μ_dot={mu_dots[ag]:+.4e} fd μ_dot={fd_mu:+.4e} "
              f"err={abs(mu_dots[ag] - fd_mu):.2e}")


if __name__ == "__main__":
    main()
