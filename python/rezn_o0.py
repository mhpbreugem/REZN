"""Order-0 (piecewise-constant) contour kernel.

Drop-in alternative to rezn_het._phi_map and rezn_pchip._phi_map_pchip.
Treats each grid cell of the price tensor as constant — the level set
{slice_(u_a, u_b) = p_obs} is the boundary between cells with values
above/below p_obs. Crossings are placed at edge midpoints.

This is the SIMPLEST possible discretisation (no interpolation along
edges, no PCHIP). The Φ map becomes a step function of P, but the
positions of the contour crossings are still well-defined as long as
the slice has finitely many sign changes.

Reuses rezn_het primitives: _f0, _f1, _demand_crra, _clearing_residual,
_clear_price, _as_vec3, _nolearning_price, _logit, _logistic.
"""
from __future__ import annotations
import numpy as np
from numba import njit
import rezn_het as rh


# ---------- Order-0 contour sum -------------------------------------

@njit(cache=True, fastmath=True)
def _contour_sum_o0(slice_, u, tau_A, tau_B, p_obs):
    """Edge-midpoint contour integral on a G×G slice. Order-0."""
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0
    # Horizontal edges: between (a, b) and (a, b+1) — vary u_b
    for a in range(G):
        ua = u[a]
        for b in range(G - 1):
            v1 = slice_[a, b] - p_obs
            v2 = slice_[a, b + 1] - p_obs
            prod = v1 * v2
            if prod < 0.0:
                ub = 0.5 * (u[b] + u[b + 1])
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif v1 == 0.0 and v2 != 0.0:
                ub = u[b]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif b == G - 2 and v2 == 0.0:
                ub = u[b + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
    # Vertical edges: between (a, b) and (a+1, b) — vary u_a
    for b in range(G):
        ub = u[b]
        for a in range(G - 1):
            v1 = slice_[a, b] - p_obs
            v2 = slice_[a + 1, b] - p_obs
            prod = v1 * v2
            if prod < 0.0:
                ua = 0.5 * (u[a] + u[a + 1])
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif v1 == 0.0 and v2 != 0.0:
                ua = u[a]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif a == G - 2 and v2 == 0.0:
                ua = u[a + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def _agent_posterior_o0(ag, i, j, l, p_obs, Pg, u, taus):
    """taus = (τ_1, τ_2, τ_3). Agent `ag` (0-indexed). Order-0 contour."""
    G = u.shape[0]
    if ag == 0:
        u_own = u[i]; tau_own = taus[0]
        slice_ = Pg[i, :, :]
        tau_A = taus[1]; tau_B = taus[2]
    elif ag == 1:
        u_own = u[j]; tau_own = taus[1]
        slice_ = Pg[:, j, :]
        tau_A = taus[0]; tau_B = taus[2]
    else:
        u_own = u[l]; tau_own = taus[2]
        slice_ = Pg[:, :, l]
        tau_A = taus[0]; tau_B = taus[1]
    A0, A1 = _contour_sum_o0(slice_, u, tau_A, tau_B, p_obs)
    f0_own = rh._f0(u_own, tau_own)
    f1_own = rh._f1(u_own, tau_own)
    num = f1_own * A1
    den = f0_own * A0 + num
    if den <= 0.0:
        return 0.5
    return num / den


@njit(cache=True)
def _posteriors_at_o0(i, j, l, p_obs, Pg, u, taus):
    m0 = _agent_posterior_o0(0, i, j, l, p_obs, Pg, u, taus)
    m1 = _agent_posterior_o0(1, i, j, l, p_obs, Pg, u, taus)
    m2 = _agent_posterior_o0(2, i, j, l, p_obs, Pg, u, taus)
    return m0, m1, m2


@njit(cache=True)
def _phi_map_o0(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at_o0(i, j, l, p_cur, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                Pnew[i, j, l] = rh._clear_price(mus, gammas, Ws)
    return Pnew


@njit(cache=True)
def _residual_array_o0(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at_o0(i, j, l, p, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                F[i, j, l] = rh._clearing_residual(mus, p, gammas, Ws)
    return F
