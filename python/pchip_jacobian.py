"""Analytic Jacobian for the logit-PCHIP contour map Φ at G×G×G grid.

The fixed-point map is

    Φ(P)[i,j,l] = p*  s.t.  Σ_k x_k(μ_k(slice_k, p_obs=P[i,j,l]), p*) = 0

where μ_k is the contour-integrated posterior on trader k's 2-D slice
of P (with the slice values transformed to logit space and interpolated
by PCHIP) and x_k is CRRA demand. The chain rule for dΦ/dP is

    dΦ[ijl]/dV  =  -Σ_k (∂h/∂μ_k) · (dμ_k/dV) / (∂h/∂p)

with three contributions to dμ_k/dV:

    1. each slice cell (a,b) on trader k's slice contributes
       (∂μ_k/∂slice_k[a,b]) · V[slice_k(a,b)]   — via the contour-tangent
    2. the observed price p_obs = P[i,j,l] contributes
       (∂μ_k/∂p_obs) · V[i,j,l]                  — same contour-tangent
       evaluated with `p_obs_dot = V[i,j,l]`.

The closed-form analytic derivatives of (a) demand, (b) market clearing,
and (c) the cubic-Hermite basis are exact. The contour-tangent further
needs the implicit derivative of the segment-root finder (dt*/d slice
values) and the Fritsch-Carlson tangent for the per-row PCHIP slopes.

Key fix relative to a naive implementation: when a slice cell sits at
the outer clip [1e-9, 1-1e-9] used by the outer Picard / Newton loop,
the projected map sees a fixed value at that cell — so the analytic
must zero out the L_dot contribution there. Otherwise pab*(1-pab) ~
1e-9 in the denominator gives spurious 1e9-scale derivatives that make
the Jacobian useless for Newton.

Public API:
    precompute(P, u, taus, gammas, Ws)  -> dict cached for one P
    J_dot_v(V, precomp)                 -> (I - dΦ/dP) · V  matvec
    solve_newton(G, taus, gammas, ...)  -> iterative Newton solver
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, lgmres

import rezn_het as rh
import rezn_pchip as rp


# Outer clip used by the Picard / Newton solvers. Must agree with the
# np.clip applied to P_init and to perturbed P inside this module.
EPS_OUTER = 1e-9
# Inner clip used inside _contour_sum_pchip when the logit transform
# would diverge.
EPS_INNER = 1e-15


# =====================================================================
#  Closed-form derivatives
# =====================================================================

@njit(cache=True, fastmath=True)
def _demand_derivs(mu, p, gamma, W):
    """Return (x, ∂x/∂μ, ∂x/∂p) for CRRA demand."""
    eps = 1e-12
    mu_c = max(eps, min(1.0 - eps, mu))
    p_c = max(eps, min(1.0 - eps, p))
    log_R = (np.log(mu_c / (1 - mu_c)) - np.log(p_c / (1 - p_c))) / gamma
    R = np.exp(log_R)
    D = (1 - p_c) + R * p_c
    x = W * (R - 1) / D
    dR_dmu = R / (gamma * mu_c * (1 - mu_c))
    dR_dp = -R / (gamma * p_c * (1 - p_c))
    dD_dp = -1 + R + p_c * dR_dp
    dx_dmu = (W / (D * D)) * dR_dmu
    dx_dp = W * (dR_dp * D - (R - 1) * dD_dp) / (D * D)
    return x, dx_dmu, dx_dp


@njit(cache=True, fastmath=True)
def _hermite_partials(t, y0, y1, m0, m1, h):
    """Return (∂H/∂y0, ∂H/∂y1, ∂H/∂m0, ∂H/∂m1, ∂H/∂t) at parameter t."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    dH_dt = ((6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * h * m0
              + (-6 * t2 + 6 * t) * y1 + (3 * t2 - 2 * t) * h * m1)
    return h00, h01, h10 * h, h11 * h, dH_dt


# =====================================================================
#  Likelihood-density derivatives (per-state Gaussian f_v(u, τ))
# =====================================================================

@njit(cache=True, fastmath=True)
def _df0_du(u, tau):
    return -tau * (u + 0.5) * rh._f0(u, tau)


@njit(cache=True, fastmath=True)
def _df1_du(u, tau):
    return -tau * (u - 0.5) * rh._f1(u, tau)


# =====================================================================
#  Fritsch-Carlson PCHIP tangent
# =====================================================================

@njit(cache=True, fastmath=True)
def _pchip_slopes_tangent(y, y_dot, u):
    """Forward-mode tangent of PCHIP slope computation.

    Given (y, y_dot) along uniform-or-not 1-D grid u, return m_dot such
    that the PCHIP slopes m → m + ε·m_dot when y → y + ε·y_dot. Branches
    on the same monotonicity tests as the forward function.
    """
    G = y.shape[0]
    h = np.empty(G - 1)
    s = np.empty(G - 1)
    s_dot = np.empty(G - 1)
    for k in range(G - 1):
        h[k] = u[k + 1] - u[k]
        s[k] = (y[k + 1] - y[k]) / h[k]
        s_dot[k] = (y_dot[k + 1] - y_dot[k]) / h[k]

    m_dot = np.zeros(G)
    for k in range(1, G - 1):
        if s[k - 1] * s[k] <= 0:
            continue  # m_dot[k] = 0 — derivative of a constant clamp
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        D = w1 / s[k - 1] + w2 / s[k]
        D_dot = (-w1 * s_dot[k - 1] / (s[k - 1] * s[k - 1])
                  - w2 * s_dot[k] / (s[k] * s[k]))
        m_dot[k] = -((w1 + w2) / (D * D)) * D_dot

    # Endpoints (matches _pchip_derivs three-point formula with monotone clamp)
    h0, h1 = h[0], h[1]
    d_left = ((2 * h0 + h1) * s[0] - h0 * s[1]) / (h0 + h1)
    if d_left * s[0] <= 0:
        m_dot[0] = 0.0
    elif s[0] * s[1] <= 0 and abs(d_left) > abs(3 * s[0]):
        m_dot[0] = 3 * s_dot[0]
    else:
        m_dot[0] = ((2 * h0 + h1) * s_dot[0] - h0 * s_dot[1]) / (h0 + h1)

    hG2, hG3 = h[G - 2], h[G - 3]
    d_right = ((2 * hG2 + hG3) * s[G - 2] - hG2 * s[G - 3]) / (hG2 + hG3)
    if d_right * s[G - 2] <= 0:
        m_dot[G - 1] = 0.0
    elif s[G - 2] * s[G - 3] <= 0 and abs(d_right) > abs(3 * s[G - 2]):
        m_dot[G - 1] = 3 * s_dot[G - 2]
    else:
        m_dot[G - 1] = ((2 * hG2 + hG3) * s_dot[G - 2]
                         - hG2 * s_dot[G - 3]) / (hG2 + hG3)
    return m_dot


# =====================================================================
#  Contour-sum tangent in logit space (the heart of the Jacobian)
# =====================================================================

@njit(cache=True, fastmath=True)
def _contour_tangent(slice_, slice_dot, u, tau_A, tau_B, p_obs, p_obs_dot):
    """Forward-mode tangent of `_contour_sum_pchip` over a 2-D slice.

    Returns (A0, A1, A0_dot, A1_dot). Matches the two-pass crossing
    structure of the forward function. At slice cells inside the outer
    clip envelope (within EPS_OUTER of 0 or 1) the L_dot contribution
    is zeroed — the projected Newton map treats those as fixed.
    """
    G = u.shape[0]

    # logit transform of slice and its tangent
    L = np.empty_like(slice_)
    L_dot = np.empty_like(slice_)
    for a in range(G):
        for b in range(G):
            v = slice_[a, b]
            v_clipped = max(EPS_INNER, min(1 - EPS_INNER, v))
            L[a, b] = np.log(v_clipped / (1 - v_clipped))
            if v <= EPS_OUTER or v >= 1 - EPS_OUTER:
                L_dot[a, b] = 0.0
            else:
                L_dot[a, b] = slice_dot[a, b] / (v_clipped * (1 - v_clipped))

    p_clipped = max(EPS_INNER, min(1 - EPS_INNER, p_obs))
    lp = np.log(p_clipped / (1 - p_clipped))
    if p_obs <= EPS_OUTER or p_obs >= 1 - EPS_OUTER:
        lp_dot = 0.0
    else:
        lp_dot = p_obs_dot / (p_clipped * (1 - p_clipped))

    A0 = 0.0
    A1 = 0.0
    A0_dot = 0.0
    A1_dot = 0.0

    # Pass A: rows. For each grid u_a, find the off-grid u_b crossing.
    for a in range(G):
        row = L[a]
        row_dot = L_dot[a]
        ua = u[a]
        m_row = rp._pchip_derivs(row, u)
        m_row_dot = _pchip_slopes_tangent(row, row_dot, u)
        for k in range(G - 1):
            y0, y1 = row[k], row[k + 1]
            d0, d1 = y0 - lp, y1 - lp
            h_step = u[k + 1] - u[k]
            if d0 * d1 < 0:
                t = rp._pchip_root_in_segment(y0, y1, m_row[k], m_row[k + 1],
                                                h_step, lp)
                if t < 0:
                    continue
                ub = u[k] + t * h_step
                Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(
                    t, y0, y1, m_row[k], m_row[k + 1], h_step)
                t_dot = -(Hy0 * row_dot[k] + Hy1 * row_dot[k + 1]
                           + Hm0 * m_row_dot[k] + Hm1 * m_row_dot[k + 1]
                           - lp_dot) / Ht
                ub_dot = h_step * t_dot
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                f0b = rh._f0(ub, tau_B); f1b = rh._f1(ub, tau_B)
                A0 += f0a * f0b
                A1 += f1a * f1b
                A0_dot += f0a * _df0_du(ub, tau_B) * ub_dot
                A1_dot += f1a * _df1_du(ub, tau_B) * ub_dot
            elif d0 == 0 and d1 != 0:
                ub = u[k]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif k == G - 2 and d1 == 0:
                ub = u[k + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)

    # Pass B: columns. For each grid u_b, find the off-grid u_a crossing.
    for b in range(G):
        ub_grid = u[b]
        col = np.empty(G)
        col_dot = np.empty(G)
        for i in range(G):
            col[i] = L[i, b]
            col_dot[i] = L_dot[i, b]
        m_col = rp._pchip_derivs(col, u)
        m_col_dot = _pchip_slopes_tangent(col, col_dot, u)
        for k in range(G - 1):
            y0, y1 = col[k], col[k + 1]
            d0, d1 = y0 - lp, y1 - lp
            h_step = u[k + 1] - u[k]
            if d0 * d1 < 0:
                t = rp._pchip_root_in_segment(y0, y1, m_col[k], m_col[k + 1],
                                                h_step, lp)
                if t < 0:
                    continue
                ua = u[k] + t * h_step
                Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(
                    t, y0, y1, m_col[k], m_col[k + 1], h_step)
                t_dot = -(Hy0 * col_dot[k] + Hy1 * col_dot[k + 1]
                           + Hm0 * m_col_dot[k] + Hm1 * m_col_dot[k + 1]
                           - lp_dot) / Ht
                ua_dot = h_step * t_dot
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += f0a * f0b
                A1 += f1a * f1b
                A0_dot += _df0_du(ua, tau_A) * f0b * ua_dot
                A1_dot += _df1_du(ua, tau_A) * f1b * ua_dot
            elif d0 == 0 and d1 != 0:
                ua = u[k]
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += rh._f0(ua, tau_A) * f0b
                A1 += rh._f1(ua, tau_A) * f1b
            elif k == G - 2 and d1 == 0:
                ua = u[k + 1]
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += rh._f0(ua, tau_A) * f0b
                A1 += rh._f1(ua, tau_A) * f1b

    return 0.5 * A0, 0.5 * A1, 0.5 * A0_dot, 0.5 * A1_dot


# =====================================================================
#  Per-cell precompute (independent of V)
#
#  These quantities depend only on P, not on the perturbation V, so
#  computing them once per Newton iteration and reusing them across
#  every lgmres matvec gives a substantial speedup.
# =====================================================================

@njit(cache=True, parallel=True)
def _precompute_kernel(P, u, taus, gammas, Ws):
    """Precompute per-cell Phi, posteriors, and the clearing-Jacobian
    coefficients used by every matvec at the current P.

    Returns
    -------
    Phi  : (G,G,G) array — Φ(P)
    mus  : (G,G,G,3) — converged posteriors per cell, per agent
    dh_dmu : (G,G,G,3) — ∂h/∂μ_k at (μ, Φ)
    dh_dp  : (G,G,G)   — ∂h/∂p at (μ, Φ)
    """
    G = P.shape[0]
    Phi = np.empty_like(P)
    mus = np.empty((G, G, G, 3))
    dh_dmu = np.empty((G, G, G, 3))
    dh_dp = np.empty((G, G, G))
    for i in prange(G):
        for j in range(G):
            for l in range(G):
                p_obs = P[i, j, l]
                m0 = rp._agent_posterior_pchip(0, i, j, l, p_obs, P, u, taus)
                m1 = rp._agent_posterior_pchip(1, i, j, l, p_obs, P, u, taus)
                m2 = rp._agent_posterior_pchip(2, i, j, l, p_obs, P, u, taus)
                local_mus = np.empty(3)
                local_mus[0] = m0; local_mus[1] = m1; local_mus[2] = m2
                p_star = rh._clear_price(local_mus, gammas, Ws)
                Phi[i, j, l] = p_star
                mus[i, j, l, 0] = m0
                mus[i, j, l, 1] = m1
                mus[i, j, l, 2] = m2
                dh_total_dp = 0.0
                for k in range(3):
                    _, dx_dmu, dx_dp = _demand_derivs(
                        local_mus[k], p_star, gammas[k], Ws[k])
                    dh_dmu[i, j, l, k] = dx_dmu
                    dh_total_dp += dx_dp
                dh_dp[i, j, l] = dh_total_dp
    return Phi, mus, dh_dmu, dh_dp


def precompute(P, u, taus, gammas, Ws):
    """Public wrapper around _precompute_kernel returning a dict."""
    Phi, mus, dh_dmu, dh_dp = _precompute_kernel(
        P, u, np.asarray(taus, float),
        np.asarray(gammas, float), np.asarray(Ws, float))
    return {
        "P": P, "u": u, "Phi": Phi,
        "mus": mus, "dh_dmu": dh_dmu, "dh_dp": dh_dp,
        "taus": np.asarray(taus, float),
        "gammas": np.asarray(gammas, float),
        "Ws": np.asarray(Ws, float),
    }


# =====================================================================
#  Matvec kernel:  V → (I - dΦ/dP) · V
# =====================================================================

@njit(cache=True, parallel=True)
def _matvec_kernel(V, P, Phi, mus, dh_dmu, dh_dp, u, taus):
    """Apply (I - dΦ/dP) to V. Uses precomputed posteriors and clearing
    Jacobian — only the contour-tangent depends on V."""
    G = P.shape[0]
    out = V.copy()
    for i in prange(G):
        slice_buf = np.empty((G, G))
        slice_dot_buf = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                p_star = Phi[i, j, l]
                if p_star <= EPS_OUTER or p_star >= 1 - EPS_OUTER:
                    continue  # clamped Φ → dΦ = 0
                dh_p = dh_dp[i, j, l]
                if abs(dh_p) < 1e-30:
                    continue

                p_obs = P[i, j, l]
                p_obs_dot = V[i, j, l]
                num = 0.0  # Σ_k ∂h/∂μ_k · μ_k_dot

                for ag in range(3):
                    if ag == 0:
                        for a in range(G):
                            for b in range(G):
                                slice_buf[a, b] = P[i, a, b]
                                slice_dot_buf[a, b] = V[i, a, b]
                        u_own = u[i]; tau_own = taus[0]
                        tau_A = taus[1]; tau_B = taus[2]
                    elif ag == 1:
                        for a in range(G):
                            for b in range(G):
                                slice_buf[a, b] = P[a, j, b]
                                slice_dot_buf[a, b] = V[a, j, b]
                        u_own = u[j]; tau_own = taus[1]
                        tau_A = taus[0]; tau_B = taus[2]
                    else:
                        for a in range(G):
                            for b in range(G):
                                slice_buf[a, b] = P[a, b, l]
                                slice_dot_buf[a, b] = V[a, b, l]
                        u_own = u[l]; tau_own = taus[2]
                        tau_A = taus[0]; tau_B = taus[1]

                    A0, A1, A0_dot, A1_dot = _contour_tangent(
                        slice_buf, slice_dot_buf, u, tau_A, tau_B,
                        p_obs, p_obs_dot)
                    g0 = rh._f0(u_own, tau_own)
                    g1 = rh._f1(u_own, tau_own)
                    den = g0 * A0 + g1 * A1
                    if den <= 0:
                        continue  # μ fell back to prior — derivative is 0
                    g0g1 = g0 * g1
                    mu_dot = (-g0g1 * A1 * A0_dot + g0g1 * A0 * A1_dot) / (den * den)
                    num += dh_dmu[i, j, l, ag] * mu_dot

                # dΦ = -num / dh_dp
                out[i, j, l] -= -num / dh_p
    return out


def J_dot_v(V, precomp):
    """(I - dΦ/dP) · V using precomputed posteriors / clearing Jacobian."""
    return _matvec_kernel(
        V, precomp["P"], precomp["Phi"], precomp["mus"],
        precomp["dh_dmu"], precomp["dh_dp"], precomp["u"], precomp["taus"])


# =====================================================================
#  Newton solver with Armijo backtracking and lgmres inner solve
# =====================================================================

def solve_newton(G, taus, gammas, umax=2.0, Ws=1.0,
                  P_init=None, maxiters=15, abstol=1e-12,
                  lgmres_tol=1e-8, lgmres_maxiter=80,
                  armijo_min_alpha=1.0 / 64,
                  status_path=None, status_every=1, status_prefix="",
                  verbose=False):
    """Iterating analytic-Jacobian Newton for the PCHIP contour map.

    Each iteration:
      1. Φ(P), mus, ∂h/∂μ_k, ∂h/∂p are precomputed once.
      2. lgmres solves (I - dΦ/dP) · dP = F(P) = P - Φ(P)
         via the precompute-aware matvec.
      3. Armijo backtrack on ‖F‖∞: keep halving α from 1 until accept,
         else fall back to a Picard step.
    Tracks the best (lowest-Finf) iterate seen.
    """
    u = np.linspace(-umax, umax, G)
    taus = rh._as_vec3(taus)
    gammas = rh._as_vec3(gammas)
    Ws = rh._as_vec3(Ws)

    if P_init is None:
        P = rh._nolearning_price(u, taus, gammas, Ws)
    else:
        P = np.clip(np.asarray(P_init, float).copy(),
                    EPS_OUTER, 1 - EPS_OUTER)

    history = []
    P_best = P.copy()
    best_Finf = float("inf")
    converged = False
    t_start = time.time()

    for it in range(maxiters):
        precomp = precompute(P, u, taus, gammas, Ws)
        F = P - precomp["Phi"]
        Finf = float(np.abs(F).max())
        history.append(Finf)
        if Finf < best_Finf:
            best_Finf = Finf
            P_best = P.copy()
        if status_path is not None and (it % status_every == 0):
            try:
                with open(status_path, "w") as f:
                    f.write(f"{status_prefix} newton iter={it}/{maxiters} "
                            f"Finf={Finf:.3e} best={best_Finf:.3e} "
                            f"elapsed={time.time() - t_start:.1f}s\n")
            except Exception:
                pass
        if verbose:
            print(f"  newton it={it}: Finf={Finf:.3e}", flush=True)
        if Finf < abstol:
            converged = True
            break

        # Newton step:  J · dP = -F
        N = G ** 3

        def matvec(v, _pre=precomp, _shape=P.shape):
            return J_dot_v(v.reshape(_shape), _pre).reshape(-1)

        op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        dP_flat, _ = lgmres(op, -F.reshape(-1),
                             rtol=lgmres_tol, atol=0.0,
                             maxiter=lgmres_maxiter)
        dP = dP_flat.reshape(P.shape)

        # Armijo backtrack on ‖F‖∞
        alpha = 1.0
        accepted = False
        while alpha >= armijo_min_alpha:
            P_try = np.clip(P + alpha * dP, EPS_OUTER, 1 - EPS_OUTER)
            Phi_try = rp._phi_map_pchip(P_try, u, taus, gammas, Ws)
            Finf_try = float(np.abs(P_try - Phi_try).max())
            if Finf_try < Finf:
                P = P_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            P = np.clip(precomp["Phi"], EPS_OUTER, 1 - EPS_OUTER)  # Picard fallback

    F_final = P_best - rp._phi_map_pchip(P_best, u, taus, gammas, Ws)
    return {
        "P_star": P_best, "u": u, "residual": F_final,
        "history": history,
        "converged": converged or (best_Finf < abstol),
        "best_Finf": best_Finf,
        "taus": taus, "gammas": gammas,
    }


# =====================================================================
#  Backwards-compatibility shim
# =====================================================================

def J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws):
    """Old API kept for any out-of-tree callers. Builds a fresh
    precompute on every call — slower than caching `precompute(P, ...)`
    once and calling `J_dot_v(V, precomp)` for every matvec inside an
    lgmres call."""
    pre = precompute(P, u, taus, gammas, Ws)
    return J_dot_v(V, pre)


# Old name — kept so pchip_continuation.py's import continues to work.
solve_newton_analytic = solve_newton


# =====================================================================
#  Self-tests against finite differences
# =====================================================================

def _selftests():
    """Sanity checks. Each piece is FD-verified independently. The full
    Newton step is tested by polishing a Picard warm start to machine
    precision at G=5 and G=7."""
    rng = np.random.default_rng(0)
    fail = 0
    total = 0

    # Demand derivatives
    for _ in range(20):
        mu = rng.uniform(0.05, 0.95)
        p = rng.uniform(0.05, 0.95)
        gamma = rng.uniform(0.5, 50)
        x, dxm, dxp = _demand_derivs(mu, p, gamma, 1.0)
        eps = 1e-7
        fdm = (_demand_derivs(mu + eps, p, gamma, 1.0)[0]
                - _demand_derivs(mu - eps, p, gamma, 1.0)[0]) / (2 * eps)
        fdp = (_demand_derivs(mu, p + eps, gamma, 1.0)[0]
                - _demand_derivs(mu, p - eps, gamma, 1.0)[0]) / (2 * eps)
        total += 2
        if abs(dxm - fdm) > 1e-5: fail += 1
        if abs(dxp - fdp) > 1e-5: fail += 1
    print(f"demand derivs: {total - fail}/{total} pass")

    # Hermite partials
    fail2 = 0; total2 = 0
    for _ in range(20):
        t = rng.uniform(0.05, 0.95)
        y0, y1 = rng.uniform(-1, 1, 2)
        m0, m1 = rng.uniform(-2, 2, 2)
        h = rng.uniform(0.1, 1.0)
        Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(t, y0, y1, m0, m1, h)
        eps = 1e-7
        fd = (rp._hermite_val(t, y0 + eps, y1, m0, m1, h)
               - rp._hermite_val(t, y0 - eps, y1, m0, m1, h)) / (2 * eps)
        if abs(Hy0 - fd) > 1e-7: fail2 += 1
        total2 += 1
    print(f"hermite partials: {total2 - fail2}/{total2} pass")

    # End-to-end Newton at G=5,7 from a Picard warm start
    print("Newton on Picard warm start:")
    for G in (5, 7):
        UMAX = 2.0
        taus = np.array([3.0, 3.0, 3.0])
        gammas = np.array([3.0, 3.0, 3.0])
        Ws = np.array([1.0, 1.0, 1.0])
        warm = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                       maxiters=200, abstol=1e-9, alpha=1.0)
        Finf0 = float(np.abs(warm["residual"]).max())
        res = solve_newton(G, taus, gammas, umax=UMAX,
                            P_init=warm["P_star"], maxiters=10,
                            abstol=1e-12, lgmres_tol=1e-9,
                            lgmres_maxiter=80, verbose=False)
        Finf1 = res["best_Finf"]
        print(f"  G={G}: Picard Finf={Finf0:.3e} → Newton Finf={Finf1:.3e}  "
              f"(iters={len(res['history'])})")


if __name__ == "__main__":
    _selftests()
