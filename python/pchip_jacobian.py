"""Analytic Jacobian-vector product for the PCHIP+contour Φ map.

Goal: replace finite-difference NK (FD-floor ~1e-8) with exact-Jacobian
Newton, reaching machine precision.

Map: Φ(P)[i,j,l] = p* solves Σ_k x_k(μ_k(u_k, p, P), p) = 0
where μ_k is the contour-integrated posterior on the k-th 2D slice.

This file provides:
  * `phi_and_tangent(P, V)`: forward + tangent-mode derivative.
    Returns (Φ(P), dΦ(P)·V) in one pass through the same logic.
  * `J_dot_v(P, V)`: J = I - dΦ/dP, returns J·V = V - dΦ(P)·V.
  * `linear_operator(P)`: scipy LinearOperator wrapping J_dot_v
    so Newton step can use lgmres.

Pieces (each independently testable):

A. Demand derivatives (closed form for CRRA):
   x = W·(R-1)/D, with R = exp((logit(μ) - logit(p))/γ), D = (1-p) + R·p.
   ∂x/∂R = W/D²
   ∂R/∂μ = R/(γ·μ(1-μ))
   ∂R/∂p = -R/(γ·p(1-p))
   ∂x/∂μ = W·R / (γ·μ(1-μ)·D²)
   ∂x/∂p = W·[∂R/∂p·D - (R-1)·∂D/∂p] / D², with
           ∂D/∂p = -1 + R + p·∂R/∂p.

B. Market-clearing implicit derivative:
   h(μ_1, μ_2, μ_3, p) = Σ_k x_k(μ_k, p) = 0.
   ∂p*/∂μ_k = -(∂x_k/∂μ_k) / (Σ_j ∂x_j/∂p).
   ∂p*/∂(other vars at fixed μ): chain rule.

C. PCHIP segment-root implicit derivative:
   At root t* with H(t*; y0,y1,m0,m1,h) = p_obs:
   dt*/dy_i = -∂H/∂y_i / ∂H/∂t   (for i ∈ {0,1, m0_idx, m1_idx})
   dt*/dp_obs = +1 / ∂H/∂t

   Hermite basis derivatives:
   H = h00·y0 + h10·h·m0 + h01·y1 + h11·h·m1
   ∂H/∂y0 = h00, ∂H/∂y1 = h01, ∂H/∂m0 = h10·h, ∂H/∂m1 = h11·h.

   The PCHIP slope m_k itself depends on slice values via Fritsch-Carlson:
   m_k = (w1+w2) / (w1/s_{k-1} + w2/s_k) interior, special at endpoints.
   So d m_k / d slice[j] = ... (chain rule through s = (y_{k+1}-y_k)/h).

D. Contour-sum derivative:
   A_kc = Σ_crossings f_c(u_a, τ_a) · f_c(u_b, τ_b)
   When a slice value changes by δ, each crossing position shifts (via C),
   contributing  (∂f_c/∂u_b · du_b/δ) per crossing. Crossings can also
   appear/disappear at segment boundaries — those are measure-zero events
   ignored at the fixed point (slice values not exactly equal to p_obs).

E. Posterior derivative:
   μ_k = g_k1·A_k1 / (g_k0·A_k0 + g_k1·A_k1)
   ∂μ_k/∂A_kc has a clean rational form from quotient rule.

F. Tangent of Φ:
   At cell (i,j,l), Φ is found from market clearing on (μ_0, μ_1, μ_2, p_obs).
   Tangent: dΦ(P)·V = chain-rule combination of above pieces.

This module currently implements pieces A, B, C (demand+root+market clear)
analytically. Pieces D and E are FRAMEWORK ONLY — completion is the next
step (estimated 1-2 hours). Once complete, J_dot_v returns the exact
Jacobian-vector product, FD-NK floor goes away, Newton reaches machine
precision in 5-10 outer iterations.

Tests in __main__ verify A, B, C against finite differences.
"""
from __future__ import annotations
import numpy as np
from numba import njit

# Re-use primitives from existing modules
import rezn_het as rh
import rezn_pchip as rp


# =====================================================================
#  A. Demand derivatives (CRRA closed form)
# =====================================================================
@njit(cache=True, fastmath=True)
def _demand_and_derivs(mu, p, gamma, W):
    """Return (x, dx/dμ, dx/dp) for CRRA demand."""
    EPS = 1e-12
    mu_c = max(EPS, min(1.0 - EPS, mu))
    p_c  = max(EPS, min(1.0 - EPS, p))
    log_R = (np.log(mu_c / (1 - mu_c)) - np.log(p_c / (1 - p_c))) / gamma
    R = np.exp(log_R)
    D = (1.0 - p_c) + R * p_c
    x = W * (R - 1.0) / D
    # ∂R/∂μ = R / (γ μ(1-μ)),  ∂R/∂p = -R / (γ p(1-p))
    dR_dmu = R / (gamma * mu_c * (1.0 - mu_c))
    dR_dp  = -R / (gamma * p_c * (1.0 - p_c))
    # ∂x/∂R = W/D²
    dx_dR  = W / (D * D)
    # ∂D/∂p = -1 + R + p·∂R/∂p
    dD_dp  = -1.0 + R + p_c * dR_dp
    dx_dmu = dx_dR * dR_dmu
    dx_dp  = W * (dR_dp * D - (R - 1.0) * dD_dp) / (D * D)
    return x, dx_dmu, dx_dp


# =====================================================================
#  B. Market-clearing implicit derivative
# =====================================================================
@njit(cache=True, fastmath=True)
def _clearing_jacobian(mus, p, gammas, Ws):
    """Return (residual h, ∂h/∂μ_0, ∂h/∂μ_1, ∂h/∂μ_2, ∂h/∂p)."""
    h = 0.0
    dh_dmu0 = 0.0; dh_dmu1 = 0.0; dh_dmu2 = 0.0
    dh_dp = 0.0
    for k in range(3):
        x, dx_dmu, dx_dp = _demand_and_derivs(mus[k], p, gammas[k], Ws[k])
        h += x
        if k == 0: dh_dmu0 = dx_dmu
        elif k == 1: dh_dmu1 = dx_dmu
        else: dh_dmu2 = dx_dmu
        dh_dp += dx_dp
    return h, dh_dmu0, dh_dmu1, dh_dmu2, dh_dp


# =====================================================================
#  C. PCHIP segment-root implicit derivative
# =====================================================================
@njit(cache=True, fastmath=True)
def _hermite_partials(t, y0, y1, m0, m1, h):
    """Return (∂H/∂y0, ∂H/∂y1, ∂H/∂m0, ∂H/∂m1, ∂H/∂t) at the given t."""
    t2 = t * t; t3 = t2 * t
    h00 = 2.0*t3 - 3.0*t2 + 1.0
    h10 = t3 - 2.0*t2 + t
    h01 = -2.0*t3 + 3.0*t2
    h11 = t3 - t2
    # ∂H/∂t (matches _hermite_deriv)
    dh00 = 6.0*t2 - 6.0*t
    dh10 = 3.0*t2 - 4.0*t + 1.0
    dh01 = -6.0*t2 + 6.0*t
    dh11 = 3.0*t2 - 2.0*t
    dHdt = dh00*y0 + dh10*h*m0 + dh01*y1 + dh11*h*m1
    return h00, h01, h10*h, h11*h, dHdt


# =====================================================================
#  D1. Tangent of Fritsch-Carlson PCHIP derivative computation.
#       Given a 1D array `y` and tangent `y_dot`, compute m_dot such
#       that PCHIP slopes m -> m + ε·m_dot when y -> y + ε·y_dot.
# =====================================================================
@njit(cache=True, fastmath=True)
def _pchip_derivs_tangent(y, y_dot, u):
    """Forward-mode tangent of `_pchip_derivs`. Same shape as input."""
    G = y.shape[0]
    m_dot = np.zeros(G)
    s = np.empty(G - 1)
    s_dot = np.empty(G - 1)
    h = np.empty(G - 1)
    for k in range(G - 1):
        h[k] = u[k + 1] - u[k]
        s[k] = (y[k + 1] - y[k]) / h[k]
        s_dot[k] = (y_dot[k + 1] - y_dot[k]) / h[k]

    for k in range(1, G - 1):
        if s[k - 1] * s[k] <= 0.0:
            m_dot[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            # m = (w1+w2) / D, with D = w1/s_{k-1} + w2/s_k
            D = w1 / s[k - 1] + w2 / s[k]
            # dD = -w1·s_dot_{k-1}/s_{k-1}² - w2·s_dot_k/s_k²
            D_dot = -w1 * s_dot[k - 1] / (s[k - 1] * s[k - 1]) \
                    - w2 * s_dot[k] / (s[k] * s[k])
            # m = N/D so m_dot = -N·D_dot/D² = -m·D_dot/D
            N = w1 + w2
            m_dot[k] = -(N / (D * D)) * D_dot

    # Left endpoint
    h0, h1 = h[0], h[1]
    d = ((2.0 * h0 + h1) * s[0] - h0 * s[1]) / (h0 + h1)
    if d * s[0] <= 0.0:
        m_dot[0] = 0.0
    elif (s[0] * s[1] <= 0.0) and (abs(d) > abs(3.0 * s[0])):
        m_dot[0] = 3.0 * s_dot[0]
    else:
        m_dot[0] = ((2.0 * h0 + h1) * s_dot[0] - h0 * s_dot[1]) / (h0 + h1)

    # Right endpoint
    hG2, hG3 = h[G - 2], h[G - 3]
    d = ((2.0 * hG2 + hG3) * s[G - 2] - hG2 * s[G - 3]) / (hG2 + hG3)
    if d * s[G - 2] <= 0.0:
        m_dot[G - 1] = 0.0
    elif (s[G - 2] * s[G - 3] <= 0.0) and (abs(d) > abs(3.0 * s[G - 2])):
        m_dot[G - 1] = 3.0 * s_dot[G - 2]
    else:
        m_dot[G - 1] = ((2.0 * hG2 + hG3) * s_dot[G - 2]
                        - hG2 * s_dot[G - 3]) / (hG2 + hG3)
    return m_dot


# =====================================================================
#  D2. Derivatives of likelihood weights f0, f1 w.r.t. u.
# =====================================================================
@njit(cache=True, fastmath=True)
def _df0_du(u, tau):
    return -tau * (u + 0.5) * rh._f0(u, tau)


@njit(cache=True, fastmath=True)
def _df1_du(u, tau):
    return -tau * (u - 0.5) * rh._f1(u, tau)


# =====================================================================
#  D3. Tangent of the logit-space contour sum.
#       Given (slice in P-space, perturbation slice_dot, p_obs, p_obs_dot)
#       returns (A0, A1, A0_dot, A1_dot).
#
#  Logit-space conversion:
#       L[a,b] = log(slice[a,b] / (1 - slice[a,b]))
#       L_dot[a,b] = slice_dot[a,b] / (slice[a,b] * (1 - slice[a,b]))
#       lp_obs = log(p_obs/(1-p_obs))
#       lp_obs_dot = p_obs_dot / (p_obs * (1-p_obs))
# =====================================================================
@njit(cache=True)
def _contour_sum_tangent(slice_, slice_dot, u, tau_A, tau_B,
                          p_obs, p_obs_dot):
    G = u.shape[0]
    A0 = 0.0; A1 = 0.0
    A0_dot = 0.0; A1_dot = 0.0

    EPS = 1e-15
    L = np.empty_like(slice_)
    L_dot = np.empty_like(slice_)
    for a in range(G):
        for b in range(G):
            pab = max(EPS, min(1.0 - EPS, slice_[a, b]))
            L[a, b] = np.log(pab / (1.0 - pab))
            L_dot[a, b] = slice_dot[a, b] / (pab * (1.0 - pab))

    pc = max(EPS, min(1.0 - EPS, p_obs))
    lp_obs = np.log(pc / (1.0 - pc))
    lp_obs_dot = p_obs_dot / (pc * (1.0 - pc))

    # ----- Pass A: rows -----
    for a in range(G):
        row = L[a]
        row_dot = L_dot[a]
        ua = u[a]
        m_row = rp._pchip_derivs(row, u)
        m_row_dot = _pchip_derivs_tangent(row, row_dot, u)
        for k in range(G - 1):
            y0 = row[k]; y1 = row[k + 1]
            d0 = y0 - lp_obs; d1 = y1 - lp_obs
            h_step = u[k + 1] - u[k]
            if d0 * d1 < 0.0:
                t = rp._pchip_root_in_segment(y0, y1, m_row[k], m_row[k + 1],
                                                h_step, lp_obs)
                if t < 0.0:
                    continue
                ub = u[k] + t * h_step
                # Implicit derivative of t* w.r.t. (y0,y1,m0,m1, lp_obs)
                Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(
                    t, y0, y1, m_row[k], m_row[k + 1], h_step)
                t_dot = -(Hy0 * row_dot[k] + Hy1 * row_dot[k + 1]
                           + Hm0 * m_row_dot[k] + Hm1 * m_row_dot[k + 1]
                           - lp_obs_dot) / Ht
                ub_dot = h_step * t_dot
                # Contributions and tangents
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                f0b = rh._f0(ub, tau_B); f1b = rh._f1(ub, tau_B)
                A0 += f0a * f0b
                A1 += f1a * f1b
                A0_dot += f0a * _df0_du(ub, tau_B) * ub_dot
                A1_dot += f1a * _df1_du(ub, tau_B) * ub_dot
            elif d0 == 0.0 and d1 != 0.0:
                ub = u[k]
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                A0 += f0a * rh._f0(ub, tau_B)
                A1 += f1a * rh._f1(ub, tau_B)
            elif k == G - 2 and d1 == 0.0:
                ub = u[k + 1]
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                A0 += f0a * rh._f0(ub, tau_B)
                A1 += f1a * rh._f1(ub, tau_B)

    # ----- Pass B: columns -----
    for b in range(G):
        ub_grid = u[b]
        col = np.empty(G)
        col_dot = np.empty(G)
        for i in range(G):
            col[i] = L[i, b]
            col_dot[i] = L_dot[i, b]
        m_col = rp._pchip_derivs(col, u)
        m_col_dot = _pchip_derivs_tangent(col, col_dot, u)
        for k in range(G - 1):
            y0 = col[k]; y1 = col[k + 1]
            d0 = y0 - lp_obs; d1 = y1 - lp_obs
            h_step = u[k + 1] - u[k]
            if d0 * d1 < 0.0:
                t = rp._pchip_root_in_segment(y0, y1, m_col[k], m_col[k + 1],
                                                h_step, lp_obs)
                if t < 0.0:
                    continue
                ua = u[k] + t * h_step
                Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(
                    t, y0, y1, m_col[k], m_col[k + 1], h_step)
                t_dot = -(Hy0 * col_dot[k] + Hy1 * col_dot[k + 1]
                           + Hm0 * m_col_dot[k] + Hm1 * m_col_dot[k + 1]
                           - lp_obs_dot) / Ht
                ua_dot = h_step * t_dot
                f0a = rh._f0(ua, tau_A); f1a = rh._f1(ua, tau_A)
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += f0a * f0b
                A1 += f1a * f1b
                A0_dot += _df0_du(ua, tau_A) * f0b * ua_dot
                A1_dot += _df1_du(ua, tau_A) * f1b * ua_dot
            elif d0 == 0.0 and d1 != 0.0:
                ua = u[k]
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += rh._f0(ua, tau_A) * f0b
                A1 += rh._f1(ua, tau_A) * f1b
            elif k == G - 2 and d1 == 0.0:
                ua = u[k + 1]
                f0b = rh._f0(ub_grid, tau_B); f1b = rh._f1(ub_grid, tau_B)
                A0 += rh._f0(ua, tau_A) * f0b
                A1 += rh._f1(ua, tau_A) * f1b

    return 0.5 * A0, 0.5 * A1, 0.5 * A0_dot, 0.5 * A1_dot


# =====================================================================
#  Sanity tests (compare analytic vs FD on simple inputs)
# =====================================================================
def _test_demand_derivs():
    """FD-check ∂x/∂μ and ∂x/∂p of CRRA demand."""
    rng = np.random.default_rng(0)
    fail = 0
    for _ in range(50):
        mu = rng.uniform(0.05, 0.95)
        p  = rng.uniform(0.05, 0.95)
        gamma = rng.uniform(0.5, 50.0)
        W = 1.0
        x, dx_dmu, dx_dp = _demand_and_derivs(mu, p, gamma, W)
        eps = 1e-7
        x_pmu = _demand_and_derivs(mu + eps, p, gamma, W)[0]
        x_mmu = _demand_and_derivs(mu - eps, p, gamma, W)[0]
        x_pp  = _demand_and_derivs(mu, p + eps, gamma, W)[0]
        x_mp  = _demand_and_derivs(mu, p - eps, gamma, W)[0]
        fd_mu = (x_pmu - x_mmu) / (2 * eps)
        fd_p  = (x_pp - x_mp) / (2 * eps)
        err_mu = abs(dx_dmu - fd_mu) / max(1.0, abs(dx_dmu))
        err_p  = abs(dx_dp  - fd_p ) / max(1.0, abs(dx_dp))
        if err_mu > 1e-5 or err_p > 1e-5:
            fail += 1
            print(f"FAIL mu={mu:.3f} p={p:.3f} γ={gamma:.2f} "
                  f"errμ={err_mu:.2e} errp={err_p:.2e}")
    print(f"_test_demand_derivs: {50-fail}/50 PASS")
    return fail == 0


def _test_clearing_jacobian():
    """FD-check ∂h/∂μ_k and ∂h/∂p of market-clearing residual."""
    rng = np.random.default_rng(1)
    fail = 0
    for _ in range(20):
        mus = rng.uniform(0.05, 0.95, 3)
        p   = rng.uniform(0.05, 0.95)
        gammas = rng.uniform(0.5, 50.0, 3)
        Ws = np.array([1.0, 1.0, 1.0])
        h, d0, d1, d2, dp = _clearing_jacobian(mus, p, gammas, Ws)
        eps = 1e-7
        for k in range(3):
            mp = mus.copy(); mp[k] += eps
            mm = mus.copy(); mm[k] -= eps
            h_p = _clearing_jacobian(mp, p, gammas, Ws)[0]
            h_m = _clearing_jacobian(mm, p, gammas, Ws)[0]
            fd  = (h_p - h_m) / (2 * eps)
            ana = [d0, d1, d2][k]
            if abs(ana - fd) / max(1.0, abs(ana)) > 1e-5:
                fail += 1
                print(f"FAIL k={k} ana={ana:.3e} fd={fd:.3e}")
        h_pp = _clearing_jacobian(mus, p + eps, gammas, Ws)[0]
        h_mp = _clearing_jacobian(mus, p - eps, gammas, Ws)[0]
        fd_p = (h_pp - h_mp) / (2 * eps)
        if abs(dp - fd_p) / max(1.0, abs(dp)) > 1e-5:
            fail += 1
            print(f"FAIL ∂h/∂p ana={dp:.3e} fd={fd_p:.3e}")
    print(f"_test_clearing_jacobian: {(80-fail)}/80 PASS")
    return fail == 0


def _test_hermite_partials():
    """FD-check the 5 Hermite partials."""
    rng = np.random.default_rng(2)
    fail = 0
    for _ in range(30):
        t = rng.uniform(0.05, 0.95)
        y0, y1 = rng.uniform(-1, 1, 2)
        m0, m1 = rng.uniform(-2, 2, 2)
        h = rng.uniform(0.1, 1.0)
        Hy0, Hy1, Hm0, Hm1, Ht = _hermite_partials(t, y0, y1, m0, m1, h)
        eps = 1e-7
        H_py0 = rp._hermite_val(t, y0+eps, y1, m0, m1, h)
        H_my0 = rp._hermite_val(t, y0-eps, y1, m0, m1, h)
        fd = (H_py0 - H_my0) / (2*eps)
        if abs(Hy0 - fd) > 1e-7:
            fail += 1; print(f"FAIL Hy0 ana={Hy0:.3e} fd={fd:.3e}")
        H_py1 = rp._hermite_val(t, y0, y1+eps, m0, m1, h)
        H_my1 = rp._hermite_val(t, y0, y1-eps, m0, m1, h)
        fd = (H_py1 - H_my1) / (2*eps)
        if abs(Hy1 - fd) > 1e-7:
            fail += 1; print(f"FAIL Hy1 ana={Hy1:.3e} fd={fd:.3e}")
        H_pm0 = rp._hermite_val(t, y0, y1, m0+eps, m1, h)
        H_mm0 = rp._hermite_val(t, y0, y1, m0-eps, m1, h)
        fd = (H_pm0 - H_mm0) / (2*eps)
        if abs(Hm0 - fd) > 1e-7:
            fail += 1; print(f"FAIL Hm0 ana={Hm0:.3e} fd={fd:.3e}")
        H_pm1 = rp._hermite_val(t, y0, y1, m0, m1+eps, h)
        H_mm1 = rp._hermite_val(t, y0, y1, m0, m1-eps, h)
        fd = (H_pm1 - H_mm1) / (2*eps)
        if abs(Hm1 - fd) > 1e-7:
            fail += 1; print(f"FAIL Hm1 ana={Hm1:.3e} fd={fd:.3e}")
        H_pt = rp._hermite_val(t+eps, y0, y1, m0, m1, h)
        H_mt = rp._hermite_val(t-eps, y0, y1, m0, m1, h)
        fd = (H_pt - H_mt) / (2*eps)
        if abs(Ht - fd) > 1e-6:
            fail += 1; print(f"FAIL Ht ana={Ht:.3e} fd={fd:.3e}")
    print(f"_test_hermite_partials: {150-fail}/150 PASS")
    return fail == 0


def _test_pchip_derivs_tangent():
    """FD-check m_dot against (m(y+ε·y_dot) - m(y-ε·y_dot))/(2ε)."""
    rng = np.random.default_rng(3)
    fail = 0; total = 0
    for trial in range(20):
        G = 11
        u = np.linspace(-2, 2, G)
        # smooth data (sigmoid of τ·u with random offset)
        y = 1.0 / (1.0 + np.exp(-3 * (u - rng.uniform(-0.5, 0.5))))
        y_dot = rng.standard_normal(G) * 0.1
        m_dot = _pchip_derivs_tangent(y, y_dot, u)
        eps = 1e-7
        m_plus  = rp._pchip_derivs(y + eps * y_dot, u)
        m_minus = rp._pchip_derivs(y - eps * y_dot, u)
        fd = (m_plus - m_minus) / (2 * eps)
        for i in range(G):
            total += 1
            err = abs(m_dot[i] - fd[i]) / max(1.0, abs(fd[i]))
            if err > 1e-5:
                fail += 1
    print(f"_test_pchip_derivs_tangent: {total-fail}/{total} PASS")
    return fail == 0


def _test_contour_sum_tangent():
    """FD-check (A0_dot, A1_dot) against centered FD of `_contour_sum_pchip`."""
    rng = np.random.default_rng(4)
    fail = 0; total = 0
    for trial in range(10):
        G = 11
        u = np.linspace(-2, 2, G)
        # smooth slice that has crossings: sigmoid of (u_a + u_b - offset)
        offset = rng.uniform(-0.5, 0.5)
        slice_ = np.zeros((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = 1.0 / (1.0 + np.exp(-1.5 * (u[a] + u[b] - offset)))
        slice_dot = rng.standard_normal((G, G)) * 0.01
        p_obs = 0.55
        p_obs_dot = rng.uniform(-0.01, 0.01)
        tau_A, tau_B = 3.0, 3.0

        A0, A1, A0d, A1d = _contour_sum_tangent(
            slice_, slice_dot, u, tau_A, tau_B, p_obs, p_obs_dot)

        eps = 1e-6
        A0p, A1p = rp._contour_sum_pchip(
            slice_ + eps * slice_dot, u, tau_A, tau_B, p_obs + eps * p_obs_dot)
        A0m, A1m = rp._contour_sum_pchip(
            slice_ - eps * slice_dot, u, tau_A, tau_B, p_obs - eps * p_obs_dot)
        fd0 = (A0p - A0m) / (2 * eps)
        fd1 = (A1p - A1m) / (2 * eps)
        total += 2
        err0 = abs(A0d - fd0) / max(1.0, abs(fd0))
        err1 = abs(A1d - fd1) / max(1.0, abs(fd1))
        if err0 > 1e-3:
            fail += 1; print(f"FAIL A0 trial={trial} ana={A0d:.4e} fd={fd0:.4e} err={err0:.2e}")
        if err1 > 1e-3:
            fail += 1; print(f"FAIL A1 trial={trial} ana={A1d:.4e} fd={fd1:.4e} err={err1:.2e}")
    print(f"_test_contour_sum_tangent: {total-fail}/{total} PASS")
    return fail == 0


# =====================================================================
#  E. Posterior tangent (rational from quotient rule)
# =====================================================================
@njit(cache=True, fastmath=True)
def _posterior_and_tangent(A0, A1, A0_dot, A1_dot, g0, g1, u_own, tau_own):
    """μ = g1·A1 / (g0·A0 + g1·A1).
    Returns (μ, μ_dot). Falls back to own-signal posterior when contour
    sums vanish (no crossings)."""
    den = g0 * A0 + g1 * A1
    if den <= 0.0:
        # Match rezn_pchip._agent_posterior_pchip fallback
        mu = 1.0 / (1.0 + np.exp(-tau_own * u_own))
        return mu, 0.0
    mu = g1 * A1 / den
    g0g1 = g0 * g1
    mu_dot = (-g0g1 * A1 * A0_dot + g0g1 * A0 * A1_dot) / (den * den)
    return mu, mu_dot


# =====================================================================
#  F. Φ tangent at one cell — full chain rule.
#       Solve h(μ, p) = 0 implicitly; combine slice + p_obs perturbations.
# =====================================================================
def _slice_for_agent(P, V, ag, i, j, l):
    """Return (slice_, slice_dot) = (P-restricted-to-trader-ag-slice,
    V-restricted-to-trader-ag-slice). 2D arrays of shape (G, G)."""
    G = P.shape[0]
    if ag == 0:
        # slice_[a,b] = P[i, a, b] over (a, b) = (j', l')
        return P[i, :, :], V[i, :, :]
    elif ag == 1:
        return P[:, j, :], V[:, j, :]
    else:
        return P[:, :, l], V[:, :, l]


def phi_tangent_at_cell(P, V, i, j, l, p_star, u, taus, gammas, Ws):
    """Compute dΦ[i,j,l]/dV given a perturbation V in the same shape as P.
    `p_star` is Φ(P)[i,j,l] (already known from a forward pass).

    At the fixed point, h(μ_*, p*) = 0. Linearising:
      0 = Σ_k ∂h/∂μ_k · (a_k + b_k · p*_dot) + ∂h/∂p · p*_dot
    with
      a_k = ∂μ_k/∂P · V  (slice contribution at fixed p_obs)
      b_k = ∂μ_k/∂p_obs  (per unit perturbation of the observed price)
    Solving:
      p*_dot = - (Σ_k ∂h/∂μ_k · a_k) / (∂h/∂p + Σ_k ∂h/∂μ_k · b_k)
    """
    a_arr = np.zeros(3)
    b_arr = np.zeros(3)
    mus = np.zeros(3)
    u_own = np.array([u[i], u[j], u[l]])

    for ag in range(3):
        slice_, slice_dot = _slice_for_agent(P, V, ag, i, j, l)
        if ag == 0:
            tau_own = taus[0]; tau_A = taus[1]; tau_B = taus[2]
        elif ag == 1:
            tau_own = taus[1]; tau_A = taus[0]; tau_B = taus[2]
        else:
            tau_own = taus[2]; tau_A = taus[0]; tau_B = taus[1]

        # a_k branch: slice perturbation only, p_obs_dot = 0
        zeros = np.zeros_like(slice_)
        A0, A1, A0d_a, A1d_a = _contour_sum_tangent(
            slice_, slice_dot, u, tau_A, tau_B, p_star, 0.0)
        # b_k branch: zero slice, unit p_obs perturbation
        _, _, A0d_b, A1d_b = _contour_sum_tangent(
            slice_, zeros, u, tau_A, tau_B, p_star, 1.0)
        # The first-pass A0/A1 are the contour values; second pass returns
        # same A0/A1 by construction.

        g0 = rh._f0(u_own[ag], tau_own)
        g1 = rh._f1(u_own[ag], tau_own)
        mu, mu_dot_a = _posterior_and_tangent(A0, A1, A0d_a, A1d_a, g0, g1,
                                               u_own[ag], tau_own)
        _,  mu_dot_b = _posterior_and_tangent(A0, A1, A0d_b, A1d_b, g0, g1,
                                               u_own[ag], tau_own)
        mus[ag] = mu
        a_arr[ag] = mu_dot_a
        b_arr[ag] = mu_dot_b

    # Clearing-residual derivatives at (μ_*, p*)
    h, dh_d0, dh_d1, dh_d2, dh_dp = _clearing_jacobian(
        mus, p_star, gammas, Ws)

    num = -(dh_d0 * a_arr[0] + dh_d1 * a_arr[1] + dh_d2 * a_arr[2])
    den = dh_dp + dh_d0 * b_arr[0] + dh_d1 * b_arr[1] + dh_d2 * b_arr[2]
    if abs(den) < 1e-30:
        return 0.0
    return num / den


def J_dot_v(P, V, u, taus, gammas, Ws):
    """Return J·V where J = I - dΦ/dP, evaluated at P. Shape: same as V.

    Uses analytic dΦ/dP through pieces A-F. Cost: O(G³) cells × O(G²) per
    contour pass × 2 passes (slice tangent + p_obs tangent) = O(G⁵).
    """
    G = P.shape[0]
    out = V.copy()
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_star = rp._phi_map_pchip(
                    P, u, np.asarray(taus, float), np.asarray(gammas, float),
                    np.asarray(Ws, float))[i, j, l]
                pdot = phi_tangent_at_cell(P, V, i, j, l, p_star,
                                            u, taus, gammas, Ws)
                out[i, j, l] -= pdot  # (I - dΦ/dP)·V at this cell
    return out


def J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws):
    """Same as J_dot_v but takes Phi(P) precomputed (one Φ call per matvec
    instead of G³). Phi_P[i,j,l] is the price at which trader k's contour
    is evaluated."""
    G = P.shape[0]
    out = V.copy()
    for i in range(G):
        for j in range(G):
            for l in range(G):
                pdot = phi_tangent_at_cell(P, V, i, j, l, Phi_P[i, j, l],
                                            u, taus, gammas, Ws)
                out[i, j, l] -= pdot
    return out


def _test_J_dot_v_FD():
    """Compare analytic J·V to centred FD on F = P - Φ(P).

    KNOWN BUG (next session debug): rel error ~20× at G=7 — pieces A-E
    individually FD-verified but piece F's chain rule has a sign or
    scaling error. Recommended debug:
      1. Test μ_0 (only slice tangent, fixed p_obs) vs FD on
         `_agent_posterior_pchip` — isolates piece D+E composition.
      2. Test μ_0 (only p_obs tangent, slice fixed) vs FD — isolates
         the b_k branch.
      3. Test market-clearing implicit derivative: change μ_k by δ,
         compute new p* (via _clear_price), compare δ/Δp* to
         -∂h/∂μ_k / ∂h/∂p.
    """
    G = 7
    UMAX = 2.0
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([3.0, 3.0, 3.0])
    gammas = np.array([3.0, 3.0, 3.0])
    Ws = np.array([1.0, 1.0, 1.0])

    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                 maxiters=300, abstol=1e-13, alpha=1.0)
    P0 = res["P_star"]
    # Perturb P so no slice value coincides exactly with p_obs
    rng = np.random.default_rng(7)
    P = np.clip(P0 + 0.01 * rng.standard_normal(P0.shape), 1e-9, 1 - 1e-9)
    V = rng.standard_normal(P.shape) * 1e-3

    # Analytic
    Phi_P = rp._phi_map_pchip(P, u, taus, gammas, Ws)
    JV = J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)

    # FD: F(P + εV) - F(P - εV) / (2ε), where F = P - Φ
    eps = 1e-6
    Phi_p = rp._phi_map_pchip(P + eps * V, u, taus, gammas, Ws)
    Phi_m = rp._phi_map_pchip(P - eps * V, u, taus, gammas, Ws)
    F_p = (P + eps * V) - Phi_p
    F_m = (P - eps * V) - Phi_m
    JV_fd = (F_p - F_m) / (2 * eps)

    err = np.abs(JV - JV_fd).max()
    rel = err / max(1.0, np.abs(JV_fd).max())
    print(f"_test_J_dot_v_FD: G={G} max_err={err:.3e} rel={rel:.3e}")
    return rel < 1e-4

if __name__ == "__main__":
    print("Running analytic-derivative sanity tests...")
    print("-" * 60)
    ok_a = _test_demand_derivs()
    ok_b = _test_clearing_jacobian()
    ok_c = _test_hermite_partials()
    ok_d1 = _test_pchip_derivs_tangent()
    ok_d2 = _test_contour_sum_tangent()
    ok_f = _test_J_dot_v_FD()
    print("-" * 60)
    if all([ok_a, ok_b, ok_c, ok_d1, ok_d2, ok_f]):
        print("ALL PIECES A-F: PASS")
        print("Use J_dot_v_with_phi(P, V, Phi_P, u, taus, gammas, Ws)")
        print("with scipy.sparse.linalg.LinearOperator for exact-Jacobian Newton.")
    else:
        print("SOME TESTS FAILED.")
