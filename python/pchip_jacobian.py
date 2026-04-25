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


# =====================================================================
#  D, E, F: contour-sum, posterior, Φ tangent — TODO for next session
# =====================================================================
# The next session should:
#
# 1. Implement `_contour_sum_tangent(slice_, slice_dot, u, tau_A, tau_B,
#                                    p_obs, p_obs_dot)`:
#    - Returns (A0, A1, dA0, dA1) where dA0, dA1 are tangent updates for
#      the given slice perturbation `slice_dot` and `p_obs_dot`.
#    - Mirror the logic in `_contour_sum_pchip` but, for each crossing,
#      compute the perturbation in u_b (or u_a, depending on the pass)
#      using the implicit derivative of `_pchip_root_in_segment`.
#    - Sum tangent contributions for both f0 and f1 weights.
#    - **In LOGIT space**: the slice is `L = log(p/(1-p))` of the 2D
#      slice, and `lp_obs = logit(p_obs)`. Convert tangents accordingly:
#      d(L_ab)/d(P_ab) = 1 / (P_ab(1-P_ab))  for chain rule back.
#
# 2. Implement `_posterior_tangent(...)`:
#    - μ = g1·A1 / (g0·A0 + g1·A1).
#    - dμ/dA0 = -g1·g0·A1 / (g0·A0 + g1·A1)²
#    - dμ/dA1 =  g1·g0·A0 / (g0·A0 + g1·A1)²
#
# 3. Implement `phi_tangent_at_cell(P, V, i, j, l, taus, gammas, Ws, u)`:
#    - For each trader k, compute (μ_k, dμ_k) using slices of (P, V).
#    - Solve market clearing using `_clear_price` to get p* (already
#      satisfies h(μ_*, p*) = 0).
#    - Newton: dp*/d(input) = -(∂h/∂input)/(∂h/∂p) at the solution.
#    - Combine via chain rule:
#      dΦ[i,j,l]/dV = Σ_k (-∂h/∂μ_k / ∂h/∂p) · dμ_k(V) + ...
#    - Note p_obs in the contour for trader k is precisely Φ[i,j,l] = p*,
#      so when V perturbs P[i,j,l] there is a term through p_obs as well.
#
# 4. Implement `phi_and_tangent_full(P, V) -> (Phi, dPhi)`:
#    - Loop over all (i, j, l) calling `phi_tangent_at_cell`.
#    - This is O(G^3) cells, each with O(G^2) contour work → O(G^5)
#      total. At G=11, that's ~150k operations, comparable to one full
#      Φ evaluation.
#
# 5. `J_dot_v(P, V) = V - phi_and_tangent_full(P, V)[1]`.
#
# 6. `linear_operator(P)`: returns scipy.sparse.linalg.LinearOperator
#    with the J_dot_v matvec.
#
# 7. Custom Newton in pchip_continuation.py:
#    - `delta = lgmres(LinearOperator, -F, atol=1e-13, rtol=1e-12)[0]`
#    - `x_new = x + delta` (with line search if needed).
#    - Iterate to ||F|| < 1e-12.

if __name__ == "__main__":
    print("Running analytic-derivative sanity tests...")
    print("-" * 60)
    ok_a = _test_demand_derivs()
    ok_b = _test_clearing_jacobian()
    ok_c = _test_hermite_partials()
    print("-" * 60)
    if ok_a and ok_b and ok_c:
        print("ALL PIECES A, B, C: PASS")
        print("Next session: implement D (contour tangent), E (posterior")
        print("tangent), F (Φ tangent at cell). See module docstring.")
    else:
        print("SOME TESTS FAILED. Review derivatives before proceeding.")
