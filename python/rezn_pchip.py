"""Numba-accelerated PCHIP-based contour method.

Drop-in alternative to rezn_het._contour_sum — replaces the piecewise-
linear row/column interpolant with a monotone cubic Hermite (PCHIP) that
is C¹ smooth and does NOT overshoot on monotone data. All hot paths are
@njit so the solver runs at rezn_het speeds, not at scipy speeds.

The only real work beyond rezn_het:
  - PCHIP derivatives per row/column (Fritsch–Carlson formula)
  - Find crossings of a cubic Hermite segment with a target via Newton's
    method in the parametric variable t ∈ [0, 1] (monotone-in-segment by
    PCHIP construction, so Newton converges quadratically).

Reuses rezn_het for: demand, market clearing, grid, nolearning_price.
"""
from __future__ import annotations
import numpy as np
from numba import njit
import rezn_het as rh


# ------------ PCHIP derivatives (Fritsch–Carlson, uniform spacing) ------

@njit(cache=True, fastmath=True)
def _pchip_derivs(y, u):
    """Return per-node derivatives m[0..G-1] for monotone cubic Hermite.
    Uniform spacing assumed (our grid is). Handles non-uniform too."""
    G = y.shape[0]
    m = np.empty(G)
    s = np.empty(G - 1)
    h = np.empty(G - 1)
    for k in range(G - 1):
        h[k] = u[k + 1] - u[k]
        s[k] = (y[k + 1] - y[k]) / h[k]

    # Interior: Fritsch–Carlson
    for k in range(1, G - 1):
        if s[k - 1] * s[k] <= 0.0:
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])

    # Endpoint derivatives (non-centered three-point with monotone clamp)
    # Left end
    d = ((2.0 * h[0] + h[1]) * s[0] - h[0] * s[1]) / (h[0] + h[1])
    if d * s[0] <= 0.0:
        d = 0.0
    elif (s[0] * s[1] <= 0.0) and (abs(d) > abs(3.0 * s[0])):
        d = 3.0 * s[0]
    m[0] = d

    # Right end
    d = ((2.0 * h[G - 2] + h[G - 3]) * s[G - 2] - h[G - 2] * s[G - 3]) / (h[G - 2] + h[G - 3])
    if d * s[G - 2] <= 0.0:
        d = 0.0
    elif (s[G - 2] * s[G - 3] <= 0.0) and (abs(d) > abs(3.0 * s[G - 2])):
        d = 3.0 * s[G - 2]
    m[G - 1] = d
    return m


@njit(cache=True, fastmath=True, inline="always")
def _hermite_val(t, y0, y1, m0, m1, h):
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


@njit(cache=True, fastmath=True, inline="always")
def _hermite_deriv(t, y0, y1, m0, m1, h):
    # derivative wrt t of the cubic Hermite
    t2 = t * t
    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    return dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1


@njit(cache=True)
def _pchip_root_in_segment(y0, y1, m0, m1, h, p_obs):
    """Find t* in (0,1) such that H(t*) = p_obs. Returns t* or -1 on failure.

    PCHIP is monotone within each segment when the data is monotone between
    the two endpoints, so Newton from the centre converges fast. Fall back
    to bisection on failure.
    """
    d0 = y0 - p_obs
    d1 = y1 - p_obs
    if d0 * d1 > 0.0:
        return -1.0  # no crossing

    # Newton from t=0.5 target = 0
    t = 0.5
    for _ in range(30):
        H = _hermite_val(t, y0, y1, m0, m1, h) - p_obs
        if abs(H) < 1e-14:
            break
        Hp = _hermite_deriv(t, y0, y1, m0, m1, h)
        if Hp == 0.0:
            break
        t_new = t - H / Hp
        if t_new <= 0.0 or t_new >= 1.0:
            # Newton escapes; bisect
            break
        t = t_new

    # If Newton didn't deliver, bisect
    H_here = _hermite_val(t, y0, y1, m0, m1, h) - p_obs
    if abs(H_here) > 1e-12:
        a = 0.0; b = 1.0
        fa = d0; fb = d1
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = _hermite_val(m, y0, y1, m0, m1, h) - p_obs
            if fa * fm <= 0.0:
                b = m; fb = fm
            else:
                a = m; fa = fm
            if abs(fm) < 1e-14 or (b - a) < 1e-14:
                break
        t = 0.5 * (a + b)
    return t


# ------------ Contour sum with PCHIP per row/column ---------------------

@njit(cache=True)
def _contour_sum_pchip(slice_, u, tau_A, tau_B, p_obs):
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0
    # reusable derivative buffers
    m_row = np.empty(G)
    m_col = np.empty(G)

    # Pass A: rows (along axis-B)
    for a in range(G):
        row = slice_[a]
        ua = u[a]
        # PCHIP derivatives for this row
        m_row[:] = _pchip_derivs(row, u)
        for k in range(G - 1):
            y0 = row[k]; y1 = row[k + 1]
            d0 = y0 - p_obs; d1 = y1 - p_obs
            h = u[k + 1] - u[k]
            if d0 * d1 < 0.0:
                t = _pchip_root_in_segment(y0, y1, m_row[k], m_row[k + 1], h, p_obs)
                if t < 0.0: continue
                ub = u[k] + t * h
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif d0 == 0.0 and d1 != 0.0:
                ub = u[k]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif k == G - 2 and d1 == 0.0:
                ub = u[k + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)

    # Pass B: columns (along axis-A)
    for b in range(G):
        ub_grid = u[b]
        # build column into local contiguous array
        col = np.empty(G)
        for i in range(G):
            col[i] = slice_[i, b]
        m_col[:] = _pchip_derivs(col, u)
        for k in range(G - 1):
            y0 = col[k]; y1 = col[k + 1]
            d0 = y0 - p_obs; d1 = y1 - p_obs
            h = u[k + 1] - u[k]
            if d0 * d1 < 0.0:
                t = _pchip_root_in_segment(y0, y1, m_col[k], m_col[k + 1], h, p_obs)
                if t < 0.0: continue
                ua = u[k] + t * h
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)
            elif d0 == 0.0 and d1 != 0.0:
                ua = u[k]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)
            elif k == G - 2 and d1 == 0.0:
                ua = u[k + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)

    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def _agent_posterior_pchip(ag, i, j, l, p_obs, Pg, u, taus):
    G = u.shape[0]
    if ag == 0:
        u_own = u[i]; tau_own = taus[0]
        tau_A = taus[1]; tau_B = taus[2]
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[i, a, b]
    elif ag == 1:
        u_own = u[j]; tau_own = taus[1]
        tau_A = taus[0]; tau_B = taus[2]
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[a, j, b]
    else:
        u_own = u[l]; tau_own = taus[2]
        tau_A = taus[0]; tau_B = taus[1]
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[a, b, l]
    A0, A1 = _contour_sum_pchip(slice_, u, tau_A, tau_B, p_obs)
    g0 = rh._f0(u_own, tau_own); g1 = rh._f1(u_own, tau_own)
    den = g0 * A0 + g1 * A1
    if den <= 0.0:
        return 1.0 / (1.0 + np.exp(-tau_own * u_own))
    return g1 * A1 / den


@njit(cache=True)
def _posteriors_at_pchip(i, j, l, p_obs, Pg, u, taus):
    m0 = _agent_posterior_pchip(0, i, j, l, p_obs, Pg, u, taus)
    m1 = _agent_posterior_pchip(1, i, j, l, p_obs, Pg, u, taus)
    m2 = _agent_posterior_pchip(2, i, j, l, p_obs, Pg, u, taus)
    return m0, m1, m2


@njit(cache=True)
def _phi_map_pchip(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at_pchip(i, j, l, p_cur, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                Pnew[i, j, l] = rh._clear_price(mus, gammas, Ws)
    return Pnew


@njit(cache=True)
def _residual_array_pchip(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at_pchip(i, j, l, p, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                F[i, j, l] = rh._clearing_residual(mus, p, gammas, Ws)
    return F


def solve_anderson_pchip(G, taus, gammas, umax=2.0, Ws=1.0,
                          maxiters=500, abstol=1e-13, m_window=6,
                          P_init=None, damping=1.0):
    """Anderson-accelerated fixed-point iteration on the PCHIP Φ map.

    At each step we store the last m_window iterates and fixed-point
    residuals g_n = Φ(x_n) - x_n. Next iterate is the Anderson-optimal
    linear combination:
        x_{n+1} = x_n + damping * g_n - (ΔX + damping*ΔG) γ_n
    where γ_n minimises ‖g_n - ΔG γ_n‖.
    """
    u = np.linspace(-umax, umax, G)
    taus = rh._as_vec3(taus)
    gammas = rh._as_vec3(gammas)
    Ws = rh._as_vec3(Ws)
    P0 = rh._nolearning_price(u, taus, gammas, Ws)
    x = (P_init.reshape(-1).copy() if P_init is not None else P0.reshape(-1).copy())
    x = np.clip(x, 1e-9, 1 - 1e-9)

    Xs = []
    Gs = []
    history = []
    for it in range(maxiters):
        Pcur = x.reshape(G, G, G)
        Pnew = _phi_map_pchip(Pcur, u, taus, gammas, Ws)
        g = Pnew.reshape(-1) - x
        diff = float(np.abs(g).max())
        history.append(diff)
        if diff < abstol:
            x = Pnew.reshape(-1)
            break
        Xs.append(x.copy()); Gs.append(g.copy())
        if len(Xs) > m_window + 1:
            Xs.pop(0); Gs.pop(0)

        if len(Gs) == 1:
            x_new = x + damping * g
        else:
            dG = np.column_stack([Gs[k + 1] - Gs[k] for k in range(len(Gs) - 1)])
            dX = np.column_stack([Xs[k + 1] - Xs[k] for k in range(len(Xs) - 1)])
            gamma, *_ = np.linalg.lstsq(dG, g, rcond=None)
            x_new = x + damping * g - (dX + damping * dG) @ gamma
        x = np.clip(x_new, 1e-9, 1 - 1e-9)

    Pstar = x.reshape(G, G, G)
    F = _residual_array_pchip(Pstar, u, taus, gammas, Ws)
    return dict(P_star=Pstar, P0=P0, u=u, residual=F,
                history=history, converged=(history and history[-1] < abstol),
                taus=taus, gammas=gammas)


def solve_picard_pchip(G, taus, gammas, umax=2.0, Ws=1.0,
                       maxiters=3000, abstol=1e-13, alpha=1.0,
                       P_init=None):
    u = np.linspace(-umax, umax, G)
    taus = rh._as_vec3(taus)
    gammas = rh._as_vec3(gammas)
    Ws = rh._as_vec3(Ws)
    P0 = rh._nolearning_price(u, taus, gammas, Ws)
    if P_init is not None:
        Pcur = np.clip(P_init, 1e-9, 1.0 - 1e-9).copy()
    else:
        Pcur = P0.copy()
    history = []
    for _ in range(maxiters):
        Pnew = _phi_map_pchip(Pcur, u, taus, gammas, Ws)
        diff = float(np.abs(Pnew - Pcur).max())
        if alpha == 1.0:
            Pcur = Pnew
        else:
            Pcur = alpha * Pnew + (1 - alpha) * Pcur
        history.append(diff)
        if diff < abstol:
            break
    F = _residual_array_pchip(Pcur, u, taus, gammas, Ws)
    return dict(P_star=Pcur, P0=P0, u=u, residual=F,
                history=history, converged=(history[-1] < abstol),
                taus=taus, gammas=gammas)


# ------------ Smoke test ------------------------------------------------

if __name__ == "__main__":
    import time
    print("numba JIT compile…")
    _ = solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

    known = [
        ((3, 3, 3), (50, 50, 50)),
        ((3, 3, 3), (3, 3, 3)),
        ((3, 3, 3), (0.3, 50, 50)),
    ]
    for t, g in known:
        print(f"\n=== PCHIP (numba) τ={t} γ={g} ===")
        t0 = time.time()
        res = solve_picard_pchip(9, t, g, umax=2.0,
                                  maxiters=3000, abstol=1e-14, alpha=1.0)
        dt = time.time() - t0
        PhiI = res["history"][-1]
        Finf = float(np.abs(res["residual"]).max())
        print(f"  iters   : {len(res['history'])}")
        print(f"  time    : {dt:.1f} s")
        print(f"  ‖Φ-I‖∞  : {PhiI:.3e}")
        print(f"  ‖F‖∞    : {Finf:.3e}")
        print(f"  last 5  : {['%.2e' % x for x in res['history'][-5:]]}")
