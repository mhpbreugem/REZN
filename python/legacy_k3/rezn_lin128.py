"""Pure-numpy float128 port of the linear-interp contour kernel.

Mirrors rezn_het.py exactly but in pure numpy (no numba) and parameterised
to work in float128. Slower per-call but truly float128 throughout —
no f64→f128 round-trips, no LAPACK fallback, no numba downcast.

Use Φ map: `phi_map_lin128(P, u, taus, gammas, Ws)` — all inputs and
output are np.float128.

Compatible signatures with rezn_het: _f0, _f1, _demand_crra,
_clearing_residual, _clear_price, _contour_sum, _agent_posterior,
_posteriors_at, _phi_map.
"""
from __future__ import annotations
import numpy as np

DTYPE = np.float128
EPS = DTYPE("1e-12")
ZERO = DTYPE(0.0)
ONE  = DTYPE(1.0)
HALF = DTYPE(0.5)
TWO  = DTYPE(2.0)
PI   = DTYPE(np.pi)


def _f0(u, tau):
    return np.sqrt(tau / (TWO * PI)) * np.exp(-tau / TWO * (u + HALF) ** 2)

def _f1(u, tau):
    return np.sqrt(tau / (TWO * PI)) * np.exp(-tau / TWO * (u - HALF) ** 2)

def _logit(p):
    return np.log(p / (ONE - p))


def _demand_crra(mu, p, gamma, W):
    mu_c = mu if mu >= EPS else EPS
    if mu_c > ONE - EPS: mu_c = ONE - EPS
    p_c = p if p >= EPS else EPS
    if p_c > ONE - EPS: p_c = ONE - EPS
    R = np.exp((_logit(mu_c) - _logit(p_c)) / gamma)
    return W * (R - ONE) / ((ONE - p_c) + R * p_c)


def _clearing_residual(mus, p, gammas, Ws):
    s = ZERO
    for k in range(3):
        s = s + _demand_crra(mus[k], p, gammas[k], Ws[k])
    return s


def _clear_price(mus, gammas, Ws):
    a = DTYPE("1e-9")
    b = ONE - DTYPE("1e-9")
    fa = _clearing_residual(mus, a, gammas, Ws)
    fb = _clearing_residual(mus, b, gammas, Ws)
    if fa == ZERO: return a
    if fb == ZERO: return b
    for _ in range(300):  # tighter than rezn_het's 200 (we have more precision)
        m = HALF * (a + b)
        fm = _clearing_residual(mus, m, gammas, Ws)
        if (b - a) < DTYPE("1e-25") or fm == ZERO:
            return m
        if fa * fm < ZERO:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return HALF * (a + b)


def _contour_sum(slice_, u, tau_A, tau_B, p_obs):
    """Linear interpolation contour sum (mirrors rezn_het)."""
    G = u.shape[0]
    A0 = ZERO
    A1 = ZERO
    # Pass A: rows
    for a in range(G):
        row = slice_[a]
        ua = u[a]
        for k in range(G - 1):
            y1 = row[k] - p_obs
            y2 = row[k + 1] - p_obs
            prod = y1 * y2
            if prod < ZERO:
                t = y1 / (y1 - y2)
                ub = u[k] + t * (u[k + 1] - u[k])
                A0 = A0 + _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub, tau_B)
            elif y1 == ZERO and y2 != ZERO:
                ub = u[k]
                A0 = A0 + _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub, tau_B)
            elif k == G - 2 and y2 == ZERO:
                ub = u[k + 1]
                A0 = A0 + _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub, tau_B)
    # Pass B: cols
    for b in range(G):
        ub_grid = u[b]
        for k in range(G - 1):
            y1 = slice_[k, b] - p_obs
            y2 = slice_[k + 1, b] - p_obs
            prod = y1 * y2
            if prod < ZERO:
                t = y1 / (y1 - y2)
                ua = u[k] + t * (u[k + 1] - u[k])
                A0 = A0 + _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub_grid, tau_B)
            elif y1 == ZERO and y2 != ZERO:
                ua = u[k]
                A0 = A0 + _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub_grid, tau_B)
            elif k == G - 2 and y2 == ZERO:
                ua = u[k + 1]
                A0 = A0 + _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 = A1 + _f1(ua, tau_A) * _f1(ub_grid, tau_B)
    return HALF * A0, HALF * A1


def _agent_posterior(ag, i, j, l, p_obs, Pg, u, taus):
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
    A0, A1 = _contour_sum(slice_, u, tau_A, tau_B, p_obs)
    f0o = _f0(u_own, tau_own)
    f1o = _f1(u_own, tau_own)
    num = f1o * A1
    den = f0o * A0 + num
    if den <= ZERO:
        return HALF
    return num / den


def _posteriors_at(i, j, l, p_obs, Pg, u, taus):
    m0 = _agent_posterior(0, i, j, l, p_obs, Pg, u, taus)
    m1 = _agent_posterior(1, i, j, l, p_obs, Pg, u, taus)
    m2 = _agent_posterior(2, i, j, l, p_obs, Pg, u, taus)
    return m0, m1, m2


def _phi_map(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at(i, j, l, p_cur, Pg, u, taus)
                mus = np.array([m0, m1, m2], dtype=DTYPE)
                Pnew[i, j, l] = _clear_price(mus, gammas, Ws)
    return Pnew


def to_f128(x):
    """Helper: cast ndarray or scalar to float128."""
    return np.asarray(x, dtype=DTYPE)
