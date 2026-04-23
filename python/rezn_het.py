"""Heterogeneous-preference 2-pass contour solver (numba).

Extends rezn_numba to allow per-agent risk aversion gammas ∈ R^K and
signal precisions taus ∈ R^K (K=3). Each agent k:
  • has own precision τ_k — her private signal density is f_v(u_k, τ_k)
  • has own CRRA coefficient γ_k — demand x_k = W(R-1)/((1-p)+Rp)
    with R = exp((logit μ_k - logit p)/γ_k).

For agent k, the contour is drawn on her slice of the price tensor;
the axes of that slice correspond to the OTHER agents' signals,
so the density under each state uses those OTHER agents' τ's:
  agent 1 slice = Pg[i,:,:]  → axes (u_2, u_3) use (τ_2, τ_3)
  agent 2 slice = Pg[:,j,:]  → axes (u_1, u_3) use (τ_1, τ_3)
  agent 3 slice = Pg[:,:,l]  → axes (u_1, u_2) use (τ_1, τ_2)
"""

from __future__ import annotations
import numpy as np
from numba import njit
from scipy.linalg import lu_factor, lu_solve
import time
import resource


@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))

@njit(cache=True, fastmath=True)
def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True, fastmath=True)
def _f0(u, tau):
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau / 2.0 * (u + 0.5) ** 2)

@njit(cache=True, fastmath=True)
def _f1(u, tau):
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau / 2.0 * (u - 0.5) ** 2)


@njit(cache=True, fastmath=True)
def _demand_crra(mu, p, gamma, W):
    EPS = 1e-12
    mu_c = mu if mu >= EPS else EPS
    if mu_c > 1.0 - EPS:
        mu_c = 1.0 - EPS
    p_c = p if p >= EPS else EPS
    if p_c > 1.0 - EPS:
        p_c = 1.0 - EPS
    R = np.exp((_logit(mu_c) - _logit(p_c)) / gamma)
    return W * (R - 1.0) / ((1.0 - p_c) + R * p_c)


@njit(cache=True, fastmath=True)
def _clearing_residual(mus, p, gammas, Ws):
    s = 0.0
    for k in range(3):
        s += _demand_crra(mus[k], p, gammas[k], Ws[k])
    return s


@njit(cache=True)
def _clear_price(mus, gammas, Ws):
    a = 1e-9
    b = 1.0 - 1e-9
    fa = _clearing_residual(mus, a, gammas, Ws)
    fb = _clearing_residual(mus, b, gammas, Ws)
    if fa == 0.0: return a
    if fb == 0.0: return b
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = _clearing_residual(mus, m, gammas, Ws)
        if (b - a) < 1e-12 or fm == 0.0:
            return m
        if fa * fm < 0.0:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return 0.5 * (a + b)


@njit(cache=True)
def _contour_sum(slice_, u, tau_A, tau_B, p_obs):
    """Contour integral on a G×G slice. tau_A: precision for axis-A (rows),
    tau_B: precision for axis-B (columns)."""
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0
    # Pass A: sweep row a, crossings along columns. Grid point u_a is axis-A
    # (density uses tau_A), crossing u_b off-grid is axis-B (density uses tau_B).
    for a in range(G):
        row = slice_[a]
        ua = u[a]
        for k in range(G - 1):
            y1 = row[k] - p_obs
            y2 = row[k + 1] - p_obs
            prod = y1 * y2
            if prod < 0.0:
                t = y1 / (y1 - y2)
                ub = u[k] + t * (u[k + 1] - u[k])
                A0 += _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub, tau_B)
            elif y1 == 0.0 and y2 != 0.0:
                ub = u[k]
                A0 += _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub, tau_B)
            elif k == G - 2 and y2 == 0.0:
                ub = u[k + 1]
                A0 += _f0(ua, tau_A) * _f0(ub, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub, tau_B)
    # Pass B: sweep column b, crossings along rows. Grid point u_b is axis-B.
    for b in range(G):
        ub_grid = u[b]
        for k in range(G - 1):
            y1 = slice_[k, b] - p_obs
            y2 = slice_[k + 1, b] - p_obs
            prod = y1 * y2
            if prod < 0.0:
                t = y1 / (y1 - y2)
                ua = u[k] + t * (u[k + 1] - u[k])
                A0 += _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub_grid, tau_B)
            elif y1 == 0.0 and y2 != 0.0:
                ua = u[k]
                A0 += _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub_grid, tau_B)
            elif k == G - 2 and y2 == 0.0:
                ua = u[k + 1]
                A0 += _f0(ua, tau_A) * _f0(ub_grid, tau_B)
                A1 += _f1(ua, tau_A) * _f1(ub_grid, tau_B)
    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def _agent_posterior(ag, i, j, l, p_obs, Pg, u, taus):
    """taus = (τ_1, τ_2, τ_3). Agent `ag` (0-indexed)."""
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
    A0, A1 = _contour_sum(slice_, u, tau_A, tau_B, p_obs)
    g0 = _f0(u_own, tau_own); g1 = _f1(u_own, tau_own)
    den = g0 * A0 + g1 * A1
    if den <= 0.0:
        return _logistic(tau_own * u_own)
    return g1 * A1 / den


@njit(cache=True)
def _posteriors_at(i, j, l, p_obs, Pg, u, taus):
    m0 = _agent_posterior(0, i, j, l, p_obs, Pg, u, taus)
    m1 = _agent_posterior(1, i, j, l, p_obs, Pg, u, taus)
    m2 = _agent_posterior(2, i, j, l, p_obs, Pg, u, taus)
    return m0, m1, m2


@njit(cache=True)
def _residual_array(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at(i, j, l, p, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                F[i, j, l] = _clearing_residual(mus, p, gammas, Ws)
    return F


@njit(cache=True)
def _nolearning_price(u, taus, gammas, Ws):
    G = u.shape[0]
    P0 = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                m0 = _logistic(taus[0] * u[i])
                m1 = _logistic(taus[1] * u[j])
                m2 = _logistic(taus[2] * u[l])
                mus = np.array([m0, m1, m2])
                P0[i, j, l] = _clear_price(mus, gammas, Ws)
    return P0


@njit(cache=True)
def _phi_map(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at(i, j, l, p_cur, Pg, u, taus)
                mus = np.array([m0, m1, m2])
                Pnew[i, j, l] = _clear_price(mus, gammas, Ws)
    return Pnew


# --------------------------------------------------------
# Python wrappers
# --------------------------------------------------------

def _as_vec3(x):
    if np.isscalar(x):
        return np.asarray([float(x)] * 3)
    a = np.asarray(x, dtype=np.float64)
    assert a.shape == (3,)
    return a


def build_grid(G, umax=2.0):
    return np.linspace(-umax, umax, G)


def solve_picard(G, taus, gammas, umax=2.0, Ws=1.0,
                 maxiters=3000, abstol=1e-13, alpha=1.0):
    u = build_grid(G, umax)
    taus = _as_vec3(taus)
    gammas = _as_vec3(gammas)
    Ws = _as_vec3(Ws)
    P0 = _nolearning_price(u, taus, gammas, Ws)
    Pcur = P0.copy()
    history = []
    for _ in range(maxiters):
        Pnew = _phi_map(Pcur, u, taus, gammas, Ws)
        diff = float(np.abs(Pnew - Pcur).max())
        if alpha == 1.0:
            Pcur = Pnew
        else:
            Pcur = alpha * Pnew + (1 - alpha) * Pcur
        history.append(diff)
        if diff < abstol:
            break
    F = _residual_array(Pcur, u, taus, gammas, Ws)
    return dict(P_star=Pcur, P0=P0, u=u, residual=F, history=history,
                converged=bool(history and history[-1] < abstol),
                taus=taus, gammas=gammas)


def posteriors_at(i, j, l, p_obs, Pg, u, taus):
    taus = _as_vec3(taus)
    return _posteriors_at(i, j, l, p_obs, Pg, u, taus)


def one_minus_R2(Pg, u, tau_sum_scale):
    """1-R² where the CARA FR reference is logit(p) = (Σ τ_k u_k)/K."""
    G = u.shape[0]
    taus = _as_vec3(tau_sum_scale)
    y = np.log(Pg / (1.0 - Pg)).reshape(-1)
    T = np.empty(G ** 3)
    k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T[k] = taus[0] * u[i] + taus[1] * u[j] + taus[2] * u[l]
                k += 1
    y_c = y - y.mean()
    T_c = T - T.mean()
    Syy = float((y_c * y_c).sum())
    STT = float((T_c * T_c).sum())
    SyT = float((y_c * T_c).sum())
    if Syy == 0.0 or STT == 0.0:
        return 0.0
    R2 = (SyT * SyT) / (Syy * STT)
    return 1.0 - R2


def peak_rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")
