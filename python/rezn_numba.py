"""Numba-accelerated port of the 2-pass contour REE solver (CRRA).

All hot paths compiled with @njit. Used for a fair language-overhead
comparison with Julia/src/Rezn.jl. Uses scipy.linalg.lu_factor /
lu_solve (MKL/OpenBLAS) for the Newton-LU dense solve.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from scipy.linalg import lu_factor, lu_solve
import time
import resource

SQRT_2PI_INV = 1.0 / np.sqrt(2 * np.pi)

# ---------------------------------------------------------------
# Scalar primitives (all @njit)
# ---------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True, fastmath=True)
def _logit(p):
    return np.log(p / (1.0 - p))


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
def _clearing_residual(mu0, mu1, mu2, p, gamma, W):
    return (_demand_crra(mu0, p, gamma, W)
          + _demand_crra(mu1, p, gamma, W)
          + _demand_crra(mu2, p, gamma, W))


# ---------------------------------------------------------------
# Bisection for market clearing (closure-free)
# ---------------------------------------------------------------

@njit(cache=True)
def _clear_price(mu0, mu1, mu2, gamma, W):
    a = 1e-9
    b = 1.0 - 1e-9
    fa = _clearing_residual(mu0, mu1, mu2, a, gamma, W)
    fb = _clearing_residual(mu0, mu1, mu2, b, gamma, W)
    if fa == 0.0: return a
    if fb == 0.0: return b
    # assume fa * fb < 0
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = _clearing_residual(mu0, mu1, mu2, m, gamma, W)
        if (b - a) < 1e-12 or fm == 0.0:
            return m
        if fa * fm < 0:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return 0.5 * (a + b)


# ---------------------------------------------------------------
# Piecewise-linear contour sum (single slice, both passes)
# Returns (A0, A1).
# ---------------------------------------------------------------

@njit(cache=True)
def _contour_sum(slice_, u, tau, p_obs):
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0
    # Pass A: sweep row a, crossings along columns
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
                A0 += _f0(ua, tau) * _f0(ub, tau)
                A1 += _f1(ua, tau) * _f1(ub, tau)
            elif y1 == 0.0 and y2 != 0.0:
                ub = u[k]
                A0 += _f0(ua, tau) * _f0(ub, tau)
                A1 += _f1(ua, tau) * _f1(ub, tau)
            elif k == G - 2 and y2 == 0.0:
                ub = u[k + 1]
                A0 += _f0(ua, tau) * _f0(ub, tau)
                A1 += _f1(ua, tau) * _f1(ub, tau)
    # Pass B: sweep column b, crossings along rows
    for b in range(G):
        col0 = u[b]  # u_b grid value
        for k in range(G - 1):
            y1 = slice_[k, b]     - p_obs
            y2 = slice_[k + 1, b] - p_obs
            prod = y1 * y2
            if prod < 0.0:
                t = y1 / (y1 - y2)
                ua = u[k] + t * (u[k + 1] - u[k])
                A0 += _f0(ua, tau) * _f0(col0, tau)
                A1 += _f1(ua, tau) * _f1(col0, tau)
            elif y1 == 0.0 and y2 != 0.0:
                ua = u[k]
                A0 += _f0(ua, tau) * _f0(col0, tau)
                A1 += _f1(ua, tau) * _f1(col0, tau)
            elif k == G - 2 and y2 == 0.0:
                ua = u[k + 1]
                A0 += _f0(ua, tau) * _f0(col0, tau)
                A1 += _f1(ua, tau) * _f1(col0, tau)
    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def _agent_posterior(ag, i, j, l, p_obs, Pg, u, tau):
    G = u.shape[0]
    if ag == 0:
        u_own = u[i]
        # slice = Pg[i, :, :] as contiguous view would be ideal, but numba
        # needs explicit loops for type stability. Build a local buffer:
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[i, a, b]
    elif ag == 1:
        u_own = u[j]
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[a, j, b]
    else:
        u_own = u[l]
        slice_ = np.empty((G, G))
        for a in range(G):
            for b in range(G):
                slice_[a, b] = Pg[a, b, l]
    A0, A1 = _contour_sum(slice_, u, tau, p_obs)
    g0 = _f0(u_own, tau); g1 = _f1(u_own, tau)
    den = g0 * A0 + g1 * A1
    if den <= 0.0:
        return _logistic(tau * u_own)
    return g1 * A1 / den


@njit(cache=True)
def _posteriors_at(i, j, l, p_obs, Pg, u, tau):
    m0 = _agent_posterior(0, i, j, l, p_obs, Pg, u, tau)
    m1 = _agent_posterior(1, i, j, l, p_obs, Pg, u, tau)
    m2 = _agent_posterior(2, i, j, l, p_obs, Pg, u, tau)
    return m0, m1, m2


@njit(cache=True, parallel=False)
def _residual_array(Pg, u, tau, gamma, W):
    G = u.shape[0]
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at(i, j, l, p, Pg, u, tau)
                F[i, j, l] = _clearing_residual(m0, m1, m2, p, gamma, W)
    return F


@njit(cache=True)
def _nolearning_price(u, tau, gamma, W):
    G = u.shape[0]
    P0 = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                m0 = _logistic(tau * u[i])
                m1 = _logistic(tau * u[j])
                m2 = _logistic(tau * u[l])
                P0[i, j, l] = _clear_price(m0, m1, m2, gamma, W)
    return P0


@njit(cache=True)
def _phi_map(Pg, u, tau, gamma, W):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                m0, m1, m2 = _posteriors_at(i, j, l, p_cur, Pg, u, tau)
                Pnew[i, j, l] = _clear_price(m0, m1, m2, gamma, W)
    return Pnew


@njit(cache=True)
def _fd_jacobian(x0, u, tau, gamma, W, G, h):
    n = x0.shape[0]
    J = np.empty((n, n))
    xp = x0.copy(); xm = x0.copy()
    for k in range(n):
        xp[k] = x0[k] + h
        xm[k] = x0[k] - h
        fp = _residual_array(xp.reshape(G, G, G), u, tau, gamma, W).reshape(-1)
        fm = _residual_array(xm.reshape(G, G, G), u, tau, gamma, W).reshape(-1)
        for i in range(n):
            J[i, k] = (fp[i] - fm[i]) / (2.0 * h)
        xp[k] = x0[k]; xm[k] = x0[k]
    return J


# ---------------------------------------------------------------
# Driver (Python; calls into numba kernels)
# ---------------------------------------------------------------

def build_grid(G, umax=2.0):
    return np.linspace(-umax, umax, G)


def solve_picard(G, tau, gamma, umax=2.0, W=1.0,
                 maxiters=2000, abstol=1e-13):
    u = build_grid(G, umax)
    P0 = _nolearning_price(u, tau, gamma, W)
    Pcur = P0.copy()
    history = []
    for _ in range(maxiters):
        Pnew = _phi_map(Pcur, u, tau, gamma, W)
        diff = float(np.abs(Pnew - Pcur).max())
        Pcur = Pnew
        history.append(diff)
        if diff < abstol:
            break
    F = _residual_array(Pcur, u, tau, gamma, W)
    return dict(P_star=Pcur, P0=P0, u=u, residual=F, history=history,
                converged=bool(history and history[-1] < abstol))


def solve_newton_lu(G, tau, gamma, umax=2.0, W=1.0, x0=None,
                    maxiters=5, abstol=1e-11, h=1e-6):
    u = build_grid(G, umax)
    P0 = _nolearning_price(u, tau, gamma, W)
    x = x0.copy() if x0 is not None else P0.reshape(-1).copy()
    n = x.size

    def Fvec(y):
        return _residual_array(y.reshape(G, G, G), u, tau, gamma, W).reshape(-1)

    Fx = Fvec(x)
    fnorm = float(np.abs(Fx).max())
    lam = 0.0
    history = [fnorm]
    t_jac = t_lu = t_solve = t_ls = 0.0

    for it in range(maxiters):
        if fnorm < abstol:
            break
        t0 = time.time()
        J = _fd_jacobian(x, u, tau, gamma, W, G, h)
        t_jac += time.time() - t0

        t0 = time.time()
        lu_pv = None
        while True:
            Jd = J.copy()
            lam_eff = max(lam, 1e-10)
            np.fill_diagonal(Jd, np.diag(Jd) + lam_eff)
            try:
                lu_pv = lu_factor(Jd, check_finite=False)
                break
            except (np.linalg.LinAlgError, ValueError):
                lam = lam * 10 if lam > 0 else 1e-6
                if lam > 1e6:
                    lu_pv = None
                    break
        if lu_pv is None:
            break
        t_lu += time.time() - t0

        t0 = time.time()
        dx = lu_solve(lu_pv, -Fx, check_finite=False)
        t_solve += time.time() - t0

        t0 = time.time()
        alpha = 1.0
        xtrial = x + alpha * dx
        Ftrial = Fvec(xtrial)
        nt = float(np.abs(Ftrial).max())
        tries = 0
        while nt > fnorm and tries < 25:
            alpha *= 0.5
            xtrial = x + alpha * dx
            Ftrial = Fvec(xtrial)
            nt = float(np.abs(Ftrial).max())
            tries += 1
        t_ls += time.time() - t0

        if nt >= fnorm:
            lam = lam * 10 if lam > 0 else 1e-6
            history.append(fnorm)
            if lam > 1e12:
                break
            continue
        else:
            lam = max(lam / 2, 0.0)

        x = xtrial; Fx = Ftrial; fnorm = nt
        history.append(fnorm)

    Pg_star = x.reshape(G, G, G)
    return dict(P_star=Pg_star, P0=P0, u=u, residual=Fx.reshape(G, G, G),
                history=history, converged=(fnorm < abstol),
                timings=dict(jac=t_jac, lu=t_lu, solve=t_solve, ls=t_ls,
                             total=t_jac + t_lu + t_solve + t_ls))


def posteriors_at(i, j, l, p_obs, Pg, u, tau):
    """Convenience wrapper to call numba posteriors from Python."""
    return _posteriors_at(i, j, l, p_obs, Pg, u, tau)


def peak_rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")
