"""Python port of the 2-pass contour REE solver (CRRA).

Mirrors Julia/src/Rezn.jl. Uses numpy for the G×G slice operations;
outer loop over G^3 cells is plain Python. No numba / no threading —
realistic pure-numpy baseline for comparison with the Julia solver.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time
import resource

# ---------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------

def logistic(x):  return 1.0 / (1.0 + np.exp(-x))
def logit(p):     return np.log(p / (1.0 - p))

def f0(u, tau):   return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u + 0.5) ** 2)
def f1(u, tau):   return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - 0.5) ** 2)

EPS = 1e-12


def demand_crra(mu, p, gamma, W=1.0):
    mu_c = min(max(mu, EPS), 1.0 - EPS)
    p_c  = min(max(p,  EPS), 1.0 - EPS)
    R = np.exp((np.log(mu_c/(1-mu_c)) - np.log(p_c/(1-p_c))) / gamma)
    return W * (R - 1.0) / ((1.0 - p_c) + R * p_c)


def clearing_residual(mus, p, gamma, W=1.0):
    return sum(demand_crra(m, p, gamma, W) for m in mus)


def bisect(f, a, b, tol=1e-12, maxiter=200):
    fa = f(a); fb = f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0:
        raise ValueError(f"bisect: no sign change fa={fa}, fb={fb}")
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(b - a) < tol or fm == 0:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def clear_price(mus, gamma, W=1.0):
    return bisect(lambda p: clearing_residual(mus, p, gamma, W), 1e-9, 1 - 1e-9)


def build_grid(G, umax=2.0):
    return np.linspace(-umax, umax, G)


# ---------------------------------------------------------------
# Piecewise-linear 1D crossings — vectorised
# ---------------------------------------------------------------

def find_crossings(xvals, yvals, target):
    """Return array of x such that piecewise-linear interpolant(yvals, xvals) = target."""
    d1 = yvals[:-1] - target
    d2 = yvals[1:]  - target
    # strict crossings (sign change, neither endpoint exactly on target)
    mask_strict = d1 * d2 < 0
    # left-endpoint exactly on target — include at start of each segment
    mask_left_zero = (d1 == 0) & (d2 != 0)
    # right-endpoint on target only captured on final segment to avoid dupes
    n = len(xvals)
    tail = np.zeros(n - 1, dtype=bool)
    if n >= 2 and d2[-1] == 0:
        tail[-1] = True

    crossings = []
    if mask_strict.any():
        idx = np.where(mask_strict)[0]
        t = d1[idx] / (d1[idx] - d2[idx])
        x = xvals[idx] + t * (xvals[idx + 1] - xvals[idx])
        crossings.append(x)
    if mask_left_zero.any():
        idx = np.where(mask_left_zero)[0]
        crossings.append(xvals[idx])
    if tail.any():
        idx = np.where(tail)[0]
        crossings.append(xvals[idx + 1])

    return np.concatenate(crossings) if crossings else np.empty(0)


# ---------------------------------------------------------------
# Agent posterior via 2-pass contour — vectorised over sweep axis
# ---------------------------------------------------------------

def _contour_sum(slice_, u, tau, p_obs):
    """Return (A_0, A_1) for the 2-pass contour on a G×G slice."""
    G = len(u)
    A0 = 0.0; A1 = 0.0
    # Pass A: fix axis-A (row), crossings along axis-B (columns)
    for a in range(G):
        crosses = find_crossings(u, slice_[a, :], p_obs)
        if crosses.size:
            ua = u[a]
            A0 += np.sum(f0(ua, tau) * f0(crosses, tau))
            A1 += np.sum(f1(ua, tau) * f1(crosses, tau))
    # Pass B: fix axis-B (column), crossings along axis-A (rows)
    for b in range(G):
        crosses = find_crossings(u, slice_[:, b], p_obs)
        if crosses.size:
            ub = u[b]
            A0 += np.sum(f0(crosses, tau) * f0(ub, tau))
            A1 += np.sum(f1(crosses, tau) * f1(ub, tau))
    return 0.5 * A0, 0.5 * A1


def agent_posterior(ag, i, j, l, p_obs, Pg, u, tau):
    if ag == 0:
        u_own = u[i]; slice_ = Pg[i, :, :]
    elif ag == 1:
        u_own = u[j]; slice_ = Pg[:, j, :]
    else:
        u_own = u[l]; slice_ = Pg[:, :, l]
    A0, A1 = _contour_sum(slice_, u, tau, p_obs)
    g0 = f0(u_own, tau); g1 = f1(u_own, tau)
    den = g0 * A0 + g1 * A1
    if den <= 0:
        return logistic(tau * u_own)
    return g1 * A1 / den


def posteriors_at(i, j, l, p_obs, Pg, u, tau):
    return (agent_posterior(0, i, j, l, p_obs, Pg, u, tau),
            agent_posterior(1, i, j, l, p_obs, Pg, u, tau),
            agent_posterior(2, i, j, l, p_obs, Pg, u, tau))


def residual_array(Pg, u, tau, gamma):
    G = len(u)
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = Pg[i, j, l]
                mus = posteriors_at(i, j, l, p, Pg, u, tau)
                F[i, j, l] = clearing_residual(mus, p, gamma)
    return F


def F_flat(x, u, tau, gamma, G):
    return residual_array(x.reshape(G, G, G), u, tau, gamma).reshape(-1)


def nolearning_price(u, tau, gamma):
    G = len(u)
    P0 = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                mus = (logistic(tau * u[i]),
                       logistic(tau * u[j]),
                       logistic(tau * u[l]))
                P0[i, j, l] = clear_price(list(mus), gamma)
    return P0


def phi_map(Pg, u, tau, gamma):
    G = len(u)
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = Pg[i, j, l]
                mus = posteriors_at(i, j, l, p_cur, Pg, u, tau)
                Pnew[i, j, l] = clear_price(list(mus), gamma)
    return Pnew


# ---------------------------------------------------------------
# Picard
# ---------------------------------------------------------------

def solve_picard(G, tau, gamma, umax=2.0, maxiters=2000, abstol=1e-13):
    u = build_grid(G, umax)
    P0 = nolearning_price(u, tau, gamma)
    Pcur = P0.copy()
    history = []
    for _ in range(maxiters):
        Pnew = phi_map(Pcur, u, tau, gamma)
        diff = float(np.abs(Pnew - Pcur).max())
        Pcur = Pnew
        history.append(diff)
        if diff < abstol:
            break
    F = residual_array(Pcur, u, tau, gamma)
    return dict(P_star=Pcur, P0=P0, u=u, residual=F, history=history,
                converged=(history and history[-1] < abstol))


# ---------------------------------------------------------------
# FD Jacobian + Newton with LU
# ---------------------------------------------------------------

def fd_jacobian(x0, u, tau, gamma, G, h=1e-6):
    n = x0.size
    J = np.empty((n, n))
    xp = x0.copy(); xm = x0.copy()
    for k in range(n):
        xp[k] = x0[k] + h
        xm[k] = x0[k] - h
        fp = F_flat(xp, u, tau, gamma, G)
        fm = F_flat(xm, u, tau, gamma, G)
        J[:, k] = (fp - fm) / (2 * h)
        xp[k] = x0[k]; xm[k] = x0[k]
    return J


def solve_newton_lu(G, tau, gamma, umax=2.0, x0=None,
                    maxiters=5, abstol=1e-11, h=1e-6):
    u = build_grid(G, umax)
    P0 = nolearning_price(u, tau, gamma)
    x = x0.copy() if x0 is not None else P0.reshape(-1).copy()
    n = x.size
    Fx = F_flat(x, u, tau, gamma, G)
    fnorm = float(np.abs(Fx).max())
    lam = 0.0
    history = [fnorm]
    t_jac = t_lu = t_solve = t_ls = 0.0

    for it in range(maxiters):
        if fnorm < abstol:
            break
        t0 = time.time()
        J = fd_jacobian(x, u, tau, gamma, G, h)
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
        Ftrial = F_flat(xtrial, u, tau, gamma, G)
        nt = float(np.abs(Ftrial).max())
        tries = 0
        while nt > fnorm and tries < 25:
            alpha *= 0.5
            xtrial = x + alpha * dx
            Ftrial = F_flat(xtrial, u, tau, gamma, G)
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


def peak_rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")
