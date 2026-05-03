"""Shared mpmath helpers for Terminal 4 extractions at dps=100, tol 1e-50.

- Loads converged posterior JSON (G, u_grid, p_grid, mu_strings) at mp precision.
- Bilinear interpolation of mu*(u, p) on the (per-u, varying-p) grid.
- CRRA demand for binary v in {0, 1}.
- Bisection for market clearing on p in (eps, 1-eps).
"""

import json
from typing import Callable, List, Tuple

import mpmath as mp

mp.mp.dps = 100

ZERO = mp.mpf(0)
ONE = mp.mpf(1)
HALF = mp.mpf("0.5")
EPS_PRICE = mp.mpf("1e-50")           # market-clearing bracket epsilon
TOL_BISECT = mp.mpf("1e-50")          # ||excess|| target tolerance
EPS_MU = mp.mpf("1e-90")              # clamp for posterior


def lam(z):
    """Logistic 1/(1+exp(-z)) numerically stable."""
    if z >= 0:
        return ONE / (ONE + mp.exp(-z))
    e = mp.exp(z)
    return e / (ONE + e)


def logit(p):
    return mp.log(p) - mp.log(ONE - p)


def crra_demand_binary(mu, p, gamma):
    """Closed-form CRRA demand x for v in {0,1}: x = (R-1)/((1-p)+R p), R=exp((logit mu - logit p)/gamma)."""
    if mu < EPS_MU:
        mu = EPS_MU
    if mu > ONE - EPS_MU:
        mu = ONE - EPS_MU
    if p < EPS_PRICE or p > ONE - EPS_PRICE:
        return ZERO
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0:
        e = mp.exp(-z)
        return (ONE - e) / ((ONE - p) * e + p)
    e = mp.exp(z)
    return (e - ONE) / ((ONE - p) + p * e)


def bisect_market_clear(excess, a=None, b=None, tol=None, max_iter=2000):
    """Bisection for monotone-decreasing excess(p) on (a, b).

    Excess strictly decreases in p, so excess(a) > 0 > excess(b) generically.
    Stops when (b-a) < tol or |excess(c)| < tol.
    """
    if a is None:
        a = EPS_PRICE
    if b is None:
        b = ONE - EPS_PRICE
    if tol is None:
        tol = TOL_BISECT
    fa = excess(a)
    fb = excess(b)
    if fa <= 0:
        return a
    if fb >= 0:
        return b
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = excess(c)
        if abs(fc) < tol:
            return c
        if fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc
        if (b - a) < tol:
            return (a + b) / 2
    return (a + b) / 2


def load_posterior_mp(path):
    """Load checkpoint JSON; convert u_grid, p_grid, mu_strings to mpmath.mpf."""
    with open(path) as f:
        d = json.load(f)
    u_grid = [mp.mpf(x) for x in d["u_grid"]]
    p_grids = [[mp.mpf(x) for x in row] for row in d["p_grid"]]
    mu_grids = [[mp.mpf(x) for x in row] for row in d["mu_strings"]]
    return d, u_grid, p_grids, mu_grids


def make_mu_interpolator(u_grid, p_grids, mu_grids):
    """Bilinear interpolation of mu*(u, p), all in mpmath."""
    G = len(u_grid)

    def mu_at_u_idx(idx, p):
        p_arr = p_grids[idx]
        mu_arr = mu_grids[idx]
        n = len(p_arr)
        if p <= p_arr[0]:
            return mu_arr[0]
        if p >= p_arr[-1]:
            return mu_arr[-1]
        # Binary search for p in p_arr
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if p_arr[mid] <= p:
                lo = mid
            else:
                hi = mid
        denom = p_arr[lo + 1] - p_arr[lo]
        if denom == 0:
            return mu_arr[lo]
        frac = (p - p_arr[lo]) / denom
        return mu_arr[lo] + frac * (mu_arr[lo + 1] - mu_arr[lo])

    def find_u_bracket(u):
        if u <= u_grid[0]:
            return 0, 1
        if u >= u_grid[-1]:
            return G - 2, G - 1
        lo, hi = 0, G - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if u_grid[mid] <= u:
                lo = mid
            else:
                hi = mid
        return lo, lo + 1

    def interp(u, p):
        i_lo, i_hi = find_u_bracket(u)
        m_lo = mu_at_u_idx(i_lo, p)
        m_hi = mu_at_u_idx(i_hi, p)
        denom = u_grid[i_hi] - u_grid[i_lo]
        if denom == 0:
            return m_lo
        f = (u - u_grid[i_lo]) / denom
        if f < 0:
            f = ZERO
        if f > 1:
            f = ONE
        return m_lo + f * (m_hi - m_lo)

    return interp


def fmt_mp(x, digits=30):
    """Compact mp-faithful string for plotting (default 30 sig digits)."""
    return mp.nstr(x, digits, strip_zeros=False)
