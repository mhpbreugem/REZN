"""Float128 primitives shared across the package.

Pure-numpy float128 (np.longdouble, ~19 decimal digits on x86_64). No numba —
numba does not support extended precision. Slower than f64 numba per-call but
truly f128 throughout: no f64→f128 round-trips and no LAPACK downcast inside
the residual map.

K is read off the input vector lengths (gammas, taus, Ws); the K=3 special-
casing of rezn_het lives only in `contour.py` (which assumes a 2-D slice).
"""
from __future__ import annotations
import numpy as np

DTYPE = np.float128

EPS        = DTYPE("1e-12")          # demand-function endpoint guard
EPS_OUTER  = DTYPE("1e-9")           # solver-wide P clip boundary
ZERO       = DTYPE(0.0)
ONE        = DTYPE(1.0)
HALF       = DTYPE(0.5)
TWO        = DTYPE(2.0)
PI         = DTYPE(np.pi)
BISECT_TOL = DTYPE("1e-25")          # well below f128 precision floor (~1e-19)
BISECT_MAX = 300


def f0(u, tau):
    return np.sqrt(tau / (TWO * PI)) * np.exp(-tau / TWO * (u + HALF) ** 2)


def f1(u, tau):
    return np.sqrt(tau / (TWO * PI)) * np.exp(-tau / TWO * (u - HALF) ** 2)


def logit(p):
    return np.log(p / (ONE - p))


def logistic(x):
    return ONE / (ONE + np.exp(-x))


def demand_crra(mu, p, gamma, W):
    mu_c = mu if mu >= EPS else EPS
    if mu_c > ONE - EPS:
        mu_c = ONE - EPS
    p_c = p if p >= EPS else EPS
    if p_c > ONE - EPS:
        p_c = ONE - EPS
    R = np.exp((logit(mu_c) - logit(p_c)) / gamma)
    return W * (R - ONE) / ((ONE - p_c) + R * p_c)


def clearing_residual(mus, p, gammas, Ws):
    s = ZERO
    for k in range(mus.shape[0]):
        s = s + demand_crra(mus[k], p, gammas[k], Ws[k])
    return s


def clear_price(mus, gammas, Ws):
    """Bisect for the unique p ∈ (EPS_OUTER, 1−EPS_OUTER) clearing the market."""
    a = EPS_OUTER
    b = ONE - EPS_OUTER
    fa = clearing_residual(mus, a, gammas, Ws)
    fb = clearing_residual(mus, b, gammas, Ws)
    if fa == ZERO:
        return a
    if fb == ZERO:
        return b
    for _ in range(BISECT_MAX):
        m = HALF * (a + b)
        fm = clearing_residual(mus, m, gammas, Ws)
        if (b - a) < BISECT_TOL or fm == ZERO:
            return m
        if fa * fm < ZERO:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return HALF * (a + b)


def to_f128(x):
    return np.asarray(x, dtype=DTYPE)


def cast_problem(taus, gammas, Ws):
    """Return (taus, gammas, Ws) all as float128 1-D arrays of equal length."""
    taus = np.asarray(taus, dtype=DTYPE).reshape(-1)
    gammas = np.asarray(gammas, dtype=DTYPE).reshape(-1)
    if np.isscalar(Ws) or (hasattr(Ws, "shape") and Ws.shape == ()):
        Ws = np.full(taus.shape, DTYPE(float(Ws)), dtype=DTYPE)
    else:
        Ws = np.asarray(Ws, dtype=DTYPE).reshape(-1)
    if not (taus.shape == gammas.shape == Ws.shape):
        raise ValueError(
            f"taus/gammas/Ws shape mismatch: {taus.shape}, {gammas.shape}, {Ws.shape}")
    return taus, gammas, Ws


def build_grid(G, umax):
    return np.linspace(DTYPE(-float(umax)), DTYPE(float(umax)), G).astype(DTYPE)
