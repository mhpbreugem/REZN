"""Φ map and residual array (full f128).

Φ(P)[idx] = clear_price(posteriors_at(idx, P[idx], P, u, taus), gammas, Ws)
F(P) = P - Φ(P)
"""
from __future__ import annotations
import numpy as np

from .primitives import DTYPE, clear_price, clearing_residual, logistic
from .posterior import posteriors_at


def phi_map(P, u, taus, gammas, Ws):
    Pnew = np.empty_like(P)
    for idx in np.ndindex(*P.shape):
        mus = posteriors_at(idx, P[idx], P, u, taus)
        Pnew[idx] = clear_price(mus, gammas, Ws)
    return Pnew


def residual_array(P, u, taus, gammas, Ws):
    """Cell-by-cell market-clearing residual at the agents' posteriors.

    This is NOT the fixed-point residual P − Φ(P). It is the demand imbalance
    at the *current* price P[idx] given posteriors derived from P. Used by
    diagnose_v7 to compare with Finf.
    """
    F = np.empty_like(P)
    for idx in np.ndindex(*P.shape):
        p = P[idx]
        mus = posteriors_at(idx, p, P, u, taus)
        F[idx] = clearing_residual(mus, p, gammas, Ws)
    return F


def fixed_point_residual(P, u, taus, gammas, Ws):
    """F = P − Φ(P)."""
    return P - phi_map(P, u, taus, gammas, Ws)


def nolearning_seed(u, taus, gammas, Ws):
    """No-learning P0: each agent uses logistic(τ_k · u_k) as posterior, then
    the market clears. Same construction as rezn_het._nolearning_price."""
    K = taus.shape[0]
    G = u.shape[0]
    shape = (G,) * K
    P0 = np.empty(shape, dtype=DTYPE)
    for idx in np.ndindex(*shape):
        mus = np.empty(K, dtype=DTYPE)
        for k in range(K):
            mus[k] = logistic(taus[k] * u[idx[k]])
        P0[idx] = clear_price(mus, gammas, Ws)
    return P0
