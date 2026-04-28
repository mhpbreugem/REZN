"""Per-agent posterior μ_k via linear-interp contour integral.

The slice for agent k is extracted with np.moveaxis so the code is K-agnostic
in expression, even though `contour_integral` itself assumes K-1 = 2 (i.e. K=3).
"""
from __future__ import annotations
import numpy as np

from .primitives import DTYPE, ZERO, HALF, f0, f1, logistic
from .contour import contour_integral


def agent_posterior(k, idx, p_obs, P, u, taus):
    """Posterior μ_k for agent k at grid cell idx, observing price p_obs.

    idx: K-tuple of grid indices (one per agent)
    P:   K-D price tensor of shape (G,)*K
    u:   1-D grid (length G), float128
    taus: 1-D precision vector of length K, float128
    """
    K = P.ndim
    # Slice for agent k: drop her own axis. moveaxis(P, k, 0)[idx[k]] is a
    # (K-1)-D array indexed by the OTHER agents' signals in their original
    # order (k removed).
    slice_ = np.moveaxis(P, k, 0)[idx[k]]
    taus_other = np.delete(taus, k)
    u_own = u[idx[k]]
    tau_own = taus[k]

    # contour_integral assumes 2-D slice → K=3
    if slice_.ndim != 2:
        raise NotImplementedError(
            f"contour_integral requires K=3 (2-D slice); got slice.ndim={slice_.ndim}")
    tau_A, tau_B = taus_other[0], taus_other[1]
    A0, A1 = contour_integral(slice_, u, tau_A, tau_B, p_obs)

    f0o = f0(u_own, tau_own)
    f1o = f1(u_own, tau_own)
    num = f1o * A1
    den = f0o * A0 + num
    if den <= ZERO:
        # No crossings AND degenerate density: fall back to single-signal
        # logistic. Matches rezn_het._agent_posterior; rezn_lin128 returned
        # 0.5 here, which is wrong (drops the agent's own signal).
        return logistic(tau_own * u_own)
    return num / den


def posteriors_at(idx, p_obs, P, u, taus):
    K = P.ndim
    out = np.empty(K, dtype=DTYPE)
    for k in range(K):
        out[k] = agent_posterior(k, idx, p_obs, P, u, taus)
    return out
