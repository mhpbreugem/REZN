"""2-pass linear-interpolation contour integral.

Given agent k's price slice — a (G, G) array of prices indexed by the OTHER
two agents' signal grid points — find every cell where the bilinear-interpolated
price crosses p_obs and accumulate the joint signal density at that crossing.

This is the K=3-specific kernel (the slice is 2-D). For arbitrary K the slice
becomes (K-1)-D and a different scheme is needed — out of scope here.

Returns (A0, A1) in float128, the integrals against f₀ and f₁ respectively.
"""
from __future__ import annotations
import numpy as np

from .primitives import DTYPE, ZERO, HALF, f0, f1


def contour_integral(slice_, u, tau_A, tau_B, p_obs):
    """Two-pass linear-interp contour sum on a G×G slice.

    slice_[a, b] is the price at (u_A=u[a], u_B=u[b]); axis A uses tau_A,
    axis B uses tau_B.

    Pass A: for each row a, walk along columns and find off-grid crossings
    in u_B (axis-B) — u_A is on-grid.
    Pass B: for each column b, walk along rows and find off-grid crossings
    in u_A.
    """
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
                A0 = A0 + f0(ua, tau_A) * f0(ub, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub, tau_B)
            elif y1 == ZERO and y2 != ZERO:
                ub = u[k]
                A0 = A0 + f0(ua, tau_A) * f0(ub, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub, tau_B)
            elif k == G - 2 and y2 == ZERO:
                ub = u[k + 1]
                A0 = A0 + f0(ua, tau_A) * f0(ub, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub, tau_B)
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
                A0 = A0 + f0(ua, tau_A) * f0(ub_grid, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub_grid, tau_B)
            elif y1 == ZERO and y2 != ZERO:
                ua = u[k]
                A0 = A0 + f0(ua, tau_A) * f0(ub_grid, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub_grid, tau_B)
            elif k == G - 2 and y2 == ZERO:
                ua = u[k + 1]
                A0 = A0 + f0(ua, tau_A) * f0(ub_grid, tau_B)
                A1 = A1 + f1(ua, tau_A) * f1(ub_grid, tau_B)
    return HALF * A0, HALF * A1
