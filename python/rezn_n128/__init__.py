"""rezn_n128 — strict float128 K=3 REE solver.

Public API:
  solve(...)         — end-to-end Picard → LM → TSVD pipeline
  phi_map, residual_array, nolearning_seed
  picard_adaptive, build_fd_jacobian, lm_solve, tsvd_solve
  load, save         — pickle I/O with metadata
  one_minus_R2

The package assumes K = number of agents = 3 (the contour kernel works on
2-D price slices). taus/gammas/Ws lengths set K; mismatches against the
contour assumption raise NotImplementedError.
"""
from .primitives import DTYPE, EPS_OUTER, build_grid, cast_problem, to_f128
from .phi import phi_map, residual_array, fixed_point_residual, nolearning_seed
from .picard import picard_adaptive
from .newton import build_fd_jacobian, lm_solve, tsvd_solve
from .diagnostics import analyse, pretty
from .io import load, save
from .solver import solve, one_minus_R2

__all__ = [
    "DTYPE", "EPS_OUTER", "build_grid", "cast_problem", "to_f128",
    "phi_map", "residual_array", "fixed_point_residual", "nolearning_seed",
    "picard_adaptive",
    "build_fd_jacobian", "lm_solve", "tsvd_solve",
    "analyse", "pretty",
    "load", "save",
    "solve", "one_minus_R2",
]
