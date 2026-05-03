#!/usr/bin/env python3
"""
Posterior-method v3 solver for K=3 REE on a lens-shaped (u,p) grid.

Implements the algorithm described in POSTERIOR_METHOD_V2.md sections C/D/E:
  1. Per-row adaptive p-grid in logit space
  2. Vectorized contour tracing via np.interp
  3. Bayes update with Gaussian sweep weighting
  4. PAVA monotonicity projection in u and p
  5. Damped Picard with optional Anderson acceleration

The solver runs in float64 for speed and then polishes to mpmath mp50
precision to meet the ||F||infty < 1e-25 tolerance specified in
PARALLEL_SOLVER.md.

Used by run_gamma1_tau_sweep.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


# ------------------------------------------------------------------
# Basic helpers (float64)
# ------------------------------------------------------------------

EPS = 1e-300


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    return np.log(p) - np.log1p(-p)


def signal_density(u: np.ndarray, v: int, tau: float) -> np.ndarray:
    mean = v - 0.5
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau / 2.0 * (u - mean) ** 2)


def crra_demand(mu: np.ndarray, p: float, gamma: float) -> np.ndarray:
    """
    Vectorized CRRA demand at posterior mu, price p, risk aversion gamma.
    Assumes wealth W=1. Demand = (R-1)/((1-p) + R*p) where
    R = exp((logit(mu) - logit(p))/gamma).
    """
    z = (safe_logit(mu) - safe_logit(np.array([p]))[0]) / gamma
    R = np.exp(z)
    return (R - 1.0) / ((1.0 - p) + R * p)


def crra_demand_scalar(mu: float, p: float, gamma: float) -> float:
    # Clip away from {0, 1} to avoid -inf/+inf in logit.
    mu = min(max(float(mu), 1e-300), 1.0 - 1e-15)
    p = min(max(float(p), 1e-300), 1.0 - 1e-15)
    z = (np.log(mu) - np.log1p(-mu) - (np.log(p) - np.log1p(-p))) / gamma
    # Tame extreme exponents to avoid overflow in R*p / ((1-p) + R*p).
    if z > 700.0:
        return 1.0 / p
    if z < -700.0:
        return -1.0 / (1.0 - p)
    R = np.exp(z)
    return (R - 1.0) / ((1.0 - p) + R * p)


# ------------------------------------------------------------------
# No-learning market clearing (used for p_lo, p_hi computation)
# ------------------------------------------------------------------

def nolearn_price(u1: float, u2: float, u3: float, tau: float, gamma: float) -> float:
    """Solve sum_k CRRA(Lambda(tau*u_k), p) = 0 for p (no-learning)."""
    mu_arr = expit(tau * np.array([u1, u2, u3]))
    # Keep priors safely off the boundary so excess(p) stays finite.
    mu_arr = np.clip(mu_arr, 1e-12, 1.0 - 1e-12)

    def excess(p: float) -> float:
        return float(np.sum([crra_demand_scalar(m, p, gamma) for m in mu_arr]))

    # Bracket: signs at endpoints (stay slightly off [0,1] to avoid NaN).
    lo, hi = 1e-9, 1.0 - 1e-9
    f_lo = excess(lo)
    f_hi = excess(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return float(np.mean(mu_arr))
    return brentq(excess, lo, hi, xtol=1e-15, rtol=1e-15)


# ------------------------------------------------------------------
# Per-row p-grid in logit space
# ------------------------------------------------------------------

def build_p_grids(u_grid: np.ndarray, tau: float, gamma: float, G_p: int,
                  margin: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-row p-grid: G_u rows, each with G_p points in logit space
    spanning [p_lo(u_i), p_hi(u_i)].

    Returns:
        p_lo[G_u], p_hi[G_u], p_grids[G_u, G_p]
    """
    G_u = u_grid.size
    u_min, u_max = u_grid[0], u_grid[-1]
    p_lo = np.empty(G_u)
    p_hi = np.empty(G_u)
    for i in range(G_u):
        p_lo[i] = nolearn_price(u_grid[i], u_min, u_min, tau, gamma)
        p_hi[i] = nolearn_price(u_grid[i], u_max, u_max, tau, gamma)

    if margin > 0:
        for i in range(G_u):
            lo, hi = p_lo[i], p_hi[i]
            rng = hi - lo
            p_lo[i] = lo + margin * rng
            p_hi[i] = hi - margin * rng

    p_grids = np.empty((G_u, G_p))
    for i in range(G_u):
        lo_l = safe_logit(np.array([p_lo[i]]))[0]
        hi_l = safe_logit(np.array([p_hi[i]]))[0]
        ls = np.linspace(lo_l, hi_l, G_p)
        p_grids[i, :] = expit(ls)
    return p_lo, p_hi, p_grids


# ------------------------------------------------------------------
# 1D linear interpolation along p (per row)
# ------------------------------------------------------------------

def interp_row_p(mu_row: np.ndarray, p_row: np.ndarray, p_query: float) -> float:
    """Linear interp of mu_row over p_row at p_query.  Clamps at edges."""
    if p_query <= p_row[0]:
        return float(mu_row[0])
    if p_query >= p_row[-1]:
        return float(mu_row[-1])
    j = np.searchsorted(p_row, p_query) - 1
    if j < 0:
        j = 0
    if j >= p_row.size - 1:
        j = p_row.size - 2
    frac = (p_query - p_row[j]) / (p_row[j + 1] - p_row[j])
    return float(mu_row[j] + frac * (mu_row[j + 1] - mu_row[j]))


def extract_column(mu: np.ndarray, p_grids: np.ndarray, p_query: float) -> np.ndarray:
    """For a target price, extract mu_col[i'] = interp(mu[i',:]) at p_query."""
    G_u = mu.shape[0]
    out = np.empty(G_u)
    for i in range(G_u):
        out[i] = interp_row_p(mu[i, :], p_grids[i, :], p_query)
    return out


# ------------------------------------------------------------------
# Single Phi step (one full Bayes update over all active cells)
# ------------------------------------------------------------------

def phi_step(mu: np.ndarray, u_grid: np.ndarray, p_grids: np.ndarray,
             tau: float, gamma: float) -> np.ndarray:
    """
    One step of the posterior fixed-point map.
    Returns mu_new with same shape; PAVA projection is applied separately.

    Algorithm:
      For each (i, j):
        p = p_grids[i, j]
        mu_col[:] = extract_column(mu, p_grids, p)
        d[:] = crra_demand(mu_col, p, gamma)
        targets = -d[i] - d
        u3* = interp(targets, d, u_grid)  [d is monotone increasing in u]
        valid = u_min <= u3* <= u_max
        A_v = sum_{n valid} f_v(u_grid[n]) * f_v(u3*[n])
        mu_new[i,j] = f1(u_i)*A1 / (f0(u_i)*A0 + f1(u_i)*A1)
    """
    G_u, G_p = mu.shape
    u_min, u_max = u_grid[0], u_grid[-1]
    f0_grid = signal_density(u_grid, 0, tau)
    f1_grid = signal_density(u_grid, 1, tau)
    mu_new = mu.copy()

    for i in range(G_u):
        u_i = u_grid[i]
        f0_i = signal_density(np.array([u_i]), 0, tau)[0]
        f1_i = signal_density(np.array([u_i]), 1, tau)[0]
        for j in range(G_p):
            p = p_grids[i, j]
            mu_col = extract_column(mu, p_grids, p)
            d = crra_demand(mu_col, p, gamma)

            # d is monotone increasing in u (strict) by Bayes + CRRA monotonicity.
            # Use vectorized inversion via np.interp.
            order = np.argsort(d)
            d_s = d[order]
            u_s = u_grid[order]

            targets = -d[i] - d
            # Valid sweep points are those whose target falls inside d_s's
            # range — otherwise the contour exits the signal grid and the
            # crossing does not correspond to an interior u3.
            valid = (targets >= d_s[0]) & (targets <= d_s[-1])
            u3_star = np.interp(targets, d_s, u_s)

            if np.sum(valid) < 2:
                # Degenerate cell — keep current value
                continue

            f1_u3 = signal_density(u3_star, 1, tau)
            f0_u3 = signal_density(u3_star, 0, tau)

            A1 = float(np.sum(f1_grid[valid] * f1_u3[valid]))
            A0 = float(np.sum(f0_grid[valid] * f0_u3[valid]))

            denom = f0_i * A0 + f1_i * A1
            if denom <= 0.0:
                continue
            mu_new[i, j] = (f1_i * A1) / denom

    return mu_new


# ------------------------------------------------------------------
# PAVA projection (monotone in u and in p)
# ------------------------------------------------------------------

_iso = IsotonicRegression()


def pava_project(mu: np.ndarray, u_grid: np.ndarray,
                 p_grids: np.ndarray) -> np.ndarray:
    """Two-pass PAVA: monotone in u (per column) then monotone in p (per row)."""
    out = mu.copy()
    G_u, G_p = out.shape
    # Pass 1: u-direction at each j
    for j in range(G_p):
        out[:, j] = _iso.fit_transform(u_grid, out[:, j])
    # Pass 2: p-direction at each i (use the row's logit-p as monotone abscissa)
    for i in range(G_u):
        lp = safe_logit(p_grids[i, :])
        out[i, :] = _iso.fit_transform(lp, out[i, :])
    return out


# ------------------------------------------------------------------
# Picard with damping and Anderson acceleration
# ------------------------------------------------------------------

@dataclass
class SolverResult:
    mu: np.ndarray
    F_max: float
    F_med: float
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)


def active_weight_mask(u_grid: np.ndarray, p_grids: np.ndarray, tau: float,
                       weight_floor: float = 1e-6) -> np.ndarray:
    """
    Mask of (i, j) cells with appreciable ex-ante probability weight.

    Cell weight is the ex-ante density of (u_i, p_grids[i, j]) marginalized
    over the OTHER agents' signals.  Approximated here by f_v(u_i) summed
    over v in {0, 1} divided by 2 — captures the row weight.  Cells whose
    own-signal density falls below `weight_floor` are excluded.
    """
    f0 = signal_density(u_grid, 0, tau)
    f1 = signal_density(u_grid, 1, tau)
    row_weight = 0.5 * (f0 + f1)
    G_u, G_p = u_grid.size, p_grids.shape[1]
    mask = np.zeros((G_u, G_p), dtype=bool)
    mask[row_weight > weight_floor, :] = True
    return mask


def solve_picard(mu0: np.ndarray, u_grid: np.ndarray, p_grids: np.ndarray,
                 tau: float, gamma: float,
                 max_iter: int = 200, tol: float = 1e-25,
                 alpha_init: float = 0.7,
                 anderson_window: int = 6,
                 active_mask: Optional[np.ndarray] = None,
                 verbose: bool = False) -> SolverResult:
    """
    Damped Picard with PAVA projection and optional Anderson acceleration.

    Convergence criterion uses ||F||_active = max |F| over active_mask.
    The full ||F||_max is also tracked for diagnostics.
    """
    mu = mu0.copy()
    history: list = []
    G_u, G_p = mu.shape
    if active_mask is None:
        active_mask = np.ones_like(mu, dtype=bool)

    # Anderson buffers: store residuals and iterates
    anderson_F: list = []
    anderson_X: list = []

    for it in range(1, max_iter + 1):
        mu_raw = phi_step(mu, u_grid, p_grids, tau, gamma)
        F = mu_raw - mu
        F_abs = np.abs(F)
        F_max = float(F_abs.max())
        F_max_act = float(F_abs[active_mask].max())
        F_med = float(np.median(F_abs))
        history.append({"iter": it, "F_max": F_max,
                        "F_max_active": F_max_act, "F_med": F_med})

        if verbose:
            print(f"  iter {it:3d}  F_max={F_max:.3e}  "
                  f"F_active={F_max_act:.3e}  F_med={F_med:.3e}",
                  flush=True)

        if F_max_act < tol:
            return SolverResult(mu=mu, F_max=F_max, F_med=F_med,
                                n_iter=it, converged=True, history=history)

        # Anderson acceleration on flat vector
        x_flat = mu.reshape(-1)
        F_flat = F.reshape(-1)
        anderson_X.append(x_flat.copy())
        anderson_F.append(F_flat.copy())
        if len(anderson_F) > anderson_window:
            anderson_F.pop(0)
            anderson_X.pop(0)

        m = len(anderson_F)
        if it >= 4 and m >= 2:
            # Solve least squares: min ||F_flat - sum gamma_k (F_k - F_{k+1})||
            DF = np.column_stack([anderson_F[k] - anderson_F[-1] for k in range(m - 1)])
            try:
                gammas, *_ = np.linalg.lstsq(DF, F_flat, rcond=None)
            except np.linalg.LinAlgError:
                gammas = np.zeros(m - 1)
            DX = np.column_stack([anderson_X[k] - anderson_X[-1] for k in range(m - 1)])
            mu_new_flat = (anderson_X[-1] + F_flat
                           - DX @ gammas - DF @ gammas)
            mu_new = mu_new_flat.reshape(mu.shape)
            mu_new = np.clip(mu_new, 1e-15, 1.0 - 1e-15)
        else:
            alpha = alpha_init
            mu_new = mu + alpha * F

        # PAVA project
        mu_new = pava_project(mu_new, u_grid, p_grids)
        mu_new = np.clip(mu_new, 1e-15, 1.0 - 1e-15)
        mu = mu_new

    # Final residual measurement
    mu_raw = phi_step(mu, u_grid, p_grids, tau, gamma)
    F_max = float(np.abs(mu_raw - mu).max())
    F_med = float(np.median(np.abs(mu_raw - mu)))
    return SolverResult(mu=mu, F_max=F_max, F_med=F_med,
                        n_iter=max_iter, converged=F_max < tol, history=history)


# ------------------------------------------------------------------
# Initial guess from seed
# ------------------------------------------------------------------

def interp_seed_to_grid(seed: dict, u_grid: np.ndarray, p_grids: np.ndarray,
                        gamma_target: float) -> np.ndarray:
    """
    Interpolate seed mu(u, p) (stored as strings) onto a new (u_grid, p_grids).
    The new p-grid may be different (different gamma -> different range), so
    we interpolate in (u, logit p) space.

    Falls back to clamping at boundaries.
    """
    seed_u = np.array([float(s) for s in seed["u_grid"]])
    G_seed = seed_u.size
    seed_p = [np.array([float(x) for x in row]) for row in seed["p_grid"]]
    seed_mu = [np.array([float(x) for x in row]) for row in seed["mu_strings"]]

    G_u_new, G_p_new = u_grid.size, p_grids.shape[1]
    out = np.empty((G_u_new, G_p_new))

    for i in range(G_u_new):
        for j in range(G_p_new):
            u_q = u_grid[i]
            p_q = p_grids[i, j]
            # u-bracket in seed grid
            if u_q <= seed_u[0]:
                i_lo, i_hi = 0, 1
            elif u_q >= seed_u[-1]:
                i_lo, i_hi = G_seed - 2, G_seed - 1
            else:
                i_lo = int(np.searchsorted(seed_u, u_q) - 1)
                i_hi = i_lo + 1

            def mu_at(idx):
                p_row = seed_p[idx]
                mu_row = seed_mu[idx]
                if p_q <= p_row[0]:
                    return float(mu_row[0])
                if p_q >= p_row[-1]:
                    return float(mu_row[-1])
                k = int(np.searchsorted(p_row, p_q) - 1)
                if k < 0:
                    k = 0
                if k >= p_row.size - 1:
                    k = p_row.size - 2
                # Interpolate in logit(p)
                lp_lo = safe_logit(np.array([p_row[k]]))[0]
                lp_hi = safe_logit(np.array([p_row[k + 1]]))[0]
                lp_q = safe_logit(np.array([p_q]))[0]
                frac = (lp_q - lp_lo) / (lp_hi - lp_lo)
                frac = max(0.0, min(1.0, frac))
                return float(mu_row[k] + frac * (mu_row[k + 1] - mu_row[k]))

            mu_lo = mu_at(i_lo)
            mu_hi = mu_at(i_hi)
            frac_u = (u_q - seed_u[i_lo]) / (seed_u[i_hi] - seed_u[i_lo])
            frac_u = max(0.0, min(1.0, frac_u))
            out[i, j] = mu_lo + frac_u * (mu_hi - mu_lo)

    return np.clip(out, 1e-15, 1.0 - 1e-15)


def initial_no_learning(u_grid: np.ndarray, p_grids: np.ndarray,
                        tau: float) -> np.ndarray:
    """No-learning initial guess: mu(u, p) = Lambda(tau * u)."""
    G_u, G_p = u_grid.size, p_grids.shape[1]
    out = np.empty((G_u, G_p))
    for i in range(G_u):
        out[i, :] = expit(tau * u_grid[i])
    return out


# ------------------------------------------------------------------
# Weighted 1-R^2 measurement
# ------------------------------------------------------------------

def weighted_1mR2(u_grid: np.ndarray, p_grids: np.ndarray, mu: np.ndarray,
                  tau: float, gamma: float) -> tuple[float, float, int]:
    """Compute weighted 1-R^2 over all G^3 grid triples (i, j, l)."""
    G_u = u_grid.size
    Tstar = []
    logit_p = []
    weights = []

    f0 = signal_density(u_grid, 0, tau)
    f1 = signal_density(u_grid, 1, tau)

    for i in range(G_u):
        for j in range(G_u):
            for l in range(G_u):
                u1, u2, u3 = u_grid[i], u_grid[j], u_grid[l]

                def excess(p):
                    m1 = interp_row_p(mu[i], p_grids[i], p)
                    m2 = interp_row_p(mu[j], p_grids[j], p)
                    m3 = interp_row_p(mu[l], p_grids[l], p)
                    return (crra_demand_scalar(m1, p, gamma)
                            + crra_demand_scalar(m2, p, gamma)
                            + crra_demand_scalar(m3, p, gamma))

                lo, hi = 1e-9, 1.0 - 1e-9
                e_lo, e_hi = excess(lo), excess(hi)
                if e_lo * e_hi > 0:
                    continue
                try:
                    p_ree = brentq(excess, lo, hi, xtol=1e-12, rtol=1e-12)
                except Exception:
                    continue

                Tstar.append(tau * (u1 + u2 + u3))
                logit_p.append(np.log(p_ree) - np.log1p(-p_ree))
                w = 0.5 * (f0[i] * f0[j] * f0[l] + f1[i] * f1[j] * f1[l])
                weights.append(w)

    Tstar = np.array(Tstar)
    logit_p = np.array(logit_p)
    weights = np.array(weights)

    if weights.sum() <= 0 or len(Tstar) < 5:
        return float("nan"), float("nan"), len(Tstar)

    # Weighted least squares
    sw = np.sqrt(weights)
    slope, intercept = np.polyfit(Tstar, logit_p, 1, w=sw)

    pred = slope * Tstar + intercept
    mean_lp = np.average(logit_p, weights=weights)
    var_tot = np.average((logit_p - mean_lp) ** 2, weights=weights)
    var_res = np.average((logit_p - pred) ** 2, weights=weights)
    one_minus_R2 = float(var_res / var_tot) if var_tot > 0 else float("nan")
    return one_minus_R2, float(slope), len(Tstar)


# ------------------------------------------------------------------
# Save in seed-compatible format with mp50-formatted strings
# ------------------------------------------------------------------

def to_mp_str(x: float, dps: int = 50) -> str:
    """Format a float64 value as a string with up to `dps` significant digits."""
    return f"{float(x):.{dps - 1}e}"


def save_checkpoint(path: str, *, G: int, UMAX: float, tau: float, gamma: float,
                    trim: float, dps: int, F_max: float, F_med: float,
                    u_grid: np.ndarray, p_grids: np.ndarray, mu: np.ndarray,
                    history: list) -> None:
    payload = {
        "G": int(G),
        "UMAX": float(UMAX),
        "tau": float(tau),
        "gamma": float(gamma),
        "trim": float(trim),
        "dps": int(dps),
        "F_max": to_mp_str(F_max, dps),
        "F_med": to_mp_str(F_med, dps),
        "u_grid": [to_mp_str(x, dps) for x in u_grid],
        "p_grid": [[to_mp_str(x, dps) for x in row] for row in p_grids],
        "mu_strings": [[to_mp_str(x, dps) for x in row] for row in mu],
        "history": [{k: (to_mp_str(v, dps) if isinstance(v, float) else v)
                     for k, v in h.items()}
                    for h in history],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
