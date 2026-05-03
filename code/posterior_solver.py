"""Posterior-function fixed-point solver (common-p-grid v3 algorithm).

Solves for mu*(u, p) on a 2D grid where the p-grid is COMMON across
rows (logit-spaced from p_lo_global to p_hi_global). K=3 symmetric
agents (same gamma, same tau, equal wealth). All work float64.

Algorithm:
    Per price level p_j:
        1. mu_col = mu[:, j]                              (no interp)
        2. d[i] = x_crra(mu_col[i], p_j; gamma)           (vector)
        3. d_mono = maximum.accumulate(d) (force monotone if PAVA hasn't)
        4. For each own-signal i_own:
              targets = -d[i_own] - d
              u3* = np.interp(targets, d_mono, u_grid)
              valid = u_min < u3* < u_max
              A_v = sum f_v(u_grid[valid]) * f_v(u3*[valid])
              mu_new[i_own, j] = f1(u_i)*A1 / (f0(u_i)*A0 + f1(u_i)*A1)
    PAVA u-direction at each j; PAVA p-direction at each i.
    Damped Picard with optional Anderson on the active subset.

For warm-starts across tau, the same common p-grid is used (built
once large enough to contain all reasonable equilibrium prices for
the gamma).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression


EPS_PRICE = 1.0e-12
EPS_MU = 1.0e-12


# -------------------- low-level math (vectorised float64) --------------------

def lam(z):
    """Logistic 1/(1+exp(-z)), vectorised, numerically safe."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log(1.0 - p)


def f_signal(u, v: int, tau: float):
    u = np.asarray(u, dtype=np.float64)
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return math.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * d * d)


def crra_demand_arr(mu, p: float, gamma: float, W: float = 1.0):
    """Vectorised CRRA demand at fixed scalar price p."""
    z = (logit(mu) - logit(np.array([p]))[0]) / gamma
    out = np.empty_like(z)
    pos = z >= 0.0
    e_pos = np.exp(-z[pos])
    out[pos] = W * (1.0 - e_pos) / ((1.0 - p) * e_pos + p)
    e_neg = np.exp(z[~pos])
    out[~pos] = W * (e_neg - 1.0) / ((1.0 - p) + p * e_neg)
    return out


def crra_demand_scalar(mu: float, p: float, gamma: float,
                       W: float = 1.0) -> float:
    z = (math.log(mu) - math.log(1.0 - mu)
         - math.log(p) + math.log(1.0 - p)) / gamma
    if z >= 0.0:
        e = math.exp(-z)
        return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = math.exp(z)
    return W * (e - 1.0) / ((1.0 - p) + p * e)


# -------------------- common-p-grid construction --------------------

def common_pgrid_range(u_grid: np.ndarray, tau: float, gamma: float
                       ) -> Tuple[float, float]:
    """Return (p_lo, p_hi) covering all (u_min, ..., u_max) triples
    under the no-learning posteriors.  Logit-spaced p-grid spans
    [p_lo, p_hi] with small margin.
    """
    u_min = float(u_grid[0])
    u_max = float(u_grid[-1])
    mu_min = float(lam(np.array([tau * u_min]))[0])
    mu_max = float(lam(np.array([tau * u_max]))[0])

    # Most extreme prices: all bearish vs all bullish.
    def excess_lo(p):
        return 3.0 * crra_demand_scalar(mu_min, p, gamma)

    def excess_hi(p):
        return 3.0 * crra_demand_scalar(mu_max, p, gamma)

    a, b = EPS_PRICE, 1.0 - EPS_PRICE
    p_lo = brentq(excess_lo, a, b, xtol=1e-14)
    p_hi = brentq(excess_hi, a, b, xtol=1e-14)
    return p_lo, p_hi


def build_common_pgrid(u_grid: np.ndarray, tau: float, gamma: float,
                       G_p: int, margin: float = 0.05) -> np.ndarray:
    p_lo, p_hi = common_pgrid_range(u_grid, tau, gamma)
    lo_l = math.log(p_lo / (1.0 - p_lo))
    hi_l = math.log(p_hi / (1.0 - p_hi))
    span = hi_l - lo_l
    lo_l -= margin * span
    hi_l += margin * span
    ls = np.linspace(lo_l, hi_l, G_p)
    p = 1.0 / (1.0 + np.exp(-ls))
    return np.clip(p, EPS_PRICE, 1.0 - EPS_PRICE)


def init_mu_no_learning(u_grid: np.ndarray, p_grid: np.ndarray,
                        tau: float) -> np.ndarray:
    """mu[i, j] = lam(tau * u_i), independent of p (no-learning seed)."""
    G_u, G_p = u_grid.size, p_grid.size
    return np.tile(lam(tau * u_grid).reshape(-1, 1), (1, G_p))


def init_mu_FR(u_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    """mu[i, j] = p_grid[j], the full-revelation/CARA fixed point."""
    G_u, G_p = u_grid.size, p_grid.size
    return np.tile(p_grid.reshape(1, -1), (G_u, 1))


# -------------------- one Phi step (vectorised, common p-grid) --------------

def phi_step(mu: np.ndarray, p_grid: np.ndarray, u_grid: np.ndarray,
             tau: float, gamma: float, W: float = 1.0,
             min_valid: int = 3
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Bayes update.

    Returns (mu_new, active_mask). Inactive cells keep their old value.
    A cell is inactive when fewer than `min_valid` contour crossings
    landed inside the u-grid.
    """
    G_u, G_p = mu.shape
    mu_new = mu.copy()
    active_mask = np.zeros((G_u, G_p), dtype=bool)

    f1_grid = f_signal(u_grid, 1, tau)
    f0_grid = f_signal(u_grid, 0, tau)

    u_min = u_grid[0]
    u_max = u_grid[-1]

    for j in range(G_p):
        p0 = float(p_grid[j])
        mu_col = np.clip(mu[:, j], EPS_MU, 1.0 - EPS_MU)
        d = crra_demand_arr(mu_col, p0, gamma, W=W)

        # If d is non-monotone, force monotone via cumulative max. This
        # creates flat segments; np.interp will return the right edge of
        # the segment for any target in the flat range, which we treat
        # as an artificial root that gets discounted via density.
        d_mono = np.maximum.accumulate(d)

        if d_mono[-1] - d_mono[0] < 1e-15:
            # Demand essentially constant — no information from price.
            continue

        for i_own in range(G_u):
            d1 = d[i_own]
            targets = -d1 - d  # (G_u,)

            u3_star = np.interp(targets, d_mono, u_grid)
            valid = (u3_star > u_min + 1e-10) & (u3_star < u_max - 1e-10)
            n_valid = int(valid.sum())
            if n_valid < min_valid:
                continue

            f1_u3 = f_signal(u3_star[valid], 1, tau)
            f0_u3 = f_signal(u3_star[valid], 0, tau)
            A1 = float(np.sum(f1_grid[valid] * f1_u3))
            A0 = float(np.sum(f0_grid[valid] * f0_u3))

            f1o = float(f1_grid[i_own])
            f0o = float(f0_grid[i_own])
            num = f1o * A1
            den = f0o * A0 + num
            if den <= 0.0 or not np.isfinite(den):
                continue
            mu_ij = num / den
            mu_ij = max(EPS_MU, min(1.0 - EPS_MU, mu_ij))
            mu_new[i_own, j] = mu_ij
            active_mask[i_own, j] = True

    return mu_new, active_mask


# -------------------- monotonicity projection --------------------

def pava_project(mu_new: np.ndarray, active_mask: np.ndarray,
                 p_grid: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    G_u, G_p = mu_new.shape
    out = mu_new.copy()
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    # Pass 1: u-direction at each price column j
    for j in range(G_p):
        mask = active_mask[:, j]
        if mask.sum() >= 2:
            xs = u_grid[mask]
            ys = out[mask, j]
            try:
                ys_iso = iso.fit_transform(xs, ys)
                out[mask, j] = np.clip(ys_iso, EPS_MU, 1.0 - EPS_MU)
            except Exception:
                pass

    # Pass 2: p-direction at each row i (in logit-p space)
    p_log = logit(p_grid)
    for i in range(G_u):
        mask = active_mask[i, :]
        if mask.sum() >= 2:
            xs = p_log[mask]
            ys = out[i, mask]
            try:
                ys_iso = iso.fit_transform(xs, ys)
                out[i, mask] = np.clip(ys_iso, EPS_MU, 1.0 - EPS_MU)
            except Exception:
                pass

    return out


# -------------------- Anderson type-II step --------------------

def _anderson_step(residuals: List[np.ndarray], iterates: List[np.ndarray],
                   m: int) -> np.ndarray:
    n_hist = min(m, len(residuals))
    if n_hist <= 1:
        return iterates[-1] + residuals[-1]
    F = np.column_stack(residuals[-n_hist:])
    X = np.column_stack(iterates[-n_hist:])
    dF = F[:, 1:] - F[:, :-1]
    rhs = F[:, -1]
    try:
        gamma_coef, *_ = np.linalg.lstsq(dF, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return iterates[-1] + residuals[-1]
    alpha = np.empty(n_hist)
    if n_hist == 1:
        alpha[0] = 1.0
    else:
        alpha[0] = gamma_coef[0]
        for k in range(1, n_hist - 1):
            alpha[k] = gamma_coef[k] - gamma_coef[k - 1]
        alpha[-1] = 1.0 - gamma_coef[-1]
    next_x = (X + F) @ alpha
    return next_x


# -------------------- the iteration --------------------

@dataclass
class SolveResult:
    mu: np.ndarray
    p_grid: np.ndarray
    u_grid: np.ndarray
    F_max: float
    F_med: float
    F_max_active: float
    n_iter: int
    n_active: int
    history: List[dict] = field(default_factory=list)


def solve_posterior_fp(u_grid: np.ndarray, tau: float, gamma: float, *,
                       p_grid: Optional[np.ndarray] = None,
                       G_p: int = 20,
                       mu_init: Optional[np.ndarray] = None,
                       init_kind: str = "FR",     # "FR", "no_learning"
                       max_iter: int = 400,
                       tol: float = 1.0e-12,
                       alpha_init: float = 0.5,
                       alpha_min: float = 0.05,
                       use_pava: bool = True,
                       verbose: bool = False,
                       sticky_mask_after: int = 6,
                       anderson_m: int = 5,
                       anderson_after: int = 8,
                       min_valid: int = 3,
                       margin: float = 0.05,
                       ) -> SolveResult:
    G_u = u_grid.size
    if p_grid is None:
        p_grid = build_common_pgrid(u_grid, tau, gamma, G_p, margin=margin)
    G_p = p_grid.size

    if mu_init is None:
        if init_kind == "FR":
            mu = init_mu_FR(u_grid, p_grid)
        else:
            mu = init_mu_no_learning(u_grid, p_grid, tau)
    else:
        mu = np.clip(mu_init.copy(), EPS_MU, 1.0 - EPS_MU)
    mu = np.clip(mu, EPS_MU, 1.0 - EPS_MU)

    history = []
    alpha = alpha_init
    F_active_prev = float("inf")
    stalls = 0
    persistent_active = np.ones((G_u, G_p), dtype=bool)
    sticky = None

    iter_hist: List[np.ndarray] = []
    res_hist: List[np.ndarray] = []
    eff_mask_for_anderson = None

    t0 = time.perf_counter()
    F_max_act = 0.0
    F_med = 0.0
    for it in range(1, max_iter + 1):
        mu_raw, mask = phi_step(mu, p_grid, u_grid, tau, gamma,
                                min_valid=min_valid)

        persistent_active &= mask
        if it == sticky_mask_after:
            sticky = persistent_active.copy()

        eff_mask = mask if sticky is None else (mask & sticky)

        if use_pava:
            mu_proj = pava_project(mu_raw, eff_mask, p_grid, u_grid)
        else:
            mu_proj = mu_raw.copy()

        if sticky is not None:
            mu_proj = np.where(eff_mask, mu_proj, mu)

        F = (mu_proj - mu) * eff_mask
        F_max_act = float(np.max(np.abs(F))) if eff_mask.any() else 0.0
        F_med = float(np.median(np.abs(F[eff_mask]))) if eff_mask.any() else 0.0
        F_max_all = float(np.max(np.abs(mu_proj - mu)))

        rec = {"iter": it, "F_max": F_max_act, "F_med": F_med,
               "F_max_all": F_max_all, "alpha": alpha,
               "n_active": int(eff_mask.sum()),
               "elapsed_s": time.perf_counter() - t0}
        history.append(rec)
        if verbose:
            print(f"  it={it:3d}  ||F||act={F_max_act:.4e}  med={F_med:.2e}  "
                  f"alpha={alpha:.3f}  active={eff_mask.sum()}/{G_u*G_p}  "
                  f"elapsed={rec['elapsed_s']:.1f}s", flush=True)

        if F_max_act < tol:
            mu = mu_proj
            break

        # Anderson on the active mask (must be stable across iters)
        use_anderson = (sticky is not None and it >= anderson_after
                        and anderson_m > 0)
        if use_anderson:
            if eff_mask_for_anderson is None or \
                    not np.array_equal(eff_mask, eff_mask_for_anderson):
                # Reset Anderson if the active set changes
                eff_mask_for_anderson = eff_mask.copy()
                iter_hist.clear()
                res_hist.clear()
            x_flat = mu[eff_mask]
            phi_flat = mu_proj[eff_mask]
            r_flat = phi_flat - x_flat
            iter_hist.append(x_flat.copy())
            res_hist.append(r_flat.copy())
            if len(iter_hist) > anderson_m + 2:
                iter_hist.pop(0)
                res_hist.pop(0)
            if len(iter_hist) >= 2:
                try:
                    next_flat = _anderson_step(res_hist, iter_hist,
                                               anderson_m)
                    next_flat = np.clip(next_flat, EPS_MU, 1.0 - EPS_MU)
                    mu_step = mu.copy()
                    mu_step[eff_mask] = next_flat
                except Exception:
                    mu_step = np.where(eff_mask,
                                       alpha * mu_proj + (1.0 - alpha) * mu,
                                       mu)
            else:
                mu_step = np.where(eff_mask,
                                   alpha * mu_proj + (1.0 - alpha) * mu,
                                   mu)
        else:
            mu_step = np.where(eff_mask,
                               alpha * mu_proj + (1.0 - alpha) * mu,
                               mu)

        # Stall detection
        if F_max_act >= F_active_prev * 0.95:
            stalls += 1
            if stalls >= 4:
                alpha = max(alpha_min, alpha * 0.7)
                stalls = 0
                if use_anderson:
                    iter_hist.clear()
                    res_hist.clear()
        else:
            stalls = 0
        F_active_prev = F_max_act
        mu = mu_step

    return SolveResult(mu=mu, p_grid=p_grid, u_grid=u_grid.copy(),
                       F_max=float(np.max(np.abs(mu_proj - mu))),
                       F_med=F_med, F_max_active=F_max_act,
                       n_iter=it,
                       n_active=int(eff_mask.sum()),
                       history=history)


# -------------------- 1-R^2 measurement (weighted) --------------------

def measure_weighted_1mR2(mu: np.ndarray, p_grid: np.ndarray,
                          u_grid: np.ndarray, tau: float, gamma: float,
                          ) -> Tuple[float, float, int]:
    """Compute weighted 1-R^2 over all G^3 triples."""
    G = u_grid.size
    p_log = logit(p_grid)

    def mu_at(i_row: int, p: float) -> float:
        lp = math.log(p / (1.0 - p))
        if lp <= p_log[0]:
            return float(mu[i_row, 0])
        if lp >= p_log[-1]:
            return float(mu[i_row, -1])
        return float(np.interp(lp, p_log, mu[i_row]))

    f1 = f_signal(u_grid, 1, tau)
    f0 = f_signal(u_grid, 0, tau)

    n = G * G * G
    Tstar = np.empty(n, dtype=np.float64)
    logit_p = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)

    a = EPS_PRICE
    b = 1.0 - EPS_PRICE
    idx = 0
    for i in range(G):
        ui = u_grid[i]
        for j in range(G):
            uj = u_grid[j]
            for l in range(G):
                ul = u_grid[l]
                T = tau * (ui + uj + ul)

                def excess(p):
                    m1 = mu_at(i, p)
                    m2 = mu_at(j, p)
                    m3 = mu_at(l, p)
                    return (crra_demand_scalar(m1, p, gamma)
                            + crra_demand_scalar(m2, p, gamma)
                            + crra_demand_scalar(m3, p, gamma))

                fa = excess(a)
                fb = excess(b)
                if fa <= 0:
                    p_ree = a
                elif fb >= 0:
                    p_ree = b
                else:
                    try:
                        p_ree = brentq(excess, a, b, xtol=1e-13)
                    except (ValueError, RuntimeError):
                        p_ree = 0.5

                Tstar[idx] = T
                logit_p[idx] = math.log(p_ree / (1.0 - p_ree))
                w = 0.5 * (f1[i] * f1[j] * f1[l]
                           + f0[i] * f0[j] * f0[l])
                weights[idx] = w
                idx += 1

    sw = np.sqrt(weights)
    slope, intercept = np.polyfit(Tstar, logit_p, 1, w=sw)
    pred = slope * Tstar + intercept
    mean_lp = np.average(logit_p, weights=weights)
    var_tot = np.average((logit_p - mean_lp) ** 2, weights=weights)
    var_res = np.average((logit_p - pred) ** 2, weights=weights)
    one_mR2 = float(var_res / var_tot)
    return one_mR2, float(slope), n
