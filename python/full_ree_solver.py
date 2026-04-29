#!/usr/bin/env python3
"""Self-contained contour fixed-point solver for the three-agent REE.

The implementation follows ``contour.md`` and uses only NumPy.  It is meant to
be reproducible on a bare cloud worker: no SciPy, no cached tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1.0e-10


def logistic(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z)
    out = np.empty_like(z_arr, dtype=float)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    if np.isscalar(z):
        return float(out)
    return out


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p_arr = np.clip(np.asarray(p), EPS, 1.0 - EPS)
    out = np.log(p_arr / (1.0 - p_arr))
    if np.isscalar(p):
        return float(out)
    return out


def signal_density(u: float | np.ndarray, v: int, tau: float) -> float | np.ndarray:
    mean = v - 0.5
    return math.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


def signal_density_scalar(u: float, v: int, tau: float) -> float:
    mean = v - 0.5
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * (u - mean) ** 2)


def crra_demand(mu: float, p: float, gamma: float, wealth: float = 1.0) -> float:
    r = math.exp((logit(mu) - logit(p)) / gamma)
    return wealth * (r - 1.0) / ((1.0 - p) + r * p)


def clear_market(mus: Iterable[float], gamma: float) -> float:
    """Solve sum_k x_k(mu_k, p)=0 by bisection on (0,1)."""
    mus = tuple(float(np.clip(mu, EPS, 1.0 - EPS)) for mu in mus)

    def excess(p: float) -> float:
        return sum(crra_demand(mu, p, gamma) for mu in mus)

    lo, hi = 1.0e-8, 1.0 - 1.0e-8
    flo, fhi = excess(lo), excess(hi)
    if flo < 0.0 or fhi > 0.0:
        raise RuntimeError(f"market-clearing bracket failed: f(lo)={flo}, f(hi)={fhi}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = excess(mid)
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def revelation_deficit(P: np.ndarray, u: np.ndarray, tau: float) -> float:
    """Weighted 1-R^2 of logit(P) on T*=tau*(u1+u2+u3)."""
    U1, U2, U3 = np.meshgrid(u, u, u, indexing="ij")
    tstar = tau * (U1 + U2 + U3)
    y = logit(P)
    w1 = signal_density(U1, 1, tau) * signal_density(U2, 1, tau) * signal_density(U3, 1, tau)
    w0 = signal_density(U1, 0, tau) * signal_density(U2, 0, tau) * signal_density(U3, 0, tau)
    w = 0.5 * (w1 + w0)
    mask = (P > 1.0e-5) & (P < 1.0 - 1.0e-5)
    y = y[mask].ravel()
    x = tstar[mask].ravel()
    w = w[mask].ravel()
    w = w / w.sum()
    xbar = float(np.sum(w * x))
    ybar = float(np.sum(w * y))
    vx = float(np.sum(w * (x - xbar) ** 2))
    vy = float(np.sum(w * (y - ybar) ** 2))
    cxy = float(np.sum(w * (x - xbar) * (y - ybar)))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (cxy * cxy) / (vx * vy))


def no_learning_prices(u: np.ndarray, tau: float, gamma: float) -> np.ndarray:
    P = np.empty((len(u), len(u), len(u)), dtype=float)
    priors = logistic(tau * u)
    for i, mu1 in enumerate(priors):
        for j, mu2 in enumerate(priors):
            for k, mu3 in enumerate(priors):
                P[i, j, k] = clear_market((mu1, mu2, mu3), gamma)
    return P


def full_revelation_prices(u: np.ndarray, tau: float) -> np.ndarray:
    U1, U2, U3 = np.meshgrid(u, u, u, indexing="ij")
    return logistic(tau * (U1 + U2 + U3))


def interpolate_price_array(P: np.ndarray, old_grid: np.ndarray, new_grid: np.ndarray) -> np.ndarray:
    """Trilinearly prolong a coarse price array to a finer tensor grid."""
    tmp0 = np.empty((len(new_grid), P.shape[1], P.shape[2]), dtype=float)
    for j in range(P.shape[1]):
        for k in range(P.shape[2]):
            tmp0[:, j, k] = np.interp(new_grid, old_grid, P[:, j, k])

    tmp1 = np.empty((len(new_grid), len(new_grid), P.shape[2]), dtype=float)
    for i in range(len(new_grid)):
        for k in range(P.shape[2]):
            tmp1[i, :, k] = np.interp(new_grid, old_grid, tmp0[i, :, k])

    out = np.empty((len(new_grid), len(new_grid), len(new_grid)), dtype=float)
    for i in range(len(new_grid)):
        for j in range(len(new_grid)):
            out[i, j, :] = np.interp(new_grid, old_grid, tmp1[i, j, :])
    return np.clip(out, 1.0e-8, 1.0 - 1.0e-8)


def load_seed_array(path: Path, grid: np.ndarray) -> np.ndarray:
    data = np.load(path)
    old_grid = np.asarray(data["grid"], dtype=float)
    old_P = np.asarray(data["P"], dtype=float)
    if old_P.shape == (len(grid), len(grid), len(grid)) and np.allclose(old_grid, grid):
        return old_P
    return interpolate_price_array(old_P, old_grid, grid)


def _axis_crossings(values: np.ndarray, grid: np.ndarray, target: float) -> list[float]:
    roots: list[float] = []
    diff = values - target
    exact = np.flatnonzero(np.abs(diff) < 1.0e-12)
    roots.extend(float(grid[int(a)]) for a in exact)
    signs = diff[:-1] * diff[1:]
    for a in np.flatnonzero(signs < 0.0):
        denom = float(values[a + 1] - values[a])
        if abs(denom) < 1.0e-14:
            root = float(grid[a + 1])
        else:
            root = float(grid[a] + (target - values[a]) * (grid[a + 1] - grid[a]) / denom)
        roots.append(root)
    roots.sort()
    # Deduplicate roots created when the target hits a grid point exactly.
    out: list[float] = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1.0e-10:
            out.append(r)
    return out


def contour_evidence(slice2d: np.ndarray, grid: np.ndarray, p: float, tau: float) -> tuple[float, float]:
    sums = np.zeros(2, dtype=float)
    hits = 0

    # Pass A: first coordinate on grid, second coordinate off grid.
    for a, ua in enumerate(grid):
        f0a = signal_density_scalar(float(ua), 0, tau)
        f1a = signal_density_scalar(float(ua), 1, tau)
        for ub in _axis_crossings(slice2d[a, :], grid, p):
            sums[0] += f0a * signal_density_scalar(ub, 0, tau)
            sums[1] += f1a * signal_density_scalar(ub, 1, tau)
            hits += 1

    # Pass B: second coordinate on grid, first coordinate off grid.
    for b, ub in enumerate(grid):
        f0b = signal_density_scalar(float(ub), 0, tau)
        f1b = signal_density_scalar(float(ub), 1, tau)
        for ua in _axis_crossings(slice2d[:, b], grid, p):
            sums[0] += signal_density_scalar(ua, 0, tau) * f0b
            sums[1] += signal_density_scalar(ua, 1, tau) * f1b
            hits += 1

    if hits == 0:
        return 1.0, 1.0
    return float(sums[0] / hits), float(sums[1] / hits)


def posterior_from_slice(slice2d: np.ndarray, own_u: float, grid: np.ndarray, p: float, tau: float) -> float:
    a0, a1 = contour_evidence(slice2d, grid, p, tau)
    f0 = signal_density(own_u, 0, tau) * a0
    f1 = signal_density(own_u, 1, tau) * a1
    denom = f0 + f1
    if denom <= 0.0:
        return 0.5
    return float(np.clip(f1 / denom, EPS, 1.0 - EPS))


def symmetrize(P: np.ndarray) -> np.ndarray:
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    return sum(np.transpose(P, axes=p) for p in perms) / len(perms)


def phi(P: np.ndarray, grid: np.ndarray, tau: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    G = len(grid)
    P_new = np.empty_like(P)
    posteriors = np.empty((3, G, G, G), dtype=float)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = float(P[i, j, k])
                mu1 = posterior_from_slice(P[i, :, :], grid[i], grid, p, tau)
                mu2 = posterior_from_slice(P[:, j, :], grid[j], grid, p, tau)
                mu3 = posterior_from_slice(P[:, :, k], grid[k], grid, p, tau)
                posteriors[:, i, j, k] = (mu1, mu2, mu3)
                P_new[i, j, k] = clear_market((mu1, mu2, mu3), gamma)
    return symmetrize(P_new), posteriors


def residual_only(P: np.ndarray, grid: np.ndarray, tau: float, gamma: float) -> np.ndarray:
    candidate, _ = phi(P, grid, tau, gamma)
    return (candidate - P).ravel()


def gmres(
    matvec,
    b: np.ndarray,
    restart: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Small restarted GMRES implementation for matrix-free Newton steps."""
    n = b.size
    x = np.zeros(n, dtype=float)
    beta0 = float(np.linalg.norm(b))
    if beta0 == 0.0:
        return x, 0.0, 0

    total = 0
    residual = beta0
    while total < max_iter and residual > tol:
        r = b - matvec(x)
        beta = float(np.linalg.norm(r))
        if beta <= tol:
            return x, beta, total

        m = min(restart, max_iter - total)
        V = np.zeros((n, m + 1), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        V[:, 0] = r / beta
        e1 = np.zeros(m + 1, dtype=float)
        e1[0] = beta
        y = np.zeros(0, dtype=float)
        used = 0

        for j in range(m):
            w = matvec(V[:, j])
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1.0e-14:
                V[:, j + 1] = w / H[j + 1, j]
            used = j + 1
            y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], e1[: j + 2], rcond=None)
            residual = float(np.linalg.norm(e1[: j + 2] - H[: j + 2, : j + 1] @ y))
            if residual <= tol:
                break

        x = x + V[:, :used] @ y
        total += used

    return x, residual, total


@dataclass
class SolveResult:
    P: np.ndarray
    posteriors: np.ndarray
    residual_inf: float
    iterations: int
    converged: bool
    history: list[dict[str, float]]


def fixed_point_residual(P: np.ndarray, grid: np.ndarray, tau: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    candidate, post = phi(P, grid, tau, gamma)
    return candidate - P, post


def gmres(
    matvec,
    rhs: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Small unrestarted GMRES for matrix-free Newton steps."""
    beta = float(np.linalg.norm(rhs))
    if beta == 0.0:
        return np.zeros_like(rhs), 0.0, 0

    q_vectors = [rhs / beta]
    hess = np.zeros((max_iter + 1, max_iter), dtype=float)
    best_y = np.zeros(0, dtype=float)
    best_resid = beta
    used_iter = 0

    for k in range(max_iter):
        v = matvec(q_vectors[k])
        for j in range(k + 1):
            hess[j, k] = float(np.dot(q_vectors[j], v))
            v = v - hess[j, k] * q_vectors[j]
        hess[k + 1, k] = float(np.linalg.norm(v))
        if hess[k + 1, k] > 1.0e-14 and k + 1 < max_iter:
            q_vectors.append(v / hess[k + 1, k])

        e1 = np.zeros(k + 2, dtype=float)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(hess[: k + 2, : k + 1], e1, rcond=None)
        resid = float(np.linalg.norm(e1 - hess[: k + 2, : k + 1] @ y))
        best_y = y
        best_resid = resid
        used_iter = k + 1
        if resid <= tol * beta:
            break
        if hess[k + 1, k] <= 1.0e-14:
            break

    basis = np.column_stack(q_vectors[:used_iter])
    return basis @ best_y, best_resid / beta, used_iter


def solve_newton(
    grid: np.ndarray,
    tau: float,
    gamma: float,
    seed: str,
    max_iter: int,
    tol: float,
    initial_P: np.ndarray | None = None,
    progress: bool = False,
    gmres_iter: int = 25,
    gmres_tol: float = 1.0e-3,
    fd_eps: float = 1.0e-5,
    min_alpha: float = 1.0e-4,
    step_cap: float = 0.25,
) -> SolveResult:
    if initial_P is not None:
        P = np.asarray(initial_P, dtype=float)
    elif seed == "fr":
        P = full_revelation_prices(grid, tau)
    elif seed == "no-learning":
        P = no_learning_prices(grid, tau, gamma)
    elif seed == "tilted":
        U1, U2, U3 = np.meshgrid(grid, grid, grid, indexing="ij")
        P = np.clip(no_learning_prices(grid, tau, gamma) + 0.05 * np.tanh(U1 - U2 + U3), 1.0e-5, 1.0 - 1.0e-5)
    else:
        raise ValueError(f"unknown seed {seed}")

    fr_prices = full_revelation_prices(grid, tau)
    history: list[dict[str, float]] = []
    post = np.empty((3,) + P.shape, dtype=float)
    converged = False

    for it in range(1, max_iter + 1):
        start = time.perf_counter()
        F, post = fixed_point_residual(P, grid, tau, gamma)
        residual = float(np.max(np.abs(F)))
        if residual < tol:
            converged = True
            break

        x0 = P.ravel()
        f0 = F.ravel()
        rhs = -f0

        def jv(v: np.ndarray) -> np.ndarray:
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0:
                return np.zeros_like(v)
            eps = fd_eps * (1.0 + float(np.linalg.norm(x0))) / v_norm
            P_pert = np.clip((x0 + eps * v).reshape(P.shape), 1.0e-8, 1.0 - 1.0e-8)
            F_pert, _ = fixed_point_residual(P_pert, grid, tau, gamma)
            return (F_pert.ravel() - f0) / eps

        delta, gmres_relres, gmres_used = gmres(jv, rhs, gmres_iter, gmres_tol)
        delta_inf = float(np.max(np.abs(delta)))
        if delta_inf > step_cap:
            delta = delta * (step_cap / delta_inf)
            delta_inf = step_cap

        accepted = False
        alpha = 1.0
        trial_residual = math.inf
        trial_post = post
        trial_P = P
        while alpha >= min_alpha:
            candidate_P = np.clip((x0 + alpha * delta).reshape(P.shape), 1.0e-8, 1.0 - 1.0e-8)
            trial_F, trial_post = fixed_point_residual(candidate_P, grid, tau, gamma)
            trial_residual = float(np.max(np.abs(trial_F)))
            if trial_residual < residual:
                accepted = True
                trial_P = candidate_P
                break
            alpha *= 0.5

        if not accepted:
            # Last-resort damped Picard move; still record it as a failed Newton line search.
            trial_P = np.clip(P + min(0.1, step_cap) * F, 1.0e-8, 1.0 - 1.0e-8)
            trial_F, trial_post = fixed_point_residual(trial_P, grid, tau, gamma)
            trial_residual = float(np.max(np.abs(trial_F)))
            alpha = 0.0

        P = trial_P
        post = trial_post
        deficit = revelation_deficit(P, grid, tau)
        max_fr_error = float(np.max(np.abs(P - fr_prices)))
        elapsed = time.perf_counter() - start
        hist = {
            "iteration": float(it),
            "residual_inf": trial_residual,
            "revelation_deficit": deficit,
            "max_fr_error": max_fr_error,
            "newton_alpha": alpha,
            "gmres_relres": gmres_relres,
            "gmres_iter": float(gmres_used),
            "newton_step_inf": delta_inf,
        }
        history.append(hist)
        if progress:
            print(
                f"newton={it} residual={trial_residual:.6e} "
                f"alpha={alpha:.3g} gmres={gmres_relres:.3e}/{gmres_used} "
                f"1-R2={deficit:.6e} max_fr_error={max_fr_error:.6e} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )
        if trial_residual < tol:
            converged = True
            break

    F, post = fixed_point_residual(P, grid, tau, gamma)
    residual = float(np.max(np.abs(F)))
    return SolveResult(P=P, posteriors=post, residual_inf=residual, iterations=len(history), converged=converged, history=history)


def solve(
    grid: np.ndarray,
    tau: float,
    gamma: float,
    seed: str,
    max_iter: int,
    tol: float,
    damping: float,
    initial_P: np.ndarray | None = None,
    progress: bool = False,
    start_iteration: int = 0,
    anderson_m: int = 0,
    anderson_beta: float = 1.0,
) -> SolveResult:
    if initial_P is not None:
        P = np.asarray(initial_P, dtype=float)
    elif seed == "fr":
        P = full_revelation_prices(grid, tau)
    elif seed == "no-learning":
        P = no_learning_prices(grid, tau, gamma)
    elif seed == "tilted":
        U1, U2, U3 = np.meshgrid(grid, grid, grid, indexing="ij")
        P = np.clip(no_learning_prices(grid, tau, gamma) + 0.05 * np.tanh(U1 - U2 + U3), 1.0e-5, 1.0 - 1.0e-5)
    else:
        raise ValueError(f"unknown seed {seed}")

    post = np.empty((3,) + P.shape, dtype=float)
    history: list[dict[str, float]] = []
    converged = False
    residual = math.inf
    fr_prices = full_revelation_prices(grid, tau)
    x_hist: list[np.ndarray] = []
    f_hist: list[np.ndarray] = []

    for local_it in range(1, max_iter + 1):
        it = start_iteration + local_it
        start = time.perf_counter()
        candidate, post = phi(P, grid, tau, gamma)
        residual = float(np.max(np.abs(candidate - P)))
        relaxed = (1.0 - damping) * P + damping * candidate
        if anderson_m > 0:
            x_flat = P.ravel()
            f_flat = (candidate - P).ravel()
            x_hist.append(x_flat.copy())
            f_hist.append(f_flat.copy())
            if len(f_hist) > anderson_m + 1:
                x_hist.pop(0)
                f_hist.pop(0)
            if len(f_hist) >= 2:
                df = np.column_stack([f_hist[q + 1] - f_hist[q] for q in range(len(f_hist) - 1)])
                dx = np.column_stack([x_hist[q + 1] - x_hist[q] for q in range(len(x_hist) - 1)])
                try:
                    coef, *_ = np.linalg.lstsq(df, f_flat, rcond=None)
                    aa_flat = x_flat + f_flat - (dx + df) @ coef
                    if np.all(np.isfinite(aa_flat)):
                        relaxed = (1.0 - damping) * relaxed + damping * aa_flat.reshape(P.shape)
                except np.linalg.LinAlgError:
                    pass
        P = np.clip(relaxed, 1.0e-8, 1.0 - 1.0e-8)
        deficit = revelation_deficit(P, grid, tau)
        max_fr_error = float(np.max(np.abs(P - fr_prices)))
        hist = {
            "iteration": float(it),
            "residual_inf": residual,
            "revelation_deficit": deficit,
            "max_fr_error": max_fr_error,
        }
        history.append(hist)
        if progress:
            elapsed = time.perf_counter() - start
            print(
                f"iter={it} residual={residual:.6e} "
                f"1-R2={deficit:.6e} max_fr_error={max_fr_error:.6e} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )
        if residual < tol:
            converged = True
            break

    # Recompute residual and posteriors at the returned point.
    candidate, post = phi(P, grid, tau, gamma)
    residual = float(np.max(np.abs(candidate - P)))
    return SolveResult(P=P, posteriors=post, residual_inf=residual, iterations=len(history), converged=converged, history=history)


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def fixed_point_residual(P: np.ndarray, grid: np.ndarray, tau: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    candidate, post = phi(P, grid, tau, gamma)
    return P - candidate, post


def gmres(
    matvec,
    b: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Small restarted-free GMRES for dense Krylov bases."""
    n = b.size
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        return np.zeros_like(b), 0.0, 0
    q = np.zeros((n, max_iter + 1), dtype=float)
    h = np.zeros((max_iter + 1, max_iter), dtype=float)
    q[:, 0] = b / beta
    rhs = np.zeros(max_iter + 1, dtype=float)
    rhs[0] = beta
    best_y = np.zeros(0, dtype=float)
    best_resid = beta
    best_iter = 0

    for k in range(max_iter):
        v = matvec(q[:, k])
        for j in range(k + 1):
            h[j, k] = float(np.dot(q[:, j], v))
            v = v - h[j, k] * q[:, j]
        h[k + 1, k] = float(np.linalg.norm(v))
        if h[k + 1, k] > 1.0e-14 and k + 1 < max_iter + 1:
            q[:, k + 1] = v / h[k + 1, k]
        y, *_ = np.linalg.lstsq(h[: k + 2, : k + 1], rhs[: k + 2], rcond=None)
        resid = float(np.linalg.norm(rhs[: k + 2] - h[: k + 2, : k + 1] @ y))
        best_y = y
        best_resid = resid
        best_iter = k + 1
        if resid <= tol * beta:
            break
    step = q[:, :best_iter] @ best_y
    return step, best_resid, best_iter


def solve_newton(
    grid: np.ndarray,
    tau: float,
    gamma: float,
    seed: str,
    max_iter: int,
    tol: float,
    initial_P: np.ndarray | None = None,
    progress: bool = False,
    gmres_iter: int = 20,
    jvp_eps: float = 1.0e-5,
    min_step: float = 1.0e-4,
) -> SolveResult:
    if initial_P is not None:
        P = np.asarray(initial_P, dtype=float)
    elif seed == "fr":
        P = full_revelation_prices(grid, tau)
    elif seed == "no-learning":
        P = no_learning_prices(grid, tau, gamma)
    elif seed == "tilted":
        U1, U2, U3 = np.meshgrid(grid, grid, grid, indexing="ij")
        P = np.clip(no_learning_prices(grid, tau, gamma) + 0.05 * np.tanh(U1 - U2 + U3), 1.0e-5, 1.0 - 1.0e-5)
    else:
        raise ValueError(f"unknown seed {seed}")

    fr_prices = full_revelation_prices(grid, tau)
    history: list[dict[str, float]] = []
    converged = False
    post = np.empty((3,) + P.shape, dtype=float)

    for it in range(1, max_iter + 1):
        start = time.perf_counter()
        F, post = fixed_point_residual(P, grid, tau, gamma)
        residual = float(np.max(np.abs(F)))
        deficit = revelation_deficit(P, grid, tau)
        max_fr_error = float(np.max(np.abs(P - fr_prices)))
        history.append(
            {
                "iteration": float(it),
                "residual_inf": residual,
                "revelation_deficit": deficit,
                "max_fr_error": max_fr_error,
            }
        )
        if progress:
            print(
                f"newton={it} residual={residual:.6e} "
                f"1-R2={deficit:.6e} max_fr_error={max_fr_error:.6e}",
                flush=True,
            )
        if residual < tol:
            converged = True
            break

        f_flat = F.ravel()
        p_flat = P.ravel()
        jvp_scale = jvp_eps * max(1.0, float(np.linalg.norm(p_flat)))

        def matvec(v: np.ndarray) -> np.ndarray:
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0:
                return np.zeros_like(v)
            eps = jvp_scale / v_norm
            P_eps = np.clip((p_flat + eps * v).reshape(P.shape), 1.0e-8, 1.0 - 1.0e-8)
            F_eps, _ = fixed_point_residual(P_eps, grid, tau, gamma)
            return (F_eps.ravel() - f_flat) / eps

        step, krylov_resid, krylov_iter = gmres(matvec, -f_flat, max_iter=gmres_iter, tol=1.0e-2)
        accepted = False
        alpha = 1.0
        best_P = P
        best_residual = residual
        while alpha >= min_step:
            trial = np.clip((p_flat + alpha * step).reshape(P.shape), 1.0e-8, 1.0 - 1.0e-8)
            trial_F, _ = fixed_point_residual(trial, grid, tau, gamma)
            trial_residual = float(np.max(np.abs(trial_F)))
            if trial_residual < best_residual:
                best_P = trial
                best_residual = trial_residual
                accepted = True
                if trial_residual <= (1.0 - 1.0e-4 * alpha) * residual:
                    break
            alpha *= 0.5
        if not accepted:
            # Fall back to a conservative fixed-point step if Newton fails.
            candidate = P - F
            for alpha in (0.2, 0.1, 0.05, 0.02, 0.01):
                trial = np.clip((1.0 - alpha) * P + alpha * candidate, 1.0e-8, 1.0 - 1.0e-8)
                trial_F, _ = fixed_point_residual(trial, grid, tau, gamma)
                trial_residual = float(np.max(np.abs(trial_F)))
                if trial_residual < best_residual:
                    best_P = trial
                    best_residual = trial_residual
                    accepted = True
                    break
        P = best_P
        if progress:
            elapsed = time.perf_counter() - start
            print(
                f"  gmres_iter={krylov_iter} gmres_resid={krylov_resid:.3e} "
                f"line_residual={best_residual:.6e} seconds={elapsed:.2f}",
                flush=True,
            )
        if not accepted:
            break

    F, post = fixed_point_residual(P, grid, tau, gamma)
    residual = float(np.max(np.abs(F)))
    return SolveResult(P=P, posteriors=post, residual_inf=residual, iterations=len(history), converged=converged, history=history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--umax", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", choices=["no-learning", "fr", "tilted", "array"], default="no-learning")
    parser.add_argument("--seed-array", type=Path, help="npz file with arrays named grid and P; interpolated if needed")
    parser.add_argument("--save-array", action="store_true", help="write the solved price tensor as an npz file")
    parser.add_argument("--label", help="optional filename seed label")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--method", choices=["picard", "newton"], default="picard")
    parser.add_argument("--anderson", type=int, default=0, help="Anderson window; 0 uses damped Picard")
    parser.add_argument("--anderson-beta", type=float, default=1.0, help="relaxation applied to Anderson candidate")
    parser.add_argument("--gmres-max-iter", type=int, default=40)
    parser.add_argument("--gmres-tol", type=float, default=1.0e-4)
    parser.add_argument("--fd-eps", type=float, default=1.0e-5)
    parser.add_argument("--progress", action="store_true", help="print one progress line after each Picard iteration")
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree"))
    args = parser.parse_args()

    grid = np.linspace(-args.umax, args.umax, args.G)
    initial_P = load_seed_array(args.seed_array, grid) if args.seed_array is not None else None
    if args.method == "newton":
        result = solve_newton(
            grid,
            args.tau,
            args.gamma,
            args.seed,
            args.max_iter,
            args.tol,
            initial_P=initial_P,
            progress=args.progress,
            gmres_iter=args.gmres_max_iter,
            jvp_eps=args.fd_eps,
        )
    else:
        result = solve(
            grid,
            args.tau,
            args.gamma,
            args.seed,
            args.max_iter,
            args.tol,
            args.damping,
            initial_P=initial_P,
            progress=args.progress,
            anderson_m=args.anderson,
            anderson_beta=args.anderson_beta,
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    i = nearest_index(grid, 1.0)
    j = nearest_index(grid, -1.0)
    k = nearest_index(grid, 1.0)
    summary = {
        "G": args.G,
        "umax": args.umax,
        "tau": args.tau,
        "gamma": args.gamma,
        "seed": args.seed,
        "seed_array": str(args.seed_array) if args.seed_array is not None else None,
        "method": args.method,
        "damping": args.damping,
        "anderson_window": args.anderson,
        "anderson_beta": args.anderson_beta,
        "gmres_max_iter": args.gmres_max_iter if args.method == "newton" else None,
        "gmres_tol": args.gmres_tol if args.method == "newton" else None,
        "fd_eps": args.fd_eps if args.method == "newton" else None,
        "iterations": result.iterations,
        "converged": result.converged,
        "residual_inf": result.residual_inf,
        "revelation_deficit": revelation_deficit(result.P, grid, args.tau),
        "max_fr_error": float(np.max(np.abs(result.P - full_revelation_prices(grid, args.tau)))),
        "representative_realization": {
            "u": [float(grid[i]), float(grid[j]), float(grid[k])],
            "price": float(result.P[i, j, k]),
            "posteriors": [float(result.posteriors[a, i, j, k]) for a in range(3)],
            "private_priors": [float(logistic(args.tau * grid[idx])) for idx in (i, j, k)],
            "fr_price": float(full_revelation_prices(grid, args.tau)[i, j, k]),
        },
    }

    seed_label = args.label or args.seed
    stem = f"G{args.G}_tau{args.tau:g}_gamma{args.gamma:g}_{seed_label}"
    (args.outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.outdir / f"{stem}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "residual_inf", "revelation_deficit", "max_fr_error"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.history)
    if args.save_array:
        np.savez_compressed(args.outdir / f"{stem}_prices.npz", grid=grid, P=result.P)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
