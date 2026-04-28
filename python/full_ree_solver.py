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


def _axis_crossings(values: np.ndarray, grid: np.ndarray, target: float) -> list[float]:
    roots: list[float] = []
    for a in range(len(grid) - 1):
        y0 = float(values[a] - target)
        y1 = float(values[a + 1] - target)
        if abs(y0) < 1.0e-12:
            roots.append(float(grid[a]))
        if y0 == 0.0 and y1 == 0.0:
            continue
        if y0 * y1 < 0.0 or abs(y1) < 1.0e-12:
            denom = float(values[a + 1] - values[a])
            if abs(denom) < 1.0e-14:
                root = float(grid[a + 1])
            else:
                root = float(grid[a] + (target - values[a]) * (grid[a + 1] - grid[a]) / denom)
            roots.append(root)
    if abs(float(values[-1] - target)) < 1.0e-12:
        roots.append(float(grid[-1]))
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
        for ub in _axis_crossings(slice2d[a, :], grid, p):
            sums[0] += signal_density(ua, 0, tau) * signal_density(ub, 0, tau)
            sums[1] += signal_density(ua, 1, tau) * signal_density(ub, 1, tau)
            hits += 1

    # Pass B: second coordinate on grid, first coordinate off grid.
    for b, ub in enumerate(grid):
        for ua in _axis_crossings(slice2d[:, b], grid, p):
            sums[0] += signal_density(ua, 0, tau) * signal_density(ub, 0, tau)
            sums[1] += signal_density(ua, 1, tau) * signal_density(ub, 1, tau)
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


@dataclass
class SolveResult:
    P: np.ndarray
    posteriors: np.ndarray
    residual_inf: float
    iterations: int
    converged: bool
    history: list[dict[str, float]]


def solve(
    grid: np.ndarray,
    tau: float,
    gamma: float,
    seed: str,
    max_iter: int,
    tol: float,
    damping: float,
) -> SolveResult:
    if seed == "fr":
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

    for it in range(1, max_iter + 1):
        candidate, post = phi(P, grid, tau, gamma)
        residual = float(np.max(np.abs(candidate - P)))
        P = np.clip((1.0 - damping) * P + damping * candidate, 1.0e-8, 1.0 - 1.0e-8)
        hist = {
            "iteration": float(it),
            "residual_inf": residual,
            "revelation_deficit": revelation_deficit(P, grid, tau),
            "max_fr_error": float(np.max(np.abs(P - full_revelation_prices(grid, tau)))),
        }
        history.append(hist)
        if residual < tol:
            converged = True
            break

    # Recompute residual and posteriors at the returned point.
    candidate, post = phi(P, grid, tau, gamma)
    residual = float(np.max(np.abs(candidate - P)))
    return SolveResult(P=P, posteriors=post, residual_inf=residual, iterations=len(history), converged=converged, history=history)


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--umax", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", choices=["no-learning", "fr", "tilted"], default="no-learning")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree"))
    args = parser.parse_args()

    grid = np.linspace(-args.umax, args.umax, args.G)
    result = solve(grid, args.tau, args.gamma, args.seed, args.max_iter, args.tol, args.damping)
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
        "damping": args.damping,
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

    stem = f"G{args.G}_tau{args.tau:g}_gamma{args.gamma:g}_{args.seed}"
    (args.outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.outdir / f"{stem}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "residual_inf", "revelation_deficit", "max_fr_error"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.history)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
