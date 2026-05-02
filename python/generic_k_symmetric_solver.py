#!/usr/bin/env python3
"""Symmetric K-agent contour solver for small G.

This generalizes the three-agent contour method to K=4 and K=5 at G=5.  The
state remains the full tensor P[i_1,...,i_K], but after each map evaluation the
price tensor is symmetrized over all agent permutations.  For an agent's
(K-1)-dimensional slice, the contour integral is approximated by sweeping all
but one axis on grid and root-finding the remaining axis; the procedure is
averaged over all choices of root axis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_solver():
    path = Path(__file__).with_name("full_ree_solver.py")
    spec = importlib.util.spec_from_file_location("full_ree_solver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["full_ree_solver"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def symmetrize_k(P: np.ndarray) -> np.ndarray:
    return sum(np.transpose(P, axes=perm) for perm in itertools.permutations(range(P.ndim))) / math.factorial(P.ndim)


def full_revelation_prices(ree, grid: np.ndarray, tau: float, K: int) -> np.ndarray:
    meshes = np.meshgrid(*([grid] * K), indexing="ij")
    total = sum(meshes)
    return ree.logistic(tau * total)


def no_learning_prices(ree, grid: np.ndarray, tau: float, gamma: float, K: int) -> np.ndarray:
    priors = ree.logistic(tau * grid)
    P = np.empty((len(grid),) * K, dtype=float)
    for idx in np.ndindex(P.shape):
        P[idx] = ree.clear_market([float(priors[i]) for i in idx], gamma)
    return P


def crra_demand_local(ree, mu: float, p: float, gamma: float) -> float:
    log_r = (ree.logit(mu) - ree.logit(p)) / gamma
    log_r = float(np.clip(log_r, -700.0, 700.0))
    r = math.exp(log_r)
    return (r - 1.0) / ((1.0 - p) + r * p)


def clear_market_local(ree, mus: list[float], gamma: float) -> float:
    """Market clearing with brackets adapted to extreme contour posteriors."""
    mus = [float(np.clip(mu, 1.0e-14, 1.0 - 1.0e-14)) for mu in mus]

    def excess(p: float) -> float:
        return sum(crra_demand_local(ree, mu, p, gamma) for mu in mus)

    lo = max(1.0e-14, min(mus) * 1.0e-4)
    hi = min(1.0 - 1.0e-14, 1.0 - (1.0 - max(mus)) * 1.0e-4)
    flo = excess(lo)
    fhi = excess(hi)
    if flo < 0.0:
        lo = 1.0e-14
        flo = excess(lo)
    if fhi > 0.0:
        hi = 1.0 - 1.0e-14
        fhi = excess(hi)
    if flo < 0.0 or fhi > 0.0:
        raise RuntimeError(f"market-clearing bracket failed: f(lo)={flo}, f(hi)={fhi}, mus={mus}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = excess(mid)
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def seed_from_lower_k(lower: np.ndarray, K: int) -> np.ndarray:
    """Initialize K tensor by averaging logit prices over all K subsets of size K-1."""
    ree = load_solver()
    G = lower.shape[0]
    out_logit = np.empty((G,) * K, dtype=float)
    for idx in np.ndindex(out_logit.shape):
        vals = []
        for drop in range(K):
            sub_idx = idx[:drop] + idx[drop + 1 :]
            vals.append(ree.logit(float(lower[sub_idx])))
        out_logit[idx] = float(np.mean(vals))
    return np.clip(ree.logistic(out_logit), 1.0e-8, 1.0 - 1.0e-8)


def contour_evidence_nd(ree, slice_nd: np.ndarray, grid: np.ndarray, p: float, tau: float) -> tuple[float, float]:
    dims = slice_nd.ndim
    sums = np.zeros(2, dtype=float)
    hits = 0
    axes = range(dims)
    for root_axis in axes:
        sweep_axes = [axis for axis in axes if axis != root_axis]
        for sweep_idx in itertools.product(range(len(grid)), repeat=dims - 1):
            selector: list[object] = [slice(None)] * dims
            coord_by_axis: dict[int, float] = {}
            for axis, grid_idx in zip(sweep_axes, sweep_idx):
                selector[axis] = grid_idx
                coord_by_axis[axis] = float(grid[grid_idx])
            values = np.asarray(slice_nd[tuple(selector)], dtype=float)
            for root in ree._axis_crossings(values, grid, p):
                coords = dict(coord_by_axis)
                coords[root_axis] = float(root)
                prod0 = 1.0
                prod1 = 1.0
                for axis in axes:
                    u = coords[axis]
                    prod0 *= ree.signal_density_scalar(u, 0, tau)
                    prod1 *= ree.signal_density_scalar(u, 1, tau)
                sums[0] += prod0
                sums[1] += prod1
                hits += 1
    if hits == 0:
        return 1.0, 1.0
    return float(sums[0] / hits), float(sums[1] / hits)


def posterior_for_agent(ree, P: np.ndarray, idx: tuple[int, ...], agent: int, grid: np.ndarray, p: float, tau: float) -> float:
    selector: list[object] = [slice(None)] * P.ndim
    selector[agent] = idx[agent]
    slice_nd = P[tuple(selector)]
    a0, a1 = contour_evidence_nd(ree, slice_nd, grid, p, tau)
    own_u = float(grid[idx[agent]])
    f0 = ree.signal_density_scalar(own_u, 0, tau) * a0
    f1 = ree.signal_density_scalar(own_u, 1, tau) * a1
    return float(np.clip(f1 / (f0 + f1), ree.EPS, 1.0 - ree.EPS))


def phi_k(ree, P: np.ndarray, grid: np.ndarray, tau: float, gamma: float, symmetrize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    K = P.ndim
    P_new = np.empty_like(P)
    post = np.empty((K,) + P.shape, dtype=float)
    for idx in np.ndindex(P.shape):
        p = float(P[idx])
        mus = [posterior_for_agent(ree, P, idx, agent, grid, p, tau) for agent in range(K)]
        post[(slice(None),) + idx] = mus
        P_new[idx] = clear_market_local(ree, mus, gamma)
    if symmetrize:
        P_new = symmetrize_k(P_new)
    return P_new, post


def solve_k(
    ree,
    P0: np.ndarray,
    grid: np.ndarray,
    tau: float,
    gamma: float,
    max_iter: int,
    tol: float,
    damping: float,
    anderson_m: int,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, float, bool, list[dict[str, float]]]:
    P = np.asarray(P0, dtype=float)
    K = P.ndim
    fr = full_revelation_prices(ree, grid, tau, K)
    x_hist: list[np.ndarray] = []
    f_hist: list[np.ndarray] = []
    history: list[dict[str, float]] = []
    post = np.empty((K,) + P.shape, dtype=float)
    converged = False
    for it in range(1, max_iter + 1):
        start = time.perf_counter()
        candidate, post = phi_k(ree, P, grid, tau, gamma, symmetrize=True)
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
                    aa = x_flat + f_flat - (dx + df) @ coef
                    if np.all(np.isfinite(aa)):
                        relaxed = (1.0 - damping) * relaxed + damping * aa.reshape(P.shape)
                except np.linalg.LinAlgError:
                    pass
        P = np.clip(relaxed, 1.0e-8, 1.0 - 1.0e-8)
        hist = {
            "iteration": float(it),
            "residual_inf": residual,
            "max_fr_error": float(np.max(np.abs(P - fr))),
        }
        history.append(hist)
        if progress:
            print(
                f"K={K} iter={it} residual={residual:.6e} "
                f"max_fr_error={hist['max_fr_error']:.6e} seconds={time.perf_counter() - start:.2f}",
                flush=True,
            )
        if residual < tol:
            converged = True
            break
    candidate, post = phi_k(ree, P, grid, tau, gamma, symmetrize=True)
    residual = float(np.max(np.abs(candidate - P)))
    return P, post, residual, converged or residual < tol, history


def representative_summary(ree, P: np.ndarray, post: np.ndarray, grid: np.ndarray, tau: float) -> dict:
    K = P.ndim
    idx = tuple(int(np.argmin(np.abs(grid - target))) for target in ([1.0, -1.0] + [1.0] * (K - 2)))
    fr = full_revelation_prices(ree, grid, tau, K)
    return {
        "index": list(idx),
        "u": [float(grid[i]) for i in idx],
        "price": float(P[idx]),
        "posteriors": [float(post[(agent,) + idx]) for agent in range(K)],
        "fr_price": float(fr[idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, choices=[4, 5])
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", choices=["no-learning", "fr", "lower"], default="no-learning")
    parser.add_argument("--lower-array", type=Path)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--anderson", type=int, default=5)
    parser.add_argument("--label")
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree/generic_k"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    ree = load_solver()
    grid = np.linspace(-2.0, 2.0, args.G)
    if args.seed == "fr":
        P0 = full_revelation_prices(ree, grid, args.tau, args.K)
    elif args.seed == "lower":
        if args.lower_array is None:
            raise ValueError("--lower-array is required for lower seed")
        P0 = seed_from_lower_k(np.load(args.lower_array)["P"], args.K)
    else:
        P0 = no_learning_prices(ree, grid, args.tau, args.gamma, args.K)

    P, post, residual, converged, history = solve_k(
        ree,
        P0,
        grid,
        args.tau,
        args.gamma,
        args.max_iter,
        args.tol,
        args.damping,
        args.anderson,
        args.progress,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    label = args.label or args.seed
    stem = f"K{args.K}_G{args.G}_tau{args.tau:g}_gamma{args.gamma:g}_{label}"
    fr = full_revelation_prices(ree, grid, args.tau, args.K)
    summary = {
        "K": args.K,
        "G": args.G,
        "tau": args.tau,
        "gamma": args.gamma,
        "seed": args.seed,
        "lower_array": str(args.lower_array) if args.lower_array is not None else None,
        "iterations": len(history),
        "converged": converged,
        "residual_inf": residual,
        "max_fr_error": float(np.max(np.abs(P - fr))),
        "representative": representative_summary(ree, P, post, grid, args.tau),
    }
    (args.outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.outdir / f"{stem}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "residual_inf", "max_fr_error"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(args.outdir / f"{stem}_prices.npz", grid=grid, P=P)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
