#!/usr/bin/env python3
"""G=6 full-REE search with slightly heterogeneous signal precisions.

This script intentionally does not symmetrize the price tensor: when
``tau=(1.9999, 2.0, 2.0001)`` the three agent dimensions are no longer
exchangeable.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def density(ree, u: float | np.ndarray, v: int, tau: float):
    return ree.signal_density(u, v, tau)


def posterior_from_slice_hetero(
    ree,
    slice2d: np.ndarray,
    own_u: float,
    grid: np.ndarray,
    p: float,
    own_tau: float,
    other_taus: tuple[float, float],
) -> float:
    sums = np.zeros(2, dtype=float)
    hits = 0
    logit_slice = ree.logit(slice2d)
    logit_p = ree.logit(p)

    tau_a, tau_b = other_taus
    for a, ua in enumerate(grid):
        f0a = ree.signal_density_scalar(float(ua), 0, tau_a)
        f1a = ree.signal_density_scalar(float(ua), 1, tau_a)
        for ub in ree._axis_crossings(logit_slice[a, :], grid, logit_p):
            sums[0] += f0a * ree.signal_density_scalar(ub, 0, tau_b)
            sums[1] += f1a * ree.signal_density_scalar(ub, 1, tau_b)
            hits += 1

    for b, ub in enumerate(grid):
        f0b = ree.signal_density_scalar(float(ub), 0, tau_b)
        f1b = ree.signal_density_scalar(float(ub), 1, tau_b)
        for ua in ree._axis_crossings(logit_slice[:, b], grid, logit_p):
            sums[0] += ree.signal_density_scalar(ua, 0, tau_a) * f0b
            sums[1] += ree.signal_density_scalar(ua, 1, tau_a) * f1b
            hits += 1

    if hits == 0:
        a0, a1 = 1.0, 1.0
    else:
        a0, a1 = float(sums[0] / hits), float(sums[1] / hits)

    f0 = ree.signal_density_scalar(float(own_u), 0, own_tau) * a0
    f1 = ree.signal_density_scalar(float(own_u), 1, own_tau) * a1
    return float(np.clip(f1 / (f0 + f1), ree.EPS, 1.0 - ree.EPS))


def phi_hetero(ree, P: np.ndarray, grid: np.ndarray, taus: np.ndarray, gamma: float):
    G = len(grid)
    P_new = np.empty_like(P)
    post = np.empty((3, G, G, G), dtype=float)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = float(P[i, j, k])
                mu1 = posterior_from_slice_hetero(ree, P[i, :, :], grid[i], grid, p, taus[0], (taus[1], taus[2]))
                mu2 = posterior_from_slice_hetero(ree, P[:, j, :], grid[j], grid, p, taus[1], (taus[0], taus[2]))
                mu3 = posterior_from_slice_hetero(ree, P[:, :, k], grid[k], grid, p, taus[2], (taus[0], taus[1]))
                post[:, i, j, k] = (mu1, mu2, mu3)
                P_new[i, j, k] = ree.clear_market((mu1, mu2, mu3), gamma)
    return P_new, post


def full_revelation_prices_hetero(ree, grid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    U1, U2, U3 = np.meshgrid(grid, grid, grid, indexing="ij")
    return ree.logistic(taus[0] * U1 + taus[1] * U2 + taus[2] * U3)


def no_learning_prices_hetero(ree, grid: np.ndarray, taus: np.ndarray, gamma: float) -> np.ndarray:
    P = np.empty((len(grid), len(grid), len(grid)), dtype=float)
    priors = [ree.logistic(tau * grid) for tau in taus]
    for i, mu1 in enumerate(priors[0]):
        for j, mu2 in enumerate(priors[1]):
            for k, mu3 in enumerate(priors[2]):
                P[i, j, k] = ree.clear_market((float(mu1), float(mu2), float(mu3)), gamma)
    return P


def revelation_deficit_hetero(ree, P: np.ndarray, grid: np.ndarray, taus: np.ndarray) -> float:
    U1, U2, U3 = np.meshgrid(grid, grid, grid, indexing="ij")
    tstar = taus[0] * U1 + taus[1] * U2 + taus[2] * U3
    y = ree.logit(P)
    w1 = density(ree, U1, 1, taus[0]) * density(ree, U2, 1, taus[1]) * density(ree, U3, 1, taus[2])
    w0 = density(ree, U1, 0, taus[0]) * density(ree, U2, 0, taus[1]) * density(ree, U3, 0, taus[2])
    w = 0.5 * (w1 + w0)
    mask = (P > 1.0e-5) & (P < 1.0 - 1.0e-5)
    x = tstar[mask].ravel()
    y = y[mask].ravel()
    w = w[mask].ravel()
    w = w / w.sum()
    xb = float(np.sum(w * x))
    yb = float(np.sum(w * y))
    vx = float(np.sum(w * (x - xb) ** 2))
    vy = float(np.sum(w * (y - yb) ** 2))
    cxy = float(np.sum(w * (x - xb) * (y - yb)))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (cxy * cxy) / (vx * vy))


def solve_picard(
    ree,
    initial: np.ndarray,
    grid: np.ndarray,
    taus: np.ndarray,
    gamma: float,
    tol: float,
    max_iter: int,
    damping: float,
    anderson_m: int = 5,
    adaptive: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    P = np.asarray(initial, dtype=float)
    x_hist: list[np.ndarray] = []
    f_hist: list[np.ndarray] = []
    post = np.empty((3,) + P.shape, dtype=float)
    residual = float("inf")
    converged = False
    for it in range(1, max_iter + 1):
        candidate, post = phi_hetero(ree, P, grid, taus, gamma)
        F = candidate - P
        residual = float(np.max(np.abs(F)))
        if residual < tol:
            converged = True
            break

        relaxed = (1.0 - damping) * P + damping * candidate
        if anderson_m > 0:
            x_flat = P.ravel()
            f_flat = F.ravel()
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

        if adaptive:
            step = relaxed - P
            scale = 1.0
            best = np.clip(relaxed, 1.0e-8, 1.0 - 1.0e-8)
            best_res = float("inf")
            while damping * scale >= 1.0e-5:
                trial = np.clip(P + scale * step, 1.0e-8, 1.0 - 1.0e-8)
                trial_phi, _ = phi_hetero(ree, trial, grid, taus, gamma)
                trial_res = float(np.max(np.abs(trial_phi - trial)))
                if trial_res < best_res:
                    best = trial
                    best_res = trial_res
                if trial_res <= residual:
                    break
                scale *= 0.5
            P = best
        else:
            P = np.clip(relaxed, 1.0e-8, 1.0 - 1.0e-8)

    candidate, post = phi_hetero(ree, P, grid, taus, gamma)
    residual = float(np.max(np.abs(candidate - P)))
    return P, post, residual, it, converged or residual < tol


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    raw = rng.normal(size=shape)
    out = raw.copy()
    for axis in range(3):
        out += np.roll(raw, 1, axis=axis) + np.roll(raw, -1, axis=axis)
    out /= np.std(out)
    return out


def solve_seed(task: dict) -> dict:
    ree = load_solver()
    grid = np.linspace(-2.0, 2.0, 6)
    taus = np.asarray(task["taus"], dtype=float)
    gamma = 0.5
    P_fr = full_revelation_prices_hetero(ree, grid, taus)
    P_nonfr_ref = np.load(task["nonfr_path"])["P"]
    rng = np.random.default_rng(task["rng_seed"])

    family = task["family"]
    if family == "near_fr":
        base = P_fr
        amp = 10 ** rng.uniform(-10.0, -3.0)
    elif family == "near_nonfr":
        base = P_nonfr_ref
        amp = 10 ** rng.uniform(task["amp_min_log10"], task["amp_max_log10"])
    elif family == "mixed_fr_nonfr":
        w = rng.uniform(0.0, 1.0)
        base = w * P_nonfr_ref + (1.0 - w) * P_fr
        amp = 10 ** rng.uniform(-9.0, -3.0)
    else:
        P_no = no_learning_prices_hetero(ree, grid, taus, gamma)
        w = rng.uniform(0.0, 1.0)
        base = w * P_nonfr_ref + (1.0 - w) * P_no
        amp = 10 ** rng.uniform(-8.0, -2.0)

    initial = np.clip(ree.logistic(ree.logit(base) + amp * smooth_noise(rng, base.shape)), 1.0e-8, 1.0 - 1.0e-8)
    P, _, residual, iterations, converged = solve_picard(
        ree,
        initial,
        grid,
        taus,
        gamma,
        tol=task["stage1_tol"],
        max_iter=task["stage1_iter"],
        damping=0.25,
        anderson_m=5,
    )
    d_fr = float(np.max(np.abs(P - P_fr)))
    d_nonfr = float(np.max(np.abs(P - P_nonfr_ref)))
    nearest = "FR" if d_fr <= d_nonfr else "non-FR"
    return {
        "seed_index": task["seed_index"],
        "family": family,
        "amplitude": amp,
        "iterations": iterations,
        "stage1_converged": converged,
        "residual_inf": residual,
        "nearest_known": nearest,
        "distance_to_fr": d_fr,
        "distance_to_nonfr": d_nonfr,
        "revelation_deficit": revelation_deficit_hetero(ree, P, grid, taus),
    }


def summarize_cluster(ree, name: str, P: np.ndarray, grid: np.ndarray, taus: np.ndarray, count: int, refine_iters: int, converged: bool):
    phiP, post = phi_hetero(ree, P, grid, taus, 0.5)
    P_fr = full_revelation_prices_hetero(ree, grid, taus)
    residual = float(np.max(np.abs(phiP - P)))
    i = int(np.argmin(np.abs(grid - 1.0)))
    j = int(np.argmin(np.abs(grid + 1.0)))
    k = i
    return {
        "name": name,
        "stage1_nearest_count": count,
        "refine_iterations": refine_iters,
        "refine_converged": converged,
        "residual_inf": residual,
        "revelation_deficit": revelation_deficit_hetero(ree, P, grid, taus),
        "max_fr_error": float(np.max(np.abs(P - P_fr))),
        "representative_u": [float(grid[i]), float(grid[j]), float(grid[k])],
        "representative_price": float(P[i, j, k]),
        "representative_posteriors": [float(post[a, i, j, k]) for a in range(3)],
        "fr_price": float(P_fr[i, j, k]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260429)
    parser.add_argument("--taus", type=float, nargs=3, default=[1.9999, 2.0, 2.0001])
    parser.add_argument("--tol", type=float, default=1.0e-14)
    parser.add_argument("--stage1-iter", type=int, default=90)
    parser.add_argument("--stage1-tol", type=float, default=1.0e-7)
    parser.add_argument("--refine-iter", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-mode", choices=["mixed", "near_nonfr"], default="mixed")
    parser.add_argument("--amp-min-log10", type=float, default=-14.0)
    parser.add_argument("--amp-max-log10", type=float, default=-8.0)
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree/g6_hetero_tau_search"))
    args = parser.parse_args()

    ree = load_solver()
    grid = np.linspace(-2.0, 2.0, 6)
    taus = np.asarray(args.taus, dtype=float)
    gamma = 0.5
    args.outdir.mkdir(parents=True, exist_ok=True)

    P_fr = full_revelation_prices_hetero(ree, grid, taus)
    P_hom = np.load("results/full_ree/G6_tau2_gamma0.5_G6_floor_check_prices.npz")["P"]
    P_nonfr, post_nonfr, nonfr_res, nonfr_iters, nonfr_conv = solve_picard(
        ree,
        P_hom,
        grid,
        taus,
        gamma,
        tol=args.tol,
        max_iter=args.refine_iter,
        damping=0.25,
        anderson_m=5,
    )
    np.savez_compressed(args.outdir / "hetero_nonfr_ref_prices.npz", grid=grid, P=P_nonfr, taus=taus)

    tasks = []
    families = ["near_fr", "near_nonfr", "mixed_fr_nonfr", "mixed_no_learning"]
    for idx in range(args.seeds):
        family = "near_nonfr" if args.seed_mode == "near_nonfr" else families[idx % len(families)]
        tasks.append(
            {
                "seed_index": idx,
                "rng_seed": args.rng_seed + 7919 * idx,
                "family": family,
                "taus": list(map(float, taus)),
                "stage1_iter": args.stage1_iter,
                "stage1_tol": args.stage1_tol,
                "amp_min_log10": args.amp_min_log10,
                "amp_max_log10": args.amp_max_log10,
                "nonfr_path": str(args.outdir / "hetero_nonfr_ref_prices.npz"),
            }
        )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_seed, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"seed={row['seed_index']:03d} family={row['family']} "
                f"residual={row['residual_inf']:.3e} nearest={row['nearest_known']} "
                f"dFR={row['distance_to_fr']:.3e} dNF={row['distance_to_nonfr']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda r: r["seed_index"])

    counts = {
        "FR": sum(row["nearest_known"] == "FR" for row in rows),
        "non-FR": sum(row["nearest_known"] == "non-FR" for row in rows),
    }
    fr_phi, post_fr = phi_hetero(ree, P_fr, grid, taus, gamma)
    fr_res = float(np.max(np.abs(fr_phi - P_fr)))
    nonfr_summary = summarize_cluster(ree, "non-FR", P_nonfr, grid, taus, counts["non-FR"], nonfr_iters, nonfr_conv)
    fr_summary = summarize_cluster(ree, "FR", P_fr, grid, taus, counts["FR"], 0, fr_res < args.tol)

    summary = {
        "G": 6,
        "taus": list(map(float, taus)),
        "tau_spread": float(np.max(taus) - np.min(taus)),
        "gamma": gamma,
        "rng_seed": args.rng_seed,
        "requested_seeds": args.seeds,
        "tolerance": args.tol,
        "seed_mode": args.seed_mode,
        "amp_min_log10": args.amp_min_log10,
        "amp_max_log10": args.amp_max_log10,
        "stage1_iterations": args.stage1_iter,
        "stage1_nearest_counts": counts,
        "distinct_converged_clusters": int(fr_summary["refine_converged"]) + int(nonfr_summary["refine_converged"]),
        "clusters": [fr_summary, nonfr_summary],
    }

    with (args.outdir / "seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
