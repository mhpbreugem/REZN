#!/usr/bin/env python3
"""Slight-perturbation restarts for the symmetric K=4, G=5 contour solver."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def smooth_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    raw = rng.normal(size=shape)
    out = raw.copy()
    for axis in range(len(shape)):
        out += np.roll(raw, 1, axis=axis) + np.roll(raw, -1, axis=axis)
    std = float(np.std(out))
    return out / std if std > 0 else out


def run_seed(task: dict) -> dict:
    root = Path(__file__).resolve().parent
    ree = load_module("full_ree_solver", root / "full_ree_solver.py")
    generic = load_module("generic_k_symmetric_solver", root / "generic_k_symmetric_solver.py")

    grid = np.linspace(-2.0, 2.0, 5)
    tau = 2.0
    gamma = 0.5
    rng = np.random.default_rng(task["rng_seed"])
    base = np.load(task["base_array"])["P"]
    amp = 10.0 ** rng.uniform(task["amp_min_log10"], task["amp_max_log10"])
    initial = np.clip(ree.logistic(ree.logit(base) + amp * smooth_noise(rng, base.shape)), 1.0e-8, 1.0 - 1.0e-8)

    P, post, residual, converged, history = generic.solve_k(
        ree,
        initial,
        grid,
        tau,
        gamma,
        task["max_iter"],
        task["tol"],
        task["damping"],
        task["anderson"],
        False,
    )
    fr = generic.full_revelation_prices(ree, grid, tau, 4)
    idx = (3, 1, 3, 3)
    return {
        "seed_index": task["seed_index"],
        "rng_seed": task["rng_seed"],
        "amplitude": amp,
        "iterations": len(history),
        "converged": converged,
        "residual_inf": residual,
        "max_fr_error": float(np.max(np.abs(P - fr))),
        "representative_price": float(P[idx]),
        "representative_posteriors": ";".join(f"{float(post[(agent,) + idx]):.12g}" for agent in range(4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=20260429)
    parser.add_argument("--base-array", type=Path, default=Path("results/full_ree/generic_k/K4_G5_tau2_gamma0.5_K4_G5_prices.npz"))
    parser.add_argument("--amp-min-log10", type=float, default=-8.0)
    parser.add_argument("--amp-max-log10", type=float, default=-3.0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--anderson", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree/generic_k/k4_perturbations"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "seed_index": idx,
            "rng_seed": args.rng_seed + 104729 * idx,
            "base_array": str(args.base_array),
            "amp_min_log10": args.amp_min_log10,
            "amp_max_log10": args.amp_max_log10,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "damping": args.damping,
            "anderson": args.anderson,
        }
        for idx in range(args.runs)
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_seed, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"run={row['seed_index']:03d} residual={row['residual_inf']:.3e} "
                f"converged={row['converged']} amp={row['amplitude']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda row: row["seed_index"])
    best = min(rows, key=lambda row: row["residual_inf"])
    summary = {
        "K": 4,
        "G": 5,
        "tau": 2.0,
        "gamma": 0.5,
        "base_array": str(args.base_array),
        "runs": args.runs,
        "rng_seed": args.rng_seed,
        "amplitude_log10_range": [args.amp_min_log10, args.amp_max_log10],
        "max_iter": args.max_iter,
        "damping": args.damping,
        "anderson": args.anderson,
        "tol": args.tol,
        "converged_count": sum(bool(row["converged"]) for row in rows),
        "best": best,
    }

    with (args.outdir / "seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
