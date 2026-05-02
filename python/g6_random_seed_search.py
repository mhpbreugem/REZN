#!/usr/bin/env python3
"""Random-seed basin search for G=6 full REE fixed points.

The search intentionally records every seed, including failures, and clusters
only solutions with true residual below the requested tolerance.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
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


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    raw = rng.normal(size=shape)
    out = raw.copy()
    # A cheap low-pass filter: average nearest neighbours on the tensor grid.
    for axis in range(3):
        out += np.roll(raw, 1, axis=axis) + np.roll(raw, -1, axis=axis)
    out /= np.std(out)
    return out


def classify_solution(P: np.ndarray, clusters: list[dict], tolerance: float) -> int:
    for idx, cluster in enumerate(clusters):
        dist = float(np.max(np.abs(P - cluster["P"])))
        if dist <= tolerance:
            cluster["count"] += 1
            cluster["max_member_distance"] = max(cluster["max_member_distance"], dist)
            return idx
    clusters.append({"P": P.copy(), "count": 1, "max_member_distance": 0.0})
    return len(clusters) - 1


def nearest_reference(P: np.ndarray, references: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {name: float(np.max(np.abs(P - ref))) for name, ref in references.items()}
    name = min(distances, key=distances.get)
    return name, distances[name]


def run_seed(payload: dict) -> dict:
    ree = load_solver()
    seed_idx = payload["seed_index"]
    rng = np.random.default_rng(payload["rng_seed"])
    grid = np.linspace(-2.0, 2.0, 6)
    tau = 2.0
    gamma = 0.5
    P_fr = payload["P_fr"]
    P_nonfr = payload["P_nonfr"]
    P_no_learning = payload["P_no_learning"]
    references = {"FR": P_fr, "non-FR": P_nonfr}

    family_choice = rng.uniform()
    if family_choice < 0.25:
        family = "near_fr"
        base = P_fr
        amp = 10 ** rng.uniform(-8.0, -3.0)
    elif family_choice < 0.50:
        family = "near_nonfr"
        base = P_nonfr
        amp = 10 ** rng.uniform(-8.0, -3.0)
    elif family_choice < 0.75:
        family = "mixed_fr_nonfr"
        weight = rng.uniform(0.0, 1.0)
        base = weight * P_nonfr + (1.0 - weight) * P_fr
        amp = 10 ** rng.uniform(-7.0, -2.0)
    else:
        family = "mixed_no_learning"
        weight = rng.uniform(0.0, 1.0)
        base = weight * P_nonfr + (1.0 - weight) * P_no_learning
        amp = 10 ** rng.uniform(-7.0, -2.0)

    noise = smooth_noise(rng, base.shape)
    initial = np.clip(ree.logistic(ree.logit(base) + amp * noise), 1.0e-8, 1.0 - 1.0e-8)
    initial_ref, initial_ref_distance = nearest_reference(initial, references)
    result = ree.solve(
        grid,
        tau,
        gamma,
        seed="array",
        max_iter=payload["stage1_iter"],
        tol=payload["stage1_tol"],
        damping=payload["damping"],
        initial_P=initial,
        anderson_m=payload["anderson"],
        anderson_beta=payload["anderson_beta"],
    )
    final_ref, final_ref_distance = nearest_reference(result.P, references)
    return {
        "seed_index": seed_idx,
        "family": family,
        "amplitude": amp,
        "stage1_iterations": result.iterations,
        "stage1_residual_inf": result.residual_inf,
        "stage1_revelation_deficit": ree.revelation_deficit(result.P, grid, tau),
        "stage1_max_fr_error": float(np.max(np.abs(result.P - P_fr))),
        "initial_nearest_reference": initial_ref,
        "initial_reference_distance": initial_ref_distance,
        "final_nearest_reference": final_ref,
        "final_reference_distance": final_ref_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260429)
    parser.add_argument("--tol", type=float, default=1.0e-14)
    parser.add_argument("--cluster-tol", type=float, default=1.0e-8)
    parser.add_argument("--stage1-iter", type=int, default=45)
    parser.add_argument("--stage1-tol", type=float, default=1.0e-8)
    parser.add_argument("--refine-iter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=Path("results/full_ree/random_g6"))
    args = parser.parse_args()

    ree = load_solver()
    rng = np.random.default_rng(args.rng_seed)
    grid = np.linspace(-2.0, 2.0, 6)
    tau = 2.0
    gamma = 0.5
    base_dir = Path("results/full_ree")
    P_fr = ree.full_revelation_prices(grid, tau)
    P_nonfr = ree.load_seed_array(base_dir / "G6_tau2_gamma0.5_G6_floor_check_prices.npz", grid)
    P_no_learning = ree.no_learning_prices(grid, tau, gamma)

    args.outdir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for seed_idx in range(args.seeds):
        payloads.append(
            {
                "seed_index": seed_idx,
                "rng_seed": int(rng.integers(0, 2**32 - 1)),
                "P_fr": P_fr,
                "P_nonfr": P_nonfr,
                "P_no_learning": P_no_learning,
                "stage1_iter": args.stage1_iter,
                "stage1_tol": args.stage1_tol,
                "damping": 0.25,
                "anderson": 5,
                "anderson_beta": 0.7,
            }
        )

    rows: list[dict] = []
    with mp.Pool(processes=args.workers) as pool:
        for row in pool.imap_unordered(run_seed, payloads):
            rows.append(row)
            print(
                f"seed={row['seed_index']:03d} family={row['family']} "
                f"residual={row['stage1_residual_inf']:.3e} "
                f"nearest={row['final_nearest_reference']} "
                f"distance={row['final_reference_distance']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda r: r["seed_index"])

    # Tighten each reference branch that was reached by at least one random seed.
    reached = sorted({row["final_nearest_reference"] for row in rows})
    reference_tensors = {"FR": P_fr, "non-FR": P_nonfr}
    cluster_summaries = []
    for idx, name in enumerate(reached):
        result = ree.solve(
            grid,
            tau,
            gamma,
            seed="array",
            max_iter=args.refine_iter,
            tol=args.tol,
            damping=0.25,
            initial_P=reference_tensors[name],
            anderson_m=5,
            anderson_beta=0.7,
        )
        P = result.P
        phiP, post = ree.phi(P, grid, tau, gamma)
        residual = float(np.max(np.abs(phiP - P)))
        i = int(np.argmin(np.abs(grid - 1.0)))
        j = int(np.argmin(np.abs(grid + 1.0)))
        k = i
        cluster_summaries.append(
            {
                "cluster_id": idx,
                "name": name,
                "stage1_nearest_count": sum(row["final_nearest_reference"] == name for row in rows),
                "refine_iterations": result.iterations,
                "refine_converged": residual < args.tol,
                "residual_inf": residual,
                "revelation_deficit": ree.revelation_deficit(P, grid, tau),
                "max_fr_error": float(np.max(np.abs(P - P_fr))),
                "representative_u": [float(grid[i]), float(grid[j]), float(grid[k])],
                "representative_price": float(P[i, j, k]),
                "representative_posteriors": [float(post[a, i, j, k]) for a in range(3)],
                "fr_price": float(P_fr[i, j, k]),
            }
        )

    summary = {
        "G": 6,
        "tau": tau,
        "gamma": gamma,
        "rng_seed": args.rng_seed,
        "requested_seeds": args.seeds,
        "stage1_iterations": args.stage1_iter,
        "tolerance": args.tol,
        "stage1_nearest_counts": {
            name: sum(row["final_nearest_reference"] == name for row in rows) for name in sorted(reference_tensors)
        },
        "refined_converged_count": sum(bool(cluster["refine_converged"]) for cluster in cluster_summaries),
        "cluster_tolerance": args.cluster_tol,
        "distinct_converged_clusters": len(cluster_summaries),
        "clusters": cluster_summaries,
    }

    with (args.outdir / "seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
