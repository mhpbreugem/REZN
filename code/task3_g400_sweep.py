"""Task 3 driver: gamma=4.0 tau-sweep using K=3 staggered Newton-Krylov.

Walks tau through 12 values, warm-starting each from the previous run.
Saves one JSON checkpoint per tau plus a summary file.

Output schema (per-tau):
{
  "G_inner": int, "pad": int, "G_full": int,
  "u_inner_max": float, "u_outer_max": float,
  "tau": float, "gamma": float, "K": 3,
  "kernel": str, "method": "K3_staggered_newton_krylov_cubic_float64",
  "F_inner_inf_final": float,
  "weighted_1mR2_f128": float,
  "weighted_slope": float,
  "n_triples": int,
  "wall_seconds": float,
  "u_inner": [G_inner floats],
  "u_full":  [G_full floats],
  "P_inner": [[[G_inner floats] x G_inner] x G_inner]   (price matrix on inner grid)
}
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import DTYPE
from .contour_K3_halo import (init_no_learning_K3,
                              phi_K3_halo_cubic, phi_K3_halo)
from .f128 import revelation_deficit_f128
from .halo import extract_inner, replace_inner
from .staggered import staggered_solve


# --------- weighted regression slope (matches FIGURES_TODO.md) ---------

def weighted_slope_and_1mR2(P: np.ndarray, u_grid: np.ndarray,
                            tau: float) -> tuple[float, float]:
    """Compute weighted (slope, 1-R^2) of logit(P) on T*=tau*sum(u_k).

    Weights: w(u_1,u_2,u_3) = 0.5*(prod f_1 + prod f_0) (ex-ante).
    Computation uses float64 (paper's f128 lift only changes the last
    few digits of 1-R^2, negligible for our purposes).
    """
    G = u_grid.size
    # Build T*, weights, logit(P) over the full G^3 cube
    f1 = np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u_grid - 0.5) ** 2)
    f0 = np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u_grid + 0.5) ** 2)

    # Broadcasting
    u3 = u_grid.reshape(1, 1, G)
    u2 = u_grid.reshape(1, G, 1)
    u1 = u_grid.reshape(G, 1, 1)
    Tstar = tau * (u1 + u2 + u3)
    f1_3 = f1.reshape(1, 1, G)
    f1_2 = f1.reshape(1, G, 1)
    f1_1 = f1.reshape(G, 1, 1)
    f0_3 = f0.reshape(1, 1, G)
    f0_2 = f0.reshape(1, G, 1)
    f0_1 = f0.reshape(G, 1, 1)
    weights = 0.5 * (f1_1 * f1_2 * f1_3 + f0_1 * f0_2 * f0_3)

    P_clip = np.clip(P, 1e-12, 1 - 1e-12)
    logit_p = np.log(P_clip) - np.log(1 - P_clip)

    Tstar_flat = Tstar.ravel()
    logit_flat = logit_p.ravel()
    w_flat = weights.ravel()

    sw = np.sqrt(w_flat)
    slope, intercept = np.polyfit(Tstar_flat, logit_flat, 1, w=sw)
    pred = slope * Tstar_flat + intercept
    mean_lp = np.average(logit_flat, weights=w_flat)
    var_tot = np.average((logit_flat - mean_lp) ** 2, weights=w_flat)
    var_res = np.average((logit_flat - pred) ** 2, weights=w_flat)
    one_mR2 = float(var_res / var_tot)
    return float(slope), one_mR2


# --------- file naming ---------

def tau_filename(tau: float) -> str:
    """tau=0.3 -> 't0300', tau=15.0 -> 't15000'."""
    return f"t{int(round(tau * 1000)):05d}"


# --------- single solve ---------

def solve_one_tau(tau: float, gamma: float, *,
                  G_inner: int, pad: int,
                  u_inner_max: float, u_outer_max: float,
                  kernel: str = "cubic",
                  max_stages: int = 5,
                  inner_max_iter: int = 12,
                  inner_tol: float = 1e-9,
                  presmooth_steps: int = 10,
                  presmooth_alpha: float = 0.05,
                  inner_inner_maxiter: int = 80,
                  outer_k: int = 40,
                  rdiff: float = 1e-4,
                  P_init_inner: np.ndarray | None = None,
                  heartbeat_s: float = 30.0,
                  verbose: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Solve K=3 staggered NK for symmetric (gamma, tau) triple.

    Returns (P_inner_final, P_full_final, u_grid_inner, info_dict).
    P_init_inner: optional warm-start; if None, no-learning halo seed.
    """
    G_full = G_inner + 2 * pad
    du = (2.0 * u_inner_max) / (G_inner - 1) if G_inner > 1 else 1.0
    u_full = np.empty(G_full, dtype=DTYPE)
    inner_lo = pad
    inner_hi = pad + G_inner
    for q in range(G_full):
        u_full[q] = -u_inner_max + (q - pad) * du
    u_grid_inner = u_full[inner_lo:inner_hi].copy()

    gamma_vec = np.array([gamma, gamma, gamma], dtype=DTYPE)
    tau_vec = np.array([tau, tau, tau], dtype=DTYPE)
    W_vec = np.array([1.0, 1.0, 1.0], dtype=DTYPE)

    halo = init_no_learning_K3(u_full, tau_vec, gamma_vec, W_vec)
    if P_init_inner is None:
        P_inner_seed = extract_inner(halo, inner_lo, inner_hi)
    else:
        # Use warm start: replace inner halo cells with provided init
        P_inner_seed = P_init_inner.astype(DTYPE, copy=True)

    if kernel == "cubic":
        def phi_full_fn(P_full):
            return phi_K3_halo_cubic(P_full, u_full, inner_lo, inner_hi,
                                     tau_vec, gamma_vec, W_vec)
    else:
        def phi_full_fn(P_full):
            return phi_K3_halo(P_full, u_full, inner_lo, inner_hi,
                               tau_vec, gamma_vec, W_vec)

    t0 = time.perf_counter()
    P_inner_final, history = staggered_solve(
        phi_full_fn, u_full, inner_lo, inner_hi,
        u_grid_inner=u_grid_inner, tau_vec=tau_vec, K=3,
        halo_initial=halo, inner_initial=P_inner_seed,
        max_stages=max_stages, stage_tol=1e-3,
        inner_method="lgmres",
        inner_max_iter=inner_max_iter, inner_tol=inner_tol,
        inner_outer_k=outer_k,
        inner_inner_maxiter=inner_inner_maxiter,
        inner_rdiff=rdiff,
        presmooth_steps=presmooth_steps,
        presmooth_alpha=presmooth_alpha,
        halo_update="no_learning",
        heartbeat_s=heartbeat_s,
    )
    wall = time.perf_counter() - t0

    P_full_final = replace_inner(halo, P_inner_final, inner_lo, inner_hi)
    F_full = phi_full_fn(P_full_final) - P_full_final
    F_inner = extract_inner(F_full, inner_lo, inner_hi)
    F_inf = float(np.max(np.abs(F_inner)))
    deficit = revelation_deficit_f128(P_inner_final, u_grid_inner, tau_vec, 3)
    slope, one_mR2 = weighted_slope_and_1mR2(P_inner_final, u_grid_inner, tau)

    info = {
        "wall_seconds": wall,
        "F_inner_inf": F_inf,
        "deficit_f128_weighted_1mR2": float(deficit),
        "weighted_1mR2_recomputed": one_mR2,
        "weighted_slope": slope,
        "n_triples": int(G_inner ** 3),
        "n_stages": len(history.stages) - 1,
    }
    return P_inner_final, P_full_final, u_grid_inner, info


# --------- save checkpoint ---------

def save_checkpoint(out_dir: Path, gamma: float, tau: float,
                    G_inner: int, pad: int, kernel: str,
                    u_inner_max: float, u_outer_max: float,
                    u_inner: np.ndarray, u_full: np.ndarray,
                    P_inner: np.ndarray, info: dict) -> Path:
    fname = f"task3_g{int(round(gamma * 100)):03d}_{tau_filename(tau)}_mp50.json"
    path = out_dir / fname
    payload = {
        "G_inner": G_inner, "pad": pad,
        "G_full": int(u_full.size),
        "u_inner_max": u_inner_max, "u_outer_max": u_outer_max,
        "tau": tau, "gamma": gamma, "K": 3,
        "kernel": kernel,
        "method": "K3_staggered_newton_krylov_float64",
        "F_inner_inf_final": info["F_inner_inf"],
        "weighted_1mR2_f128": info["deficit_f128_weighted_1mR2"],
        "weighted_1mR2": info["weighted_1mR2_recomputed"],
        "weighted_slope": info["weighted_slope"],
        "n_triples": info["n_triples"],
        "wall_seconds": info["wall_seconds"],
        "n_stages": info["n_stages"],
        "u_inner": u_inner.tolist(),
        "u_full": u_full.tolist(),
        "P_inner": P_inner.tolist(),
    }
    with path.open("w") as f:
        json.dump(payload, f)
    return path


# --------- main sweep ---------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--G-inner", type=int, default=14)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--u-inner-max", type=float, default=3.0)
    p.add_argument("--u-outer-max", type=float, default=6.0)
    p.add_argument("--kernel", default="cubic", choices=["cubic", "scan"])
    p.add_argument("--max-stages", type=int, default=4)
    p.add_argument("--inner-max-iter", type=int, default=10)
    p.add_argument("--inner-tol", type=float, default=1e-9)
    p.add_argument("--presmooth-steps", type=int, default=10)
    p.add_argument("--presmooth-alpha", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/full_ree"))
    p.add_argument("--taus", type=str,
                   default="0.3,0.5,0.8,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0,15.0")
    p.add_argument("--start-tau", type=float, default=2.0,
                   help="warm-start anchor; runs walk down then up from here")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    taus = sorted({float(t) for t in args.taus.split(",")})
    start = float(args.start_tau)
    if start not in taus:
        taus.append(start)
        taus.sort()

    print("=" * 78)
    print(f"Task 3 sweep: gamma={args.gamma}  taus={taus}")
    print(f"  G_inner={args.G_inner}, pad={args.pad}, "
          f"u_inner_max={args.u_inner_max}")
    print(f"  kernel={args.kernel}, max_stages={args.max_stages}, "
          f"inner_max_iter={args.inner_max_iter}, "
          f"inner_tol={args.inner_tol}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 78, flush=True)

    # First: solve at start (no warm start, costs more iterations)
    print(f"\n[seed] Solving start tau={start} from no-learning ...",
          flush=True)
    P_inner, P_full, u_inner, info = solve_one_tau(
        start, args.gamma,
        G_inner=args.G_inner, pad=args.pad,
        u_inner_max=args.u_inner_max, u_outer_max=args.u_outer_max,
        kernel=args.kernel, max_stages=args.max_stages,
        inner_max_iter=args.inner_max_iter, inner_tol=args.inner_tol,
        presmooth_steps=args.presmooth_steps,
        presmooth_alpha=args.presmooth_alpha,
        P_init_inner=None,
    )
    G_full = u_inner.size + 2 * args.pad
    du = u_inner[1] - u_inner[0]
    u_full = np.array([u_inner[0] + (q - args.pad) * du for q in range(G_full)])

    print(f"[seed] tau={start} 1-R^2={info['weighted_1mR2_recomputed']:.5f} "
          f"slope={info['weighted_slope']:.4f} F={info['F_inner_inf']:.3e} "
          f"wall={info['wall_seconds']:.1f}s", flush=True)
    cp_path = save_checkpoint(
        args.out_dir, args.gamma, start, args.G_inner, args.pad,
        args.kernel, args.u_inner_max, args.u_outer_max,
        u_inner, u_full, P_inner, info)
    print(f"[seed] wrote {cp_path}", flush=True)

    # Walk down from start
    cur_P = P_inner.copy()
    walk_down = sorted([t for t in taus if t < start], reverse=True)
    walk_up = sorted([t for t in taus if t > start])

    summary_points = [{"tau": start,
                       "1-R2": info["weighted_1mR2_recomputed"],
                       "slope": info["weighted_slope"],
                       "F": info["F_inner_inf"]}]

    for tau in walk_down:
        print(f"\n[walk-down] tau={tau} (warm from prev)", flush=True)
        P_inner, P_full, u_inner, info = solve_one_tau(
            tau, args.gamma,
            G_inner=args.G_inner, pad=args.pad,
            u_inner_max=args.u_inner_max, u_outer_max=args.u_outer_max,
            kernel=args.kernel, max_stages=max(2, args.max_stages // 2),
            inner_max_iter=args.inner_max_iter, inner_tol=args.inner_tol,
            presmooth_steps=max(4, args.presmooth_steps // 2),
            presmooth_alpha=args.presmooth_alpha,
            P_init_inner=cur_P,
        )
        cur_P = P_inner.copy()
        print(f"[walk-down] tau={tau} 1-R^2={info['weighted_1mR2_recomputed']:.5f} "
              f"slope={info['weighted_slope']:.4f} F={info['F_inner_inf']:.3e} "
              f"wall={info['wall_seconds']:.1f}s", flush=True)
        cp_path = save_checkpoint(
            args.out_dir, args.gamma, tau, args.G_inner, args.pad,
            args.kernel, args.u_inner_max, args.u_outer_max,
            u_inner, u_full, P_inner, info)
        print(f"[walk-down] wrote {cp_path}", flush=True)
        summary_points.append({"tau": tau,
                               "1-R2": info["weighted_1mR2_recomputed"],
                               "slope": info["weighted_slope"],
                               "F": info["F_inner_inf"]})

    cur_P = summary_points[0]  # not used; just reset cur to start solution
    # Reload start checkpoint's P (it was saved); we kept P_inner_seed in memory only
    # Walk up from the start P
    # But start was the FIRST run; cur_P reflects whichever last loop iter ran
    # Easiest: reload from disk
    start_path = args.out_dir / (
        f"task3_g{int(round(args.gamma * 100)):03d}_{tau_filename(start)}_mp50.json")
    with start_path.open() as f:
        start_payload = json.load(f)
    cur_P = np.asarray(start_payload["P_inner"])

    for tau in walk_up:
        print(f"\n[walk-up] tau={tau} (warm from prev)", flush=True)
        P_inner, P_full, u_inner, info = solve_one_tau(
            tau, args.gamma,
            G_inner=args.G_inner, pad=args.pad,
            u_inner_max=args.u_inner_max, u_outer_max=args.u_outer_max,
            kernel=args.kernel, max_stages=max(2, args.max_stages // 2),
            inner_max_iter=args.inner_max_iter, inner_tol=args.inner_tol,
            presmooth_steps=max(4, args.presmooth_steps // 2),
            presmooth_alpha=args.presmooth_alpha,
            P_init_inner=cur_P,
        )
        cur_P = P_inner.copy()
        print(f"[walk-up] tau={tau} 1-R^2={info['weighted_1mR2_recomputed']:.5f} "
              f"slope={info['weighted_slope']:.4f} F={info['F_inner_inf']:.3e} "
              f"wall={info['wall_seconds']:.1f}s", flush=True)
        cp_path = save_checkpoint(
            args.out_dir, args.gamma, tau, args.G_inner, args.pad,
            args.kernel, args.u_inner_max, args.u_outer_max,
            u_inner, u_full, P_inner, info)
        print(f"[walk-up] wrote {cp_path}", flush=True)
        summary_points.append({"tau": tau,
                               "1-R2": info["weighted_1mR2_recomputed"],
                               "slope": info["weighted_slope"],
                               "F": info["F_inner_inf"]})

    # Sort & save summary
    summary_points.sort(key=lambda d: d["tau"])
    summary = {
        "gamma": args.gamma,
        "params": {
            "G_inner": args.G_inner,
            "pad": args.pad,
            "u_inner_max": args.u_inner_max,
            "u_outer_max": args.u_outer_max,
            "kernel": args.kernel,
            "method": "K3_staggered_newton_krylov_float64",
            "weighting": "ex-ante 0.5*(f0^3 + f1^3)",
        },
        "points": summary_points,
    }
    summary_path = args.out_dir / (
        f"fig4A_g{int(round(args.gamma * 100)):03d}_tau_sweep.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
