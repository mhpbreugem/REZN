"""Re-run problem-tau points with more stages, warm-starting from
the existing checkpoint.

Usage:
    python -m code.task3_g400_polish --gamma 4.0 --F-threshold 0.05
        --max-stages 4 --inner-max-iter 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import DTYPE
from .task3_g400_sweep import (solve_one_tau, save_checkpoint, tau_filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--F-threshold", type=float, default=0.02,
                    help="re-run any tau with F_inner > this")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/full_ree"))
    ap.add_argument("--max-stages", type=int, default=4)
    ap.add_argument("--inner-max-iter", type=int, default=12)
    ap.add_argument("--inner-tol", type=float, default=1e-9)
    ap.add_argument("--presmooth-steps", type=int, default=8)
    ap.add_argument("--presmooth-alpha", type=float, default=0.05)
    ap.add_argument("--only-tau", type=str, default="",
                    help="comma-separated taus to force-rerun")
    args = ap.parse_args()

    g_tag = f"g{int(round(args.gamma * 100)):03d}"
    cp_glob = sorted(args.out_dir.glob(f"task3_{g_tag}_t*_mp50.json"))
    if not cp_glob:
        raise SystemExit(f"No checkpoints in {args.out_dir}")

    only = set()
    if args.only_tau:
        only = {float(t) for t in args.only_tau.split(",")}

    targets = []
    for cp in cp_glob:
        with cp.open() as f:
            d = json.load(f)
        F = float(d.get("F_inner_inf_final", 0.0))
        tau = float(d["tau"])
        if only:
            if tau in only:
                targets.append((tau, cp, d, F))
        else:
            if F > args.F_threshold:
                targets.append((tau, cp, d, F))

    print(f"Polish: {len(targets)} taus to re-run")
    for tau, cp, d, F in targets:
        print(f"  tau={tau:.2f}  current F={F:.3e}")

    for tau, cp, d, F in targets:
        print(f"\n[polish] tau={tau:.2f} (warm from existing checkpoint, "
              f"max_stages={args.max_stages}) ...", flush=True)
        P_init = np.asarray(d["P_inner"], dtype=DTYPE)
        G_inner = int(d["G_inner"])
        pad = int(d["pad"])
        u_inner_max = float(d["u_inner_max"])
        u_outer_max = float(d["u_outer_max"])
        kernel = d.get("kernel", "cubic")

        P_inner, P_full, u_inner, info = solve_one_tau(
            tau, args.gamma,
            G_inner=G_inner, pad=pad,
            u_inner_max=u_inner_max, u_outer_max=u_outer_max,
            kernel=kernel,
            max_stages=args.max_stages,
            inner_max_iter=args.inner_max_iter,
            inner_tol=args.inner_tol,
            presmooth_steps=args.presmooth_steps,
            presmooth_alpha=args.presmooth_alpha,
            P_init_inner=P_init,
        )
        G_full = G_inner + 2 * pad
        du = u_inner[1] - u_inner[0]
        u_full = np.array([u_inner[0] + (q - pad) * du for q in range(G_full)])

        print(f"[polish] tau={tau:.2f} new F={info['F_inner_inf']:.3e} "
              f"f128_1mR2={info['deficit_f128_weighted_1mR2']:.5e} "
              f"wall={info['wall_seconds']:.1f}s", flush=True)
        cp_path = save_checkpoint(
            args.out_dir, args.gamma, tau, G_inner, pad,
            kernel, u_inner_max, u_outer_max,
            u_inner, u_full, P_inner, info)
        print(f"[polish] wrote {cp_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
