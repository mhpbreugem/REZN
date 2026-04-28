#!/usr/bin/env python3
"""Thin CLI wrapper for rezn_n128.solve.

Example — low-τ heterogeneous K=3 G=11:

    python3 -u run_solve.py \\
        --gammas 0.3 3.0 30.0 \\
        --taus   0.5 1.0  2.0 \\
        --G 11 --umax 2.0 \\
        --picard-iters 20000 --lm-iters 15 --tsvd-iters 8 \\
        --log solve.log \\
        --save-to PR_K3_lowtau_het.pkl
"""
from __future__ import annotations
import argparse
import sys
import numpy as np

import rezn_n128


def main(argv=None):
    p = argparse.ArgumentParser(description="Solve REE fixed point in float128")
    p.add_argument("--gammas", nargs="+", type=float, required=True)
    p.add_argument("--taus",   nargs="+", type=float, required=True)
    p.add_argument("--Ws",     nargs="+", type=float, default=None,
                   help="(default: 1.0 for each agent)")
    p.add_argument("--G",     type=int,   required=True)
    p.add_argument("--umax",  type=float, required=True)
    p.add_argument("--P-init", default=None,
                   help="path to pickle with prior P (optional warm-start)")
    p.add_argument("--picard-iters", type=int, default=20000)
    p.add_argument("--picard-alpha0", type=float, default=0.20)
    p.add_argument("--picard-alpha-min", type=float, default=0.02)
    p.add_argument("--picard-alpha-max", type=float, default=0.30)
    p.add_argument("--lm-iters",   type=int, default=15)
    p.add_argument("--tsvd-iters", type=int, default=8)
    p.add_argument("--target-finf", type=float, default=1e-12)
    p.add_argument("--log", default=None,
                   help="mirror stdout to this log file")
    p.add_argument("--log-interval-s", type=float, default=120.0)
    p.add_argument("--save-to", default=None,
                   help="pickle path for the final P_f128 + metadata")
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    gammas = np.asarray(args.gammas, dtype=np.float64)
    taus   = np.asarray(args.taus,   dtype=np.float64)
    if args.Ws is None:
        Ws = np.ones(gammas.size, dtype=np.float64)
    else:
        Ws = np.asarray(args.Ws, dtype=np.float64)
    if not (gammas.size == taus.size == Ws.size):
        sys.stderr.write(
            f"K mismatch: |gammas|={gammas.size} |taus|={taus.size} |Ws|={Ws.size}\n")
        return 2

    res = rezn_n128.solve(
        gammas=gammas, taus=taus, Ws=Ws,
        G=args.G, umax=args.umax,
        P_init=args.P_init,
        picard_iters=args.picard_iters,
        picard_alpha0=args.picard_alpha0,
        picard_alpha_min=args.picard_alpha_min,
        picard_alpha_max=args.picard_alpha_max,
        lm_iters=args.lm_iters,
        tsvd_iters=args.tsvd_iters,
        target_finf=args.target_finf,
        log_path=args.log,
        log_interval_s=args.log_interval_s,
        save_to=args.save_to,
        label=args.label,
    )
    print(f"DONE  Finf={res['Finf']:.3e}  1-R²={res['one_minus_R2']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
