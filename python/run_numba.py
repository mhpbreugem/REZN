#!/usr/bin/env python3
"""Quick numba-f64 driver — runs legacy_k3.rezn_het.solve_picard directly.

For when you want maximum speed (numba JIT) over precision (f64 vs f128).
Saves results in the same pickle schema as rezn_n128 so they can be loaded
by rezn_n128.io.load.
"""
from __future__ import annotations
import argparse
import datetime
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "legacy_k3"))
import rezn_het as rh                                                # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gammas", nargs="+", type=float, required=True)
    p.add_argument("--taus",   nargs="+", type=float, required=True)
    p.add_argument("--Ws",     nargs="+", type=float, default=None)
    p.add_argument("--G",      type=int,   required=True)
    p.add_argument("--umax",   type=float, required=True)
    p.add_argument("--maxiters", type=int, default=20000)
    p.add_argument("--abstol",   type=float, default=1e-13)
    p.add_argument("--alpha",    type=float, default=1.0)
    p.add_argument("--P-init",   default=None,
                   help="warm-start pickle (loaded via rezn_n128.io)")
    p.add_argument("--save-to",  required=True)
    p.add_argument("--label",    default="")
    args = p.parse_args(argv)

    gammas = np.asarray(args.gammas, dtype=np.float64)
    taus   = np.asarray(args.taus,   dtype=np.float64)
    if args.Ws is None:
        Ws = np.ones(gammas.size, dtype=np.float64)
    else:
        Ws = np.asarray(args.Ws, dtype=np.float64)
    if not (gammas.size == taus.size == Ws.size == 3):
        sys.stderr.write("legacy rezn_het is K=3 only.\n")
        return 2

    P_init = None
    if args.P_init:
        sys.path.insert(0, HERE)
        import rezn_n128                                              # noqa: F401
        rec = rezn_n128.load(args.P_init)
        P_init = np.asarray(rec.get("P", rec.get("P_f128")), dtype=np.float64)
        if P_init.shape != (args.G,) * 3:
            sys.stderr.write(
                f"P_init shape {P_init.shape} != ({args.G},)*3; "
                "regenerate seed or pass matching G\n")
            return 2

    print(f"=== numba-f64 solve  K=3  G={args.G}  umax={args.umax} ===", flush=True)
    print(f"  γ = {gammas.tolist()}", flush=True)
    print(f"  τ = {taus.tolist()}", flush=True)
    print(f"  W = {Ws.tolist()}", flush=True)
    print(f"  α = {args.alpha}  abstol = {args.abstol:.1e}  "
          f"maxiters = {args.maxiters}", flush=True)
    if P_init is not None:
        print(f"  warm-start: {args.P_init}", flush=True)

    t0 = time.time()
    res = rh.solve_picard(
        G=args.G, taus=taus, gammas=gammas,
        umax=args.umax, Ws=Ws,
        maxiters=args.maxiters, abstol=args.abstol, alpha=args.alpha,
        P_init=P_init,
    )
    dt = time.time() - t0

    P = res["P_star"]
    F = res["residual"]
    Finf = float(np.abs(F).max())
    one_r2 = rh.one_minus_R2(P, res["u"], taus)
    n_iter = len(res["history"])

    print(f"\nDONE  Finf={Finf:.3e}  1-R²={one_r2:.4e}  "
          f"iters={n_iter}  t={dt:.1f}s  conv={res['converged']}", flush=True)

    record = {
        "schema": "rezn_n128/1",
        "P": P,
        "P_f128": P.astype(np.float128),
        "taus": taus,
        "gammas": gammas,
        "Ws": Ws,
        "G": int(args.G),
        "umax": float(args.umax),
        "Finf": Finf,
        "1-R²": one_r2,
        "one_minus_R2": one_r2,
        "kernel": "numba f64 (legacy_k3.rezn_het) linear-interp",
        "label": args.label,
        "history": [float(h) for h in res["history"]],
        "iters": n_iter,
        "alpha": float(args.alpha),
        "converged": bool(res["converged"]),
        "wall_clock_s": float(dt),
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(args.save_to, "wb") as f:
        pickle.dump(record, f)
    print(f"saved → {args.save_to}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
