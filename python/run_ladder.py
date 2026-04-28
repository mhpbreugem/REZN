#!/usr/bin/env python3
"""γ-ladder homotopy.

Sweep γ from a high (CARA-ish) value down to a low one. Each step warms
the solver with the previous step's converged P. All other parameters
(τ, G, umax) are held constant.

Saves a tensor pickle per step; logs all stdout to a single ladder log.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

import rezn_n128


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gammas", nargs="+", type=float, required=True,
                   help="ladder of γ values (homogeneous: each agent uses this γ)")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--tau", type=float, default=1.5,
                   help="constant τ used by every agent")
    p.add_argument("--G", type=int, default=7)
    p.add_argument("--umax", type=float, default=2.0)
    p.add_argument("--picard-iters", type=int, default=2000)
    p.add_argument("--picard-alpha0", type=float, default=1.0)
    p.add_argument("--picard-alpha-min", type=float, default=0.05)
    p.add_argument("--picard-alpha-max", type=float, default=1.0)
    p.add_argument("--target-finf", type=float, default=1e-10)
    p.add_argument("--out-prefix", required=True,
                   help="basename for saved tensors and log (no extension)")
    p.add_argument("--log-interval-s", type=float, default=15.0)
    args = p.parse_args(argv)

    K = args.K
    tau = args.tau
    Ws = np.ones(K, dtype=np.float64)
    taus = np.full(K, tau, dtype=np.float64)

    log_path = f"{args.out_prefix}.log"
    log_fh = open(log_path, "w")

    def log(msg):
        line = str(msg)
        sys.stdout.write(line + "\n"); sys.stdout.flush()
        log_fh.write(line + "\n"); log_fh.flush()

    log(f"=== γ-ladder homotopy ===")
    log(f"  K={K}  τ={tau}  G={args.G}  umax={args.umax}")
    log(f"  ladder: {args.gammas}")
    log(f"  out_prefix={args.out_prefix}")

    P_warm = None
    summary = []
    t_total = time.time()
    for step_i, gamma_val in enumerate(args.gammas):
        gammas = np.full(K, gamma_val, dtype=np.float64)
        log(f"\n=== step {step_i}/{len(args.gammas)-1}  γ={gamma_val} ===")
        save_to = f"{args.out_prefix}_step{step_i:02d}_g{gamma_val}.pkl"
        sub_log = f"{args.out_prefix}_step{step_i:02d}_g{gamma_val}.log"
        t_step = time.time()

        res = rezn_n128.solve(
            gammas=gammas, taus=taus, Ws=Ws,
            G=args.G, umax=args.umax,
            P_init=P_warm,
            picard_iters=args.picard_iters,
            picard_alpha0=args.picard_alpha0,
            picard_alpha_min=args.picard_alpha_min,
            picard_alpha_max=args.picard_alpha_max,
            lm_iters=0, tsvd_iters=0,
            target_finf=args.target_finf,
            log_path=sub_log,
            log_interval_s=args.log_interval_s,
            save_to=save_to,
            label=f"ladder step {step_i} γ={gamma_val}",
        )
        dt = time.time() - t_step

        log(f"  step {step_i} done: γ={gamma_val}  Finf={res['Finf']:.3e}  "
            f"1-R²={res['one_minus_R2']:.6e}  "
            f"converged={res['Finf'] < args.target_finf}  "
            f"t={dt:.1f}s  saved={save_to}")
        summary.append(dict(
            step=step_i, gamma=gamma_val,
            Finf=res["Finf"], one_minus_R2=res["one_minus_R2"],
            converged=bool(res["Finf"] < args.target_finf),
            dt=dt, path=save_to,
        ))
        # warm-start next step from this step's converged P
        P_warm = res["P_f128"]

    log(f"\n=== ladder summary ===")
    log(f"  step  γ          Finf         1-R²         conv?    t(s)")
    for s in summary:
        log(f"  {s['step']:4d}  {s['gamma']:<9}  {s['Finf']:.3e}  "
            f"{s['one_minus_R2']:.4e}  {s['converged']!s:5}  {s['dt']:.1f}")
    log(f"  total elapsed: {time.time() - t_total:.1f}s")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
