"""Re-run specific taus from no-learning seed (no warm start) with full convergence.

Useful when warm-start gets stuck in a basin and needs to be reset.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .config import DTYPE
from .task3_g400_sweep import solve_one_tau, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--taus", type=str, required=True,
                    help="comma-separated taus to rerun (e.g. 1.5,7.0)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/full_ree"))
    ap.add_argument("--G-inner", type=int, default=14)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--u-inner-max", type=float, default=3.0)
    ap.add_argument("--u-outer-max", type=float, default=6.0)
    ap.add_argument("--kernel", default="cubic")
    ap.add_argument("--max-stages", type=int, default=5)
    ap.add_argument("--inner-max-iter", type=int, default=15)
    ap.add_argument("--inner-tol", type=float, default=1e-9)
    ap.add_argument("--presmooth-steps", type=int, default=12)
    ap.add_argument("--presmooth-alpha", type=float, default=0.05)
    ap.add_argument("--keep-best", action="store_true",
                    help="only overwrite checkpoint if F_inner improves")
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]

    for tau in taus:
        print(f"\n[fresh] tau={tau} from no-learning, max_stages={args.max_stages}",
              flush=True)
        P_inner, P_full, u_inner, info = solve_one_tau(
            tau, args.gamma,
            G_inner=args.G_inner, pad=args.pad,
            u_inner_max=args.u_inner_max, u_outer_max=args.u_outer_max,
            kernel=args.kernel, max_stages=args.max_stages,
            inner_max_iter=args.inner_max_iter, inner_tol=args.inner_tol,
            presmooth_steps=args.presmooth_steps,
            presmooth_alpha=args.presmooth_alpha,
            P_init_inner=None,  # no-learning seed
        )
        G_full = args.G_inner + 2 * args.pad
        du = u_inner[1] - u_inner[0]
        u_full = np.array([u_inner[0] + (q - args.pad) * du
                          for q in range(G_full)])

        print(f"[fresh] tau={tau} new F={info['F_inner_inf']:.3e} "
              f"f128_1mR2={info['deficit_f128_weighted_1mR2']:.5e} "
              f"wall={info['wall_seconds']:.1f}s", flush=True)

        # Compare to existing
        from .task3_g400_sweep import tau_filename
        fname = (f"task3_g{int(round(args.gamma * 100)):03d}"
                 f"_{tau_filename(tau)}_mp50.json")
        path = args.out_dir / fname
        if args.keep_best and path.exists():
            import json
            with path.open() as f:
                existing = json.load(f)
            existing_F = float(existing.get("F_inner_inf_final", float("inf")))
            new_F = info["F_inner_inf"]
            if new_F < existing_F * 0.9:
                print(f"[fresh] new F={new_F:.3e} < existing F={existing_F:.3e}, "
                      f"OVERWRITING")
                save_checkpoint(args.out_dir, args.gamma, tau, args.G_inner,
                                args.pad, args.kernel, args.u_inner_max,
                                args.u_outer_max, u_inner, u_full, P_inner,
                                info)
            else:
                print(f"[fresh] new F={new_F:.3e} >= existing F={existing_F:.3e}, "
                      f"keeping existing")
        else:
            save_checkpoint(args.out_dir, args.gamma, tau, args.G_inner,
                            args.pad, args.kernel, args.u_inner_max,
                            args.u_outer_max, u_inner, u_full, P_inner, info)


if __name__ == "__main__":
    main()
