"""mp100 Picard sharpening of existing float64 K=3 checkpoints.

For each existing task3_g400_t*_mp50.json checkpoint:
  1. Load P_inner (float64).
  2. Build full halo using mpmath no-learning at dps=100.
  3. Inject P_inner into the inner cube.
  4. Run Picard at mp100 until ||F||_inf < 1e-50 or max_iter.
  5. Overwrite checkpoint with mp100 P_inner (saved as full-precision
     decimal strings) plus the new tolerance metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mpmath as mp

from .mp_solver import (set_dps, init_no_learning_K3_mp, phi_K3_mp,
                         residual_inf, f_signal_mp, mp_array_to_strings)


def to_mpf(x):
    return mp.mpf(repr(float(x)))


def load_checkpoint_to_mp(path: Path, dps: int):
    set_dps(dps)
    with path.open() as f:
        d = json.load(f)
    G_inner = int(d["G_inner"])
    pad = int(d["pad"])
    G_full = G_inner + 2 * pad
    u_inner_max = mp.mpf(repr(float(d["u_inner_max"])))
    du = 2 * u_inner_max / (G_inner - 1)
    u_full = [-u_inner_max + (q - pad) * du for q in range(G_full)]
    P_inner = np.asarray(d["P_inner"])

    tau = mp.mpf(repr(float(d["tau"])))
    gamma = mp.mpf(repr(float(d["gamma"])))
    tau_vec = [tau, tau, tau]
    gamma_vec = [gamma, gamma, gamma]
    W_vec = [mp.mpf("1"), mp.mpf("1"), mp.mpf("1")]
    return d, G_inner, pad, G_full, u_full, tau_vec, gamma_vec, W_vec, P_inner


def replace_inner(P_full, P_inner_mp, inner_lo, inner_hi):
    G_inner = inner_hi - inner_lo
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_full[i + inner_lo][j + inner_lo][l + inner_lo] = \
                    P_inner_mp[i][j][l]


def picard_sharpen_one(checkpoint_path: Path, *, dps: int, tol_mp: str,
                       max_iter: int, log_prefix: str = ""):
    print(f"{log_prefix}=== {checkpoint_path.name} ===", flush=True)
    set_dps(dps)
    tol = mp.mpf(tol_mp)

    t_load = time.perf_counter()
    (meta, G_inner, pad, G_full, u_full, tau_vec, gamma_vec, W_vec,
     P_inner_f64) = load_checkpoint_to_mp(checkpoint_path, dps)
    inner_lo = pad
    inner_hi = pad + G_inner

    print(f"{log_prefix}  G_inner={G_inner} pad={pad} G_full={G_full} "
          f"tau={mp.nstr(tau_vec[0], 4)} gamma={mp.nstr(gamma_vec[0], 4)} "
          f"dps={dps}", flush=True)

    # Precompute densities (shared across phi steps)
    sqrt_tt2pi = tuple(mp.sqrt(tau_vec[k] / (2 * mp.pi)) for k in range(3))
    f1_grid = [[f_signal_mp(u_full[i], 1, tau_vec[k],
                            sqrt_tau_over_2pi=sqrt_tt2pi[k])
                for i in range(G_full)] for k in range(3)]
    f0_grid = [[f_signal_mp(u_full[i], 0, tau_vec[k],
                            sqrt_tau_over_2pi=sqrt_tt2pi[k])
                for i in range(G_full)] for k in range(3)]

    # Build no-learning halo (full grid)
    print(f"{log_prefix}  building no-learning halo at mp{dps} ...",
          flush=True)
    t_seed = time.perf_counter()
    P_full = init_no_learning_K3_mp(u_full, tau_vec, gamma_vec, W_vec,
                                     sqrt_tt2pi=sqrt_tt2pi)
    print(f"{log_prefix}  halo built in {time.perf_counter()-t_seed:.1f}s",
          flush=True)

    # Inject float64 inner cube
    P_inner_mp = [[[to_mpf(P_inner_f64[i][j][l])
                    for l in range(G_inner)] for j in range(G_inner)]
                  for i in range(G_inner)]
    replace_inner(P_full, P_inner_mp, inner_lo, inner_hi)

    print(f"{log_prefix}  load+seed total {time.perf_counter()-t_load:.1f}s",
          flush=True)

    # Picard iteration
    history = []
    F_inf = mp.mpf("inf")
    for it in range(1, max_iter + 1):
        t_iter = time.perf_counter()
        P_new = phi_K3_mp(P_full, u_full, inner_lo, inner_hi,
                          tau_vec, gamma_vec, W_vec,
                          f1_grid=f1_grid, f0_grid=f0_grid,
                          sqrt_tt2pi=sqrt_tt2pi)
        F_inf = residual_inf(P_new, P_full, inner_lo, inner_hi)
        elapsed = time.perf_counter() - t_iter
        history.append({"iter": it,
                        "F_inf": mp.nstr(F_inf, 6),
                        "elapsed_s": elapsed})
        print(f"{log_prefix}  iter {it:2d}  ||F||={mp.nstr(F_inf, 4)}  "
              f"elapsed={elapsed:.1f}s", flush=True)
        # Pure Picard: P = phi(P)
        P_full = P_new
        if F_inf < tol:
            print(f"{log_prefix}  CONVERGED ||F||<{mp.nstr(tol, 2)}",
                  flush=True)
            break

    # Extract inner cube
    P_inner_mp_final = [[[P_full[i + inner_lo][j + inner_lo][l + inner_lo]
                          for l in range(G_inner)] for j in range(G_inner)]
                        for i in range(G_inner)]

    # Update meta
    meta["dps"] = dps
    meta["F_inner_inf_final"] = float(F_inf)
    meta["F_inner_inf_final_str"] = mp.nstr(F_inf, dps)
    meta["method"] = "K3_staggered_NK_float64_then_mp100_picard_sharpening"
    meta["sharpening_iters"] = len(history)
    meta["sharpening_history"] = history
    meta["P_inner_strings"] = mp_array_to_strings(P_inner_mp_final)

    # Recompute weighted 1-R^2 (longdouble) on the new P_inner
    from .f128 import revelation_deficit_f128
    P_inner_f64_new = np.array([[[float(P_inner_mp_final[i][j][l])
                                  for l in range(G_inner)]
                                 for j in range(G_inner)]
                                for i in range(G_inner)])
    u_inner = np.array([float(u_full[q]) for q in range(inner_lo, inner_hi)])
    tau_arr = np.array([float(tau_vec[k]) for k in range(3)])
    one_mR2 = float(revelation_deficit_f128(P_inner_f64_new, u_inner,
                                            tau_arr, 3))
    meta["weighted_1mR2_f128"] = one_mR2
    # Recompute weighted slope (longdouble masked, see recompute_summary)
    from .recompute_summary import weighted_slope_f128
    slope = weighted_slope_f128(P_inner_f64_new, u_inner, float(tau_vec[0]))
    meta["weighted_slope_f128"] = slope
    meta["weighted_1mR2"] = one_mR2  # keep convention

    # Update P_inner stored as floats (lossy but useful)
    meta["P_inner"] = P_inner_f64_new.tolist()

    # Save
    with checkpoint_path.open("w") as f:
        json.dump(meta, f)
    print(f"{log_prefix}  wrote {checkpoint_path}  "
          f"final ||F||={mp.nstr(F_inf, 4)}  "
          f"1-R^2={one_mR2:.5e}", flush=True)

    return F_inf, one_mR2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/full_ree"))
    ap.add_argument("--dps", type=int, default=100)
    ap.add_argument("--tol", type=str, default="1e-50")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--only-tau", type=str, default="",
                    help="comma-separated taus to process (default: all)")
    args = ap.parse_args()

    g_tag = f"g{int(round(args.gamma * 100)):03d}"
    cps = sorted(args.out_dir.glob(f"task3_{g_tag}_t*_mp50.json"))
    if not cps:
        raise SystemExit(f"No checkpoints found in {args.out_dir}")

    only = set()
    if args.only_tau:
        only = {float(t) for t in args.only_tau.split(",")}

    for cp in cps:
        with cp.open() as f:
            d = json.load(f)
        if only and float(d["tau"]) not in only:
            continue
        picard_sharpen_one(cp, dps=args.dps, tol_mp=args.tol,
                           max_iter=args.max_iter)


if __name__ == "__main__":
    main()
