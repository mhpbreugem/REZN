"""Re-derive fig4A summary from per-tau checkpoints using f128 weighted 1-R^2.

Usage:
    python -m code.recompute_summary --gamma 4.0 --out-dir results/full_ree
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .f128 import (revelation_deficit_f128, t_star_f128, weights_f128,
                   _safe_logit_f128, DTYPE_F128)


def weighted_slope_f128(P: np.ndarray, u_grid: np.ndarray, tau: float,
                        eps: float = 1.0e-4) -> float:
    tau_vec = np.asarray([tau, tau, tau], dtype=np.float64)
    Ts = t_star_f128(u_grid, tau_vec, 3)
    w = weights_f128(u_grid, tau_vec, 3)
    L = _safe_logit_f128(P)
    P128 = np.asarray(P, dtype=DTYPE_F128)
    eps128 = DTYPE_F128(eps)
    mask = (P128 > eps128) & (P128 < DTYPE_F128(1) - eps128)
    w = np.where(mask, w, DTYPE_F128(0))
    Wsum = w.sum(dtype=DTYPE_F128)
    L_mean = (w * L).sum(dtype=DTYPE_F128) / Wsum
    T_mean = (w * Ts).sum(dtype=DTYPE_F128) / Wsum
    var_T = (w * (Ts - T_mean) ** 2).sum(dtype=DTYPE_F128) / Wsum
    cov = (w * (L - L_mean) * (Ts - T_mean)).sum(dtype=DTYPE_F128) / Wsum
    return float(cov / var_T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/full_ree"))
    args = ap.parse_args()

    g_tag = f"g{int(round(args.gamma * 100)):03d}"
    cp_glob = sorted(args.out_dir.glob(f"task3_{g_tag}_t*_mp50.json"))
    if not cp_glob:
        raise SystemExit(f"No task3_{g_tag}_t*_mp50.json checkpoints found "
                         f"in {args.out_dir}")

    points = []
    for cp in cp_glob:
        with cp.open() as f:
            d = json.load(f)
        P = np.asarray(d["P_inner"])
        u = np.asarray(d["u_inner"])
        tau = float(d["tau"])
        slope = weighted_slope_f128(P, u, tau)
        # Use f128 deficit if available; recompute otherwise.
        if "weighted_1mR2_f128" in d:
            one_mR2 = float(d["weighted_1mR2_f128"])
        else:
            tau_vec = np.array([tau, tau, tau])
            one_mR2 = float(revelation_deficit_f128(P, u, tau_vec, 3))
        # Also patch the per-tau checkpoint with masked weighted slope
        d["weighted_slope_f128"] = slope
        d["weighted_1mR2"] = one_mR2  # overwrite buggy value with f128
        with cp.open("w") as f:
            json.dump(d, f)
        points.append({
            "tau": tau,
            "1-R2": one_mR2,
            "slope": slope,
            "F_inf": float(d.get("F_inner_inf_final", float("nan"))),
            "n_stages": int(d.get("n_stages", -1)),
        })

    points.sort(key=lambda p: p["tau"])

    # Read params from one checkpoint
    with cp_glob[0].open() as f:
        d0 = json.load(f)

    summary = {
        "figure": "fig4A",
        "gamma": args.gamma,
        "params": {
            "G_inner": d0.get("G_inner"),
            "pad": d0.get("pad"),
            "u_inner_max": d0.get("u_inner_max"),
            "u_outer_max": d0.get("u_outer_max"),
            "kernel": d0.get("kernel"),
            "method": d0.get("method"),
            "weighting": "ex-ante 0.5*(f0^3 + f1^3), mask P in (1e-4, 1-1e-4)",
            "metric": "weighted_1mR2_f128 (longdouble), masked",
        },
        "points": points,
    }
    out_path = args.out_dir / (
        f"fig4A_{g_tag}_tau_sweep.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Also pgfplots
    tex_path = args.out_dir / f"fig4A_{g_tag}_pgfplots.tex"
    with tex_path.open("w") as f:
        f.write(f"% gamma = {args.gamma}\n")
        f.write(f"% method: K=3 staggered Newton-Krylov, "
                f"G_inner={d0.get('G_inner')}, pad={d0.get('pad')}, "
                f"kernel={d0.get('kernel')}\n")
        f.write(f"% weighted 1-R^2 (longdouble, masked)\n")
        coords = " ".join(f"({p['tau']:g},{p['1-R2']:.6e})" for p in points)
        f.write(f"\\addplot coordinates {{{coords}}};\n")
        f.write(f"% slopes per tau:\n")
        coords_s = " ".join(f"({p['tau']:g},{p['slope']:.6f})" for p in points)
        f.write(f"% \\addplot coordinates {{{coords_s}}};\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {tex_path}")
    print(f"\nSummary:")
    print(f"  {'tau':>7s}  {'1-R^2':>12s}  {'slope':>8s}  "
          f"{'F_inf':>9s}  {'stages':>6s}")
    for p in points:
        print(f"  {p['tau']:7.2f}  {p['1-R2']:12.5e}  {p['slope']:8.4f}  "
              f"{p['F_inf']:9.2e}  {p['n_stages']:6d}")


if __name__ == "__main__":
    main()
