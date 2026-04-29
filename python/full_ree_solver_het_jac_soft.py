#!/usr/bin/env python3
"""Soft-Jacobian variant of full_ree_solver_het_jac.py.

The hard Jacobian 1/|slope| diverges where the price function is locally
flat in the chosen axis. That singularity makes the Picard and Newton
iterations stiff. Replace with the soft kernel
    1 / sqrt(slope^2 + reg^2)
which equals 1/|slope| when slope >> reg and is bounded by 1/reg
elsewhere. Set reg to a small fraction of typical |slope| values.

Default reg = 0.01 (slope is ∂P/∂u ~ 0.1-0.4 for typical CRRA cells, so
reg=0.01 only kicks in at near-flat ones).
"""

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import sys

# Reuse model primitives + market clear from het_jac
sys.path.insert(0, str(Path(__file__).parent))
from full_ree_solver_het_jac import (
    EPS, logistic, logit, _f_scalar, _f_grid, crra_demand, clear_market_het,
    _symmetrize, deficit, picard_anderson_het as _picard_unused,
    newton_krylov_het as _newton_unused,
    _gmres,
)


def contour_evidence_jac_soft(slice_2d, grid, target_p, tau, reg):
    n = len(grid)
    du = float(grid[1] - grid[0])
    A0_a, A1_a, A0_b, A1_b = 0.0, 0.0, 0.0, 0.0

    for a in range(n):
        f0a = _f_scalar(float(grid[a]), 0, tau)
        f1a = _f_scalar(float(grid[a]), 1, tau)
        row = slice_2d[a, :]
        diff = row - target_p
        for j in range(n - 1):
            d0, d1 = diff[j], diff[j + 1]
            cell_diff = float(row[j + 1] - row[j])
            if d0 * d1 < 0:
                slope = cell_diff / du
                weight = 1.0 / math.sqrt(slope * slope + reg * reg)
                t = float(-d0 / (d1 - d0))
                b_star = float(grid[j]) + t * du
                fb0 = _f_scalar(b_star, 0, tau)
                fb1 = _f_scalar(b_star, 1, tau)
                A0_a += f0a * fb0 * weight
                A1_a += f1a * fb1 * weight
            elif abs(d0) < 1e-14:
                slope = cell_diff / du
                weight = 1.0 / math.sqrt(slope * slope + reg * reg)
                fb0 = _f_scalar(float(grid[j]), 0, tau)
                fb1 = _f_scalar(float(grid[j]), 1, tau)
                A0_a += 0.5 * f0a * fb0 * weight
                A1_a += 0.5 * f1a * fb1 * weight
    A0_a *= du; A1_a *= du

    for b in range(n):
        f0b = _f_scalar(float(grid[b]), 0, tau)
        f1b = _f_scalar(float(grid[b]), 1, tau)
        col = slice_2d[:, b]
        diff = col - target_p
        for i in range(n - 1):
            d0, d1 = diff[i], diff[i + 1]
            cell_diff = float(col[i + 1] - col[i])
            if d0 * d1 < 0:
                slope = cell_diff / du
                weight = 1.0 / math.sqrt(slope * slope + reg * reg)
                t = float(-d0 / (d1 - d0))
                a_star = float(grid[i]) + t * du
                fa0 = _f_scalar(a_star, 0, tau)
                fa1 = _f_scalar(a_star, 1, tau)
                A0_b += fa0 * f0b * weight
                A1_b += fa1 * f1b * weight
            elif abs(d0) < 1e-14:
                slope = cell_diff / du
                weight = 1.0 / math.sqrt(slope * slope + reg * reg)
                fa0 = _f_scalar(float(grid[i]), 0, tau)
                fa1 = _f_scalar(float(grid[i]), 1, tau)
                A0_b += 0.5 * fa0 * f0b * weight
                A1_b += 0.5 * fa1 * f1b * weight
    A0_b *= du; A1_b *= du

    return 0.5 * (A0_a + A0_b), 0.5 * (A1_a + A1_b)


def phi_het(P, grid, tau, gammas, reg, symmetric=False):
    G = len(grid)
    P_new = np.empty_like(P)
    M1 = np.empty_like(P); M2 = np.empty_like(P); M3 = np.empty_like(P)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = float(P[i, j, k])
                u1, u2, u3 = float(grid[i]), float(grid[j]), float(grid[k])
                a0_1, a1_1 = contour_evidence_jac_soft(P[i, :, :], grid, p, tau, reg)
                a0_2, a1_2 = contour_evidence_jac_soft(P[:, j, :], grid, p, tau, reg)
                a0_3, a1_3 = contour_evidence_jac_soft(P[:, :, k], grid, p, tau, reg)

                def post(uo, A0, A1):
                    f0 = _f_scalar(uo, 0, tau)
                    f1 = _f_scalar(uo, 1, tau)
                    denom = f0 * A0 + f1 * A1
                    if denom <= 0:
                        return 0.5
                    return float(np.clip(f1 * A1 / denom, EPS, 1 - EPS))

                mu1 = post(u1, a0_1, a1_1)
                mu2 = post(u2, a0_2, a1_2)
                mu3 = post(u3, a0_3, a1_3)
                M1[i,j,k] = mu1; M2[i,j,k] = mu2; M3[i,j,k] = mu3
                P_new[i,j,k] = clear_market_het([mu1, mu2, mu3], gammas)
    if symmetric:
        P_new = _symmetrize(P_new)
    return P_new, (M1, M2, M3)


def picard_anderson(P0, grid, tau, gammas, reg, damping=0.3, anderson=5,
                    anderson_beta=1.0, max_iter=600, tol=1e-15,
                    progress=False, symmetric=False):
    P = P0.copy(); history = []
    x_hist, f_hist = [], []
    for it in range(1, max_iter + 1):
        cand, _ = phi_het(P, grid, tau, gammas, reg, symmetric=symmetric)
        F = cand - P
        residual = float(np.max(np.abs(F)))
        history.append(residual)
        if residual < tol:
            return cand, history, True
        relaxed = (1 - damping) * P + damping * cand
        if anderson > 0:
            x_hist.append(P.ravel().copy()); f_hist.append(F.ravel().copy())
            if len(f_hist) > anderson + 1:
                x_hist.pop(0); f_hist.pop(0)
            if len(f_hist) >= 2:
                df = np.column_stack([f_hist[q+1] - f_hist[q] for q in range(len(f_hist)-1)])
                dx = np.column_stack([x_hist[q+1] - x_hist[q] for q in range(len(x_hist)-1)])
                try:
                    coef, *_ = np.linalg.lstsq(df, F.ravel(), rcond=None)
                    aa = P.ravel() + F.ravel() - (dx + df) @ coef
                    if np.all(np.isfinite(aa)):
                        relaxed = (1 - anderson_beta) * relaxed + anderson_beta * aa.reshape(P.shape)
                except np.linalg.LinAlgError:
                    pass
        P = np.clip(relaxed, 1e-8, 1 - 1e-8)
        if progress and (it % 25 == 0 or it == 1):
            print(f"  iter={it} resid={residual:.4e}", flush=True)
    return P, history, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--umax", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gammas", type=str, required=True)
    ap.add_argument("--seed-array", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-14)
    ap.add_argument("--damping", type=float, default=0.3)
    ap.add_argument("--anderson", type=int, default=5)
    ap.add_argument("--anderson-beta", type=float, default=0.7)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--outdir", type=str, default="results/full_ree")
    args = ap.parse_args()

    gammas = [float(x) for x in args.gammas.split(",")]
    grid = np.linspace(-args.umax, args.umax, args.G)
    seed = np.load(args.seed_array)
    P0 = seed["P"]; src_grid = seed["grid"]
    if P0.shape != (args.G, args.G, args.G):
        from scipy.ndimage import map_coordinates
        coords = (grid - src_grid[0]) / (src_grid[-1] - src_grid[0]) * (len(src_grid) - 1)
        I, J, K = np.meshgrid(coords, coords, coords, indexing="ij")
        P0 = map_coordinates(P0, [I, J, K], order=1, mode="nearest")
    P0 = np.clip(P0, 1e-8, 1 - 1e-8)

    t0 = time.time()
    P_final, hist, conv = picard_anderson(
        P0, grid, args.tau, gammas, args.reg,
        damping=args.damping, anderson=args.anderson,
        anderson_beta=args.anderson_beta, max_iter=args.max_iter,
        tol=args.tol, progress=args.progress, symmetric=args.symmetric)
    elapsed = time.time() - t0

    R2def, slope = deficit(P_final, grid, args.tau)
    i_p1 = int(np.argmin(np.abs(grid - 1.0)))
    i_m1 = int(np.argmin(np.abs(grid + 1.0)))
    T_can = args.tau * (grid[i_p1] + grid[i_m1] + grid[i_p1])
    fr_can = float(logistic(T_can))
    _, posts = phi_het(P_final, grid, args.tau, gammas, args.reg, symmetric=args.symmetric)
    M1, M2, M3 = posts
    P_fr = logistic(args.tau * (grid[:, None, None] + grid[None, :, None] + grid[None, None, :]))
    max_fr = float(np.max(np.abs(P_final - P_fr)))

    summary = {
        "G": args.G, "umax": args.umax, "tau": args.tau,
        "gammas": gammas, "method": "picard_jac_soft", "reg": args.reg,
        "seed_array": args.seed_array, "label": args.label,
        "iterations": len(hist), "converged": conv,
        "residual_inf": hist[-1] if hist else None,
        "revelation_deficit": float(R2def), "slope": float(slope),
        "max_fr_error": max_fr, "elapsed_seconds": elapsed,
        "representative_realization": {
            "u": [float(grid[i_p1]), float(grid[i_m1]), float(grid[i_p1])],
            "price": float(P_final[i_p1, i_m1, i_p1]),
            "fr_price": fr_can,
            "posteriors": [float(M1[i_p1, i_m1, i_p1]),
                           float(M2[i_p1, i_m1, i_p1]),
                           float(M3[i_p1, i_m1, i_p1])],
        },
    }
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    g_str = "_".join(f"{g:g}" for g in gammas)
    np.savez(outdir / f"G{args.G}_tau{args.tau:g}_jacsoft_reg{args.reg:g}_het{g_str}_{args.label}_prices.npz",
             P=P_final, grid=grid)
    with open(outdir / f"G{args.G}_tau{args.tau:g}_jacsoft_reg{args.reg:g}_het{g_str}_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
