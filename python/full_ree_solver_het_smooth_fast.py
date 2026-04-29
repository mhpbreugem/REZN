#!/usr/bin/env python3
"""Vectorized version of full_ree_solver_het_smooth.

Replaces the triple-loop phi (O(G^5) Python ops) with a single numpy
broadcasting expression (O(G^4) vectorized). Should give ~10-100x speedup
at G≥9 and make G=12, G=15 viable.

Other than the phi vectorization, the logic is identical to the loop version:
same kernel, same Bayes step, same market clearing.
"""

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np

EPS = 1.0e-10


def logistic(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return float(out) if out.shape == () else out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    out = np.log(p / (1.0 - p))
    return float(out) if out.shape == () else out


def f_density(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


def crra_demand(mu, p, gamma, wealth=1.0):
    r = math.exp((logit(mu) - logit(p)) / gamma)
    return wealth * (r - 1.0) / ((1.0 - p) + r * p)


def clear_market_het(mus, gammas):
    mus = tuple(float(np.clip(mu, EPS, 1.0 - EPS)) for mu in mus)
    def excess(p):
        return sum(crra_demand(mu, p, g) for mu, g in zip(mus, gammas))
    lo, hi = 1e-8, 1 - 1e-8
    flo, fhi = excess(lo), excess(hi)
    if flo < 0 or fhi > 0:
        raise RuntimeError(f"market-clearing bracket failed: f(lo)={flo}, f(hi)={fhi}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0: lo = mid
        else:               hi = mid
    return 0.5 * (lo + hi)


def _symmetrize(P):
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    return sum(np.transpose(P, axes=pp) for pp in perms) / len(perms)


def phi_het_fast(P, grid, tau, gammas, h, symmetric=False):
    """Vectorized smooth phi.

    For each agent k, compute the kernel-smoothed Bayesian posterior μ_k as
    a G×G×G array using numpy broadcasting:

        F0 = outer(f0_grid, f0_grid)  shape (G, G) -- prior signal density on the slice
        F1 = outer(f1_grid, f1_grid)
        slice = P with the own-axis fixed
        For each (i_own, j_other1, j_other2) we build K[i_own, j_other1, j_other2, a, b]:
            K = exp(-((slice[i_own, a, b] - P[i_own, j_other1, j_other2])/h)^2 / 2)
        Sum over (a, b): A_v[i_own, j_other1, j_other2] = einsum("...ab,ab->...", K, F_v)

    There are three agents; each picks its own axis. After computing all three
    posterior tensors and the per-cell market-clearing prices, symmetrize if
    requested.

    Returns (P_new, (mu1, mu2, mu3)).
    """
    G = len(grid)
    pref = (grid[1] - grid[0]) ** 2 / (h * math.sqrt(2 * math.pi))
    f0_grid = f_density(grid, 0, tau)
    f1_grid = f_density(grid, 1, tau)
    F0 = np.outer(f0_grid, f0_grid)
    F1 = np.outer(f1_grid, f1_grid)

    # Agent 1 (axis 0): for each i, slice is P[i, :, :] (G, G).
    # For each (i, j, k), p_target is P[i, j, k]. K depends on slice[i, a, b] - p_target.
    # Use broadcasting: P has shape (G, G, G). For agent 1, we need
    #   diff[i, j, k, a, b] = P[i, a, b] - P[i, j, k]
    # That's G^5 entries — manageable up to G ~ 13 (G^5 = 371k).
    # For larger G we'd want a more memory-efficient computation.
    # P_slice_a1[i, j, k, a, b] = P[i, a, b] (broadcast j, k away)
    # P_target  [i, j, k]       = P[i, j, k]

    # Agent 1
    diff1 = P[:, None, None, :, :] - P[:, :, :, None, None]   # (G,G,G,G,G)
    K1 = np.exp(-0.5 * (diff1 / h) ** 2)
    A0_1 = pref * np.einsum("ijkab,ab->ijk", K1, F0)
    A1_1 = pref * np.einsum("ijkab,ab->ijk", K1, F1)
    del diff1, K1

    # Agent 2 (axis 1): slice is P[:, j, :]. diff2[i, j, k, a, b] = P[a, j, b] - P[i, j, k]
    diff2 = P.transpose(1, 0, 2)[:, None, None, :, :].reshape(G, 1, 1, G, G) \
            - P.transpose(1, 0, 2)[:, :, None, :, None].reshape(G, G, 1, G, 1)  # tricky
    # Simpler: fully build it via broadcasting
    P_for2 = np.transpose(P, (1, 0, 2))  # P_for2[j, a, b] = P[a, j, b]
    diff2 = P_for2[:, None, None, :, :] - np.transpose(P, (1, 0, 2))[:, :, :, None, None].reshape(G, G, G, 1, 1)
    # Wait that has wrong p_target shape. Let me redo.
    # We want: for each (i, j, k): A_v[i, j, k] = sum_{a, b} K[a, b] F_v[a, b]
    # where K[a, b] = exp(-((P[a, j, b] - P[i, j, k])/h)^2 / 2).
    # P_for2[a, j, b] = P_for2[j, a, b] in transposed form.
    # diff[i, j, k, a, b] = P_for2[j, a, b] - P[i, j, k]
    p_target = P[:, :, :, None, None]  # (G, G, G, 1, 1)
    slice_a2 = np.transpose(P, (1, 0, 2))[None, :, None, :, :]  # P_for2[j, a, b], shape (1, G, 1, G, G)
    diff2 = slice_a2 - p_target  # (G, G, G, G, G)
    K2 = np.exp(-0.5 * (diff2 / h) ** 2)
    A0_2 = pref * np.einsum("ijkab,ab->ijk", K2, F0)
    A1_2 = pref * np.einsum("ijkab,ab->ijk", K2, F1)
    del diff2, K2

    # Agent 3 (axis 2): slice is P[:, :, k]. diff[i, j, k, a, b] = P[a, b, k] - P[i, j, k]
    slice_a3 = np.transpose(P, (2, 0, 1))[None, None, :, :, :]  # P_for3[k, a, b], shape (1, 1, G, G, G)
    diff3 = slice_a3 - p_target
    K3 = np.exp(-0.5 * (diff3 / h) ** 2)
    A0_3 = pref * np.einsum("ijkab,ab->ijk", K3, F0)
    A1_3 = pref * np.einsum("ijkab,ab->ijk", K3, F1)
    del diff3, K3

    # Per-cell own-signal densities
    f0_own1 = f0_grid[:, None, None]  # (G, 1, 1)
    f1_own1 = f1_grid[:, None, None]
    f0_own2 = f0_grid[None, :, None]
    f1_own2 = f1_grid[None, :, None]
    f0_own3 = f0_grid[None, None, :]
    f1_own3 = f1_grid[None, None, :]

    # Posteriors
    M1 = f1_own1 * A1_1 / np.maximum(f0_own1 * A0_1 + f1_own1 * A1_1, 1e-300)
    M2 = f1_own2 * A1_2 / np.maximum(f0_own2 * A0_2 + f1_own2 * A1_2, 1e-300)
    M3 = f1_own3 * A1_3 / np.maximum(f0_own3 * A0_3 + f1_own3 * A1_3, 1e-300)
    M1 = np.clip(M1, EPS, 1 - EPS)
    M2 = np.clip(M2, EPS, 1 - EPS)
    M3 = np.clip(M3, EPS, 1 - EPS)

    # Market clearing per cell — must be done in a loop (γ-dependent)
    P_new = np.empty_like(P)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = clear_market_het(
                    [float(M1[i, j, k]), float(M2[i, j, k]), float(M3[i, j, k])],
                    gammas)

    if symmetric:
        P_new = _symmetrize(P_new)
    return P_new, (M1, M2, M3)


def deficit(P, grid, tau):
    G = len(grid)
    Y, X, W = [], [], []
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                if not (1e-9 < p < 1 - 1e-9): continue
                T = tau * (grid[i] + grid[j] + grid[k])
                f1 = f_density(np.array([grid[i], grid[j], grid[k]]), 1, tau)
                f0 = f_density(np.array([grid[i], grid[j], grid[k]]), 0, tau)
                w = 0.5 * (np.prod(f1) + np.prod(f0))
                Y.append(logit(p)); X.append(T); W.append(w)
    Y, X, W = np.array(Y), np.array(X), np.array(W)
    W = W / W.sum()
    Yb = (W*Y).sum(); Xb = (W*X).sum()
    cov = (W*(Y-Yb)*(X-Xb)).sum()
    vy = (W*(Y-Yb)**2).sum()
    vx = (W*(X-Xb)**2).sum()
    R2 = cov**2/(vy*vx) if vy*vx > 0 else 0.0
    slope = cov/vx if vx > 0 else 0.0
    return 1.0 - R2, slope


def picard_anderson(P0, grid, tau, gammas, h, damping=0.3, anderson=5,
                    anderson_beta=1.0, max_iter=600, tol=1e-15,
                    progress=False, symmetric=False):
    P = P0.copy(); history = []
    x_hist, f_hist = [], []
    for it in range(1, max_iter + 1):
        cand, _ = phi_het_fast(P, grid, tau, gammas, h, symmetric=symmetric)
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
    ap.add_argument("--h", type=float, required=True)
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
        P0, grid, args.tau, gammas, args.h,
        damping=args.damping, anderson=args.anderson,
        anderson_beta=args.anderson_beta, max_iter=args.max_iter,
        tol=args.tol, progress=args.progress, symmetric=args.symmetric)
    elapsed = time.time() - t0

    R2def, slope = deficit(P_final, grid, args.tau)
    i_p1 = int(np.argmin(np.abs(grid - 1.0)))
    i_m1 = int(np.argmin(np.abs(grid + 1.0)))
    T_can = args.tau * (grid[i_p1] + grid[i_m1] + grid[i_p1])
    fr_can = float(logistic(T_can))
    _, posts = phi_het_fast(P_final, grid, args.tau, gammas, args.h, symmetric=args.symmetric)
    M1, M2, M3 = posts
    P_fr = logistic(args.tau * (grid[:, None, None] + grid[None, :, None] + grid[None, None, :]))
    max_fr = float(np.max(np.abs(P_final - P_fr)))

    summary = {
        "G": args.G, "umax": args.umax, "tau": args.tau,
        "gammas": gammas, "h": args.h, "method": "picard_smooth_fast",
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
    np.savez(outdir / f"G{args.G}_tau{args.tau:g}_smoothfasth{args.h:g}_het{g_str}_{args.label}_prices.npz",
             P=P_final, grid=grid)
    with open(outdir / f"G{args.G}_tau{args.tau:g}_smoothfasth{args.h:g}_het{g_str}_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
