#!/usr/bin/env python3
"""Heterogeneous-gamma full REE contour solver for K=3.

Same algorithm as python/full_ree_solver.py but agents have per-agent
gammas (gamma1, gamma2, gamma3). The contour map and Bayesian step are
gamma-free; gammas enter only at market clearing.
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


def signal_density(u, v, tau):
    mean = v - 0.5
    return math.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


def crra_demand(mu, p, gamma, wealth=1.0):
    r = math.exp((logit(mu) - logit(p)) / gamma)
    return wealth * (r - 1.0) / ((1.0 - p) + r * p)


def clear_market_het(mus, gammas):
    """Solve sum_k x_k(mu_k, p, gamma_k) = 0 by bisection on (0,1)."""
    mus = tuple(float(np.clip(mu, EPS, 1.0 - EPS)) for mu in mus)

    def excess(p):
        return sum(crra_demand(mu, p, g) for mu, g in zip(mus, gammas))

    lo, hi = 1.0e-8, 1.0 - 1.0e-8
    flo, fhi = excess(lo), excess(hi)
    if flo < 0 or fhi > 0:
        raise RuntimeError(f"market-clearing bracket failed: f(lo)={flo}, f(hi)={fhi}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = excess(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _axis_crossings(values, grid, target):
    """Mirror of full_ree_solver._axis_crossings: linear root-find with
    exact-hit deduplication."""
    roots = []
    diff = values - target
    exact = np.flatnonzero(np.abs(diff) < 1.0e-12)
    roots.extend(float(grid[int(a)]) for a in exact)
    signs = diff[:-1] * diff[1:]
    for a in np.flatnonzero(signs < 0.0):
        denom = float(values[a + 1] - values[a])
        if abs(denom) < 1.0e-14:
            root = float(grid[a + 1])
        else:
            root = float(grid[a] + (target - values[a]) * (grid[a + 1] - grid[a]) / denom)
        roots.append(root)
    roots.sort()
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1.0e-10:
            out.append(r)
    return out


def _f_scalar(u, v, tau):
    mean = v - 0.5
    return math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (u - mean) ** 2)


def contour_evidence(slice_2d, grid, p, tau):
    """Mirror of full_ree_solver.contour_evidence: averages over hits."""
    sums = np.zeros(2, dtype=float)
    hits = 0
    for a, ua in enumerate(grid):
        f0a = _f_scalar(float(ua), 0, tau)
        f1a = _f_scalar(float(ua), 1, tau)
        for ub in _axis_crossings(slice_2d[a, :], grid, p):
            sums[0] += f0a * _f_scalar(ub, 0, tau)
            sums[1] += f1a * _f_scalar(ub, 1, tau)
            hits += 1
    for b, ub in enumerate(grid):
        f0b = _f_scalar(float(ub), 0, tau)
        f1b = _f_scalar(float(ub), 1, tau)
        for ua in _axis_crossings(slice_2d[:, b], grid, p):
            sums[0] += _f_scalar(ua, 0, tau) * f0b
            sums[1] += _f_scalar(ua, 1, tau) * f1b
            hits += 1
    if hits == 0:
        return 1.0, 1.0
    return float(sums[0] / hits), float(sums[1] / hits)


def _symmetrize(P):
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    return sum(np.transpose(P, axes=pp) for pp in perms) / len(perms)


def phi_het(P, grid, tau, gammas, symmetric=False):
    """One Φ evaluation. If symmetric=True (only valid when all gammas equal),
    average over the 6 permutations of (i,j,k)."""
    G = len(grid)
    P_new = np.empty_like(P)
    M1 = np.empty_like(P); M2 = np.empty_like(P); M3 = np.empty_like(P)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = float(P[i, j, k])
                u1, u2, u3 = float(grid[i]), float(grid[j]), float(grid[k])
                a0_1, a1_1 = contour_evidence(P[i, :, :], grid, p, tau)
                a0_2, a1_2 = contour_evidence(P[:, j, :], grid, p, tau)
                a0_3, a1_3 = contour_evidence(P[:, :, k], grid, p, tau)

                def post(uo, A0, A1):
                    f0 = _f_scalar(uo, 0, tau)
                    f1 = _f_scalar(uo, 1, tau)
                    denom = f0 * A0 + f1 * A1
                    if denom <= 0:
                        return 0.5
                    return float(np.clip(f1 * A1 / denom, EPS, 1.0 - EPS))

                mu1 = post(u1, a0_1, a1_1)
                mu2 = post(u2, a0_2, a1_2)
                mu3 = post(u3, a0_3, a1_3)
                M1[i, j, k] = mu1; M2[i, j, k] = mu2; M3[i, j, k] = mu3
                P_new[i, j, k] = clear_market_het([mu1, mu2, mu3], gammas)
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
                if not (1e-9 < p < 1 - 1e-9):
                    continue
                T = tau * (grid[i] + grid[j] + grid[k])
                f1i = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[i] - 0.5) ** 2)
                f1j = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[j] - 0.5) ** 2)
                f1k = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[k] - 0.5) ** 2)
                f0i = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[i] + 0.5) ** 2)
                f0j = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[j] + 0.5) ** 2)
                f0k = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (grid[k] + 0.5) ** 2)
                w = 0.5 * (f1i * f1j * f1k + f0i * f0j * f0k)
                Y.append(logit(p)); X.append(T); W.append(w)
    Y = np.array(Y); X = np.array(X); W = np.array(W)
    W = W / W.sum()
    Yb = (W * Y).sum(); Xb = (W * X).sum()
    cov = (W * (Y - Yb) * (X - Xb)).sum()
    vy = (W * (Y - Yb) ** 2).sum()
    vx = (W * (X - Xb) ** 2).sum()
    R2 = cov ** 2 / (vy * vx) if vy * vx > 0 else 0.0
    slope = cov / vx if vx > 0 else 0.0
    return 1.0 - R2, slope


def picard_anderson_het(P0, grid, tau, gammas, damping=0.3, anderson=5,
                        anderson_beta=1.0, max_iter=600, tol=1e-15,
                        progress=False, symmetric=False):
    """Mirror of full_ree_solver.picard_iterate with per-agent gammas."""
    P = P0.copy()
    history = []
    x_hist = []
    f_hist = []
    for it in range(1, max_iter + 1):
        candidate, _ = phi_het(P, grid, tau, gammas, symmetric=symmetric)
        F = candidate - P
        residual = float(np.max(np.abs(F)))
        history.append(residual)
        if residual < tol:
            return candidate, history, True
        relaxed = (1.0 - damping) * P + damping * candidate
        if anderson > 0:
            x_flat = P.ravel()
            f_flat = F.ravel()
            x_hist.append(x_flat.copy())
            f_hist.append(f_flat.copy())
            if len(f_hist) > anderson + 1:
                x_hist.pop(0); f_hist.pop(0)
            if len(f_hist) >= 2:
                df = np.column_stack([f_hist[q + 1] - f_hist[q] for q in range(len(f_hist) - 1)])
                dx = np.column_stack([x_hist[q + 1] - x_hist[q] for q in range(len(x_hist) - 1)])
                try:
                    coef, *_ = np.linalg.lstsq(df, f_flat, rcond=None)
                    aa_flat = x_flat + f_flat - (dx + df) @ coef
                    if np.all(np.isfinite(aa_flat)):
                        relaxed = (1.0 - anderson_beta) * relaxed + anderson_beta * aa_flat.reshape(P.shape)
                except np.linalg.LinAlgError:
                    pass
        P = np.clip(relaxed, 1.0e-8, 1.0 - 1.0e-8)
        if progress and (it % 25 == 0 or it == 1):
            print(f"  iter={it} resid={residual:.4e}", flush=True)
    return P, history, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--umax", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gammas", type=str, required=True,
                    help="comma-separated three gammas, e.g. 1.5,2.0,2.5")
    ap.add_argument("--seed-array", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-14)
    ap.add_argument("--damping", type=float, default=0.3)
    ap.add_argument("--anderson", type=int, default=5)
    ap.add_argument("--anderson-beta", type=float, default=0.7)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--symmetric", action="store_true",
                    help="symmetrize over (i,j,k) permutations after each Phi (only valid if all gammas equal)")
    ap.add_argument("--outdir", type=str,
                    default="results/full_ree")
    args = ap.parse_args()

    gammas = [float(x) for x in args.gammas.split(",")]
    if len(gammas) != 3:
        raise SystemExit("--gammas must be 3 values")

    grid = np.linspace(-args.umax, args.umax, args.G)
    seed = np.load(args.seed_array)
    P0 = seed["P"]
    src_grid = seed["grid"]
    if P0.shape != (args.G, args.G, args.G):
        # tri-linear interpolate
        from scipy.ndimage import map_coordinates
        coords = np.array([(grid - src_grid[0]) / (src_grid[-1] - src_grid[0]) * (len(src_grid) - 1)] * 3)
        I, J, K = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        P0 = map_coordinates(P0, [I, J, K], order=1, mode="nearest")
    P0 = np.clip(P0, 1e-5, 1 - 1e-5)

    t0 = time.time()
    P_final, hist, converged = picard_anderson_het(
        P0, grid, args.tau, gammas,
        damping=args.damping, anderson=args.anderson,
        anderson_beta=args.anderson_beta, max_iter=args.max_iter,
        tol=args.tol, progress=args.progress, symmetric=args.symmetric,
    )
    elapsed = time.time() - t0

    R2def, slope = deficit(P_final, grid, args.tau)

    # Find canonical realization (closest to (1,-1,1))
    i_p1 = int(np.argmin(np.abs(grid - 1.0)))
    i_m1 = int(np.argmin(np.abs(grid + 1.0)))

    # FR price at the same realization: Lambda(T*)
    T_can = args.tau * (grid[i_p1] + grid[i_m1] + grid[i_p1])
    fr_can = float(logistic(T_can))

    # Posteriors at canonical
    _, posts = phi_het(P_final, grid, args.tau, gammas)
    M1, M2, M3 = posts

    # Max FR error across grid
    P_fr = logistic(args.tau * (grid[:, None, None] + grid[None, :, None] + grid[None, None, :]))
    max_fr = float(np.max(np.abs(P_final - P_fr)))

    summary = {
        "G": args.G, "umax": args.umax, "tau": args.tau,
        "gammas": gammas,
        "seed_array": args.seed_array,
        "label": args.label,
        "iterations": len(hist),
        "converged": converged,
        "residual_inf": hist[-1] if hist else None,
        "revelation_deficit": float(R2def),
        "slope": float(slope),
        "max_fr_error": max_fr,
        "elapsed_seconds": elapsed,
        "representative_realization": {
            "u": [float(grid[i_p1]), float(grid[i_m1]), float(grid[i_p1])],
            "price": float(P_final[i_p1, i_m1, i_p1]),
            "fr_price": fr_can,
            "posteriors": [float(M1[i_p1, i_m1, i_p1]),
                           float(M2[i_p1, i_m1, i_p1]),
                           float(M3[i_p1, i_m1, i_p1])],
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g_str = "_".join(f"{g:g}" for g in gammas)
    np.savez(outdir / f"G{args.G}_tau{args.tau:g}_het{g_str}_{args.label}_prices.npz",
             P=P_final, grid=grid)
    with open(outdir / f"G{args.G}_tau{args.tau:g}_het{g_str}_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
