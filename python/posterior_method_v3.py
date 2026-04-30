#!/usr/bin/env python3
"""Posterior-function method v3 per SYSTEM_EQUATIONS.md.

Stores μ[i,j] on a per-row p-grid: each row i has its own p-range
[p_lo(u_i), p_hi(u_i)] of *achievable* prices (lens-shaped domain).
Avoids the v1 problem of storing μ on extreme-p cells where the
contour is empty.

Per-iteration cost is O(G_u² · G_p) with all heavy lifting vectorized
(np.interp for the inverse-demand inversion).
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

EPS = 1e-10


def Lam(z):
    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        v = float(z)
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    out = np.log(p / (1.0 - p))
    return float(out) if out.shape == () else out


def f_v(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


def crra_demand_vec(mu, p, gamma, W=1.0):
    """CRRA demand at posterior μ and price p (both can be arrays)."""
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)


def market_clear_no_learning(u_triple, tau, gamma):
    """Solve sum_k x(Λ(τu_k), p) = 0 for p (no-learning case)."""
    mus = [Lam(tau * u) for u in u_triple]
    def Z(p):
        return sum(crra_demand_vec(np.array([m]), np.array([p]), gamma)[0] for m in mus)
    return brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)


def init_p_grid(u_grid, tau, gamma, Gp, margin=0.05):
    """For each u_i, compute p_lo and p_hi via no-learning market clearing
    with extreme-other-signal config, plus a small margin. Build the
    per-row p-grid in logit space.

    Returns:
        p_lo[Gu], p_hi[Gu], logit_p_grid[Gu, Gp], p_grid[Gu, Gp]
    """
    Gu = len(u_grid)
    u_min = u_grid[0]; u_max = u_grid[-1]
    p_lo = np.empty(Gu); p_hi = np.empty(Gu)
    for i, u_i in enumerate(u_grid):
        p_lo[i] = market_clear_no_learning((u_i, u_min, u_min), tau, gamma)
        p_hi[i] = market_clear_no_learning((u_i, u_max, u_max), tau, gamma)
        # Add margin in logit space
        l_lo = logit(p_lo[i]); l_hi = logit(p_hi[i])
        spread = l_hi - l_lo
        l_lo -= margin * spread; l_hi += margin * spread
        p_lo[i] = Lam(l_lo)
        p_hi[i] = Lam(l_hi)

    # Per-row p-grid in logit space
    logit_p_grid = np.empty((Gu, Gp))
    p_grid = np.empty((Gu, Gp))
    for i in range(Gu):
        l_lo = logit(p_lo[i]); l_hi = logit(p_hi[i])
        logit_p_grid[i, :] = np.linspace(l_lo, l_hi, Gp)
        p_grid[i, :] = Lam(logit_p_grid[i, :])
    return p_lo, p_hi, p_grid


def extract_mu_col(mu, p_grid, p0, u_grid, tau, p_lo, p_hi):
    """For target price p0, extract μ_col[i'] = μ(u_{i'}, p0).
    For rows where p0 is out of [p_lo[i'], p_hi[i']], use Λ(τu_{i'}) as boundary value.
    """
    Gu = len(u_grid)
    mu_col = np.empty(Gu)
    for i in range(Gu):
        if p0 < p_grid[i, 0]:
            mu_col[i] = mu[i, 0]      # clamp to lower edge of this row
        elif p0 > p_grid[i, -1]:
            mu_col[i] = mu[i, -1]     # clamp to upper edge
        else:
            mu_col[i] = float(np.interp(p0, p_grid[i, :], mu[i, :]))
    return np.clip(mu_col, EPS, 1 - EPS)


def phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """One Φ_μ evaluation per v3 §5/§6.

    Returns (mu_new, active_mask, n_crossings_per_cell).
    Active cells are those with ≥ 2 valid u₃* crossings.
    """
    Gu = len(u_grid); Gp = p_grid.shape[1]
    mu_new = mu.copy()
    active = np.zeros((Gu, Gp), dtype=bool)
    ncross = np.zeros((Gu, Gp), dtype=int)

    f1_grid = f_v(u_grid, 1, tau)
    f0_grid = f_v(u_grid, 0, tau)

    for i in range(Gu):
        for j in range(Gp):
            p0 = p_grid[i, j]

            # Step A: extract μ_col(u; p0)
            mu_col = extract_mu_col(mu, p_grid, p0, u_grid, tau, p_lo, p_hi)

            # Step B: demand array d[i'] at this price
            d = crra_demand_vec(mu_col, np.full_like(mu_col, p0), gamma)

            # Step C: contour. Sweep u₂ on u_grid; targets[i'] = D_i - d[i']
            D_i = -d[i]
            targets = D_i - d   # length Gu

            # interp_invert: d(u) is monotone-increasing in u (we hope).
            # Find u₃* such that d(u₃*) = targets[i']
            if d[-1] - d[0] < 1e-15:
                # d not monotone enough; skip
                continue

            # Ensure d is increasing for np.interp
            if d[-1] > d[0]:
                u3_star = np.interp(targets, d, u_grid,
                                    left=u_grid[0] - 1, right=u_grid[-1] + 1)
            else:
                u3_star = np.interp(targets, d[::-1], u_grid[::-1],
                                    left=u_grid[-1] + 1, right=u_grid[0] - 1)

            # Validity mask: u₃* must be inside u-grid
            valid = (u3_star >= u_grid[0]) & (u3_star <= u_grid[-1])
            n_valid = int(np.sum(valid))
            ncross[i, j] = n_valid

            if n_valid < 2:
                # Degenerate cell — skip update
                continue

            # Step D: signal densities
            f1_root = f_v(u3_star[valid], 1, tau)
            f0_root = f_v(u3_star[valid], 0, tau)
            f1_sweep = f1_grid[valid]
            f0_sweep = f0_grid[valid]

            # Step E: contour integrals
            A1 = float(np.sum(f1_sweep * f1_root))
            A0 = float(np.sum(f0_sweep * f0_root))

            # Step F: Bayes
            f1_own = float(f1_grid[i])
            f0_own = float(f0_grid[i])
            denom = f0_own * A0 + f1_own * A1
            if denom <= 0:
                continue
            mu_new[i, j] = float(np.clip(f1_own * A1 / denom, EPS, 1 - EPS))
            active[i, j] = True

    return mu_new, active, ncross


def picard_anderson(mu0, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                    damping=0.2, anderson=0, anderson_beta=0.7,
                    max_iter=200, tol=1e-8, progress=False):
    mu = mu0.copy(); history = []
    x_hist = []; f_hist = []
    for it in range(1, max_iter + 1):
        cand, active, ncross = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        F = cand - mu
        # Residual over active cells only
        if np.any(active):
            residual = float(np.max(np.abs(F[active])))
        else:
            residual = float("nan")
        history.append((residual, int(np.sum(active))))
        if progress and (it % 5 == 0 or it == 1):
            print(f"  iter={it} resid={residual:.4e} active={int(np.sum(active))}", flush=True)
        if residual < tol:
            return cand, history, True
        relaxed = (1 - damping) * mu + damping * cand
        if anderson > 0:
            x_hist.append(mu.ravel().copy()); f_hist.append(F.ravel().copy())
            if len(f_hist) > anderson + 1:
                x_hist.pop(0); f_hist.pop(0)
            if len(f_hist) >= 2:
                df = np.column_stack([f_hist[q+1] - f_hist[q] for q in range(len(f_hist)-1)])
                dx = np.column_stack([x_hist[q+1] - x_hist[q] for q in range(len(x_hist)-1)])
                try:
                    coef, *_ = np.linalg.lstsq(df, F.ravel(), rcond=None)
                    aa = mu.ravel() + F.ravel() - (dx + df) @ coef
                    if np.all(np.isfinite(aa)):
                        relaxed = (1 - anderson_beta) * relaxed + anderson_beta * aa.reshape(mu.shape)
                except np.linalg.LinAlgError:
                    pass
        mu = np.clip(relaxed, EPS, 1 - EPS)
    return mu, history, False


def measure_R2(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """Sample (u_1, u_2, u_3) realizations, solve market clearing for p,
    measure 1-R² and slope of logit(p) vs T*."""
    Y = []; X = []; W = []
    Gu = len(u_grid)

    for i in range(Gu):
        for j in range(Gu):
            for k in range(Gu):
                u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]

                def F(p):
                    mu1 = float(np.interp(p, p_grid[i], mu[i]) if p_lo[i] <= p <= p_hi[i] else Lam(tau*u1))
                    mu2 = float(np.interp(p, p_grid[j], mu[j]) if p_lo[j] <= p <= p_hi[j] else Lam(tau*u2))
                    mu3 = float(np.interp(p, p_grid[k], mu[k]) if p_lo[k] <= p <= p_hi[k] else Lam(tau*u3))
                    return sum(crra_demand_vec(np.array([m]), np.array([p]), gamma)[0]
                               for m in (mu1, mu2, mu3))

                lo, hi = 1e-6, 1 - 1e-6
                if F(lo) * F(hi) >= 0: continue
                for _ in range(80):
                    m = 0.5 * (lo + hi)
                    if F(m) > 0: lo = m
                    else: hi = m
                p_star = 0.5 * (lo + hi)
                T = tau * (u1 + u2 + u3)
                w = 0.5 * (f_v(u1, 1, tau) * f_v(u2, 1, tau) * f_v(u3, 1, tau)
                           + f_v(u1, 0, tau) * f_v(u2, 0, tau) * f_v(u3, 0, tau))
                Y.append(logit(p_star)); X.append(T); W.append(float(w))

    Y, X, W = np.array(Y), np.array(X), np.array(W)
    W = W / W.sum()
    Yb = (W*Y).sum(); Xb = (W*X).sum()
    cov = (W*(Y-Yb)*(X-Xb)).sum()
    vy = (W*(Y-Yb)**2).sum()
    vx = (W*(X-Xb)**2).sum()
    R2 = cov**2/(vy*vx) if vy*vx > 0 else 0.0
    slope = cov/vx if vx > 0 else 0.0
    return 1.0 - R2, slope, len(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Gu", type=int, default=20)
    ap.add_argument("--Gp", type=int, default=20)
    ap.add_argument("--umax", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--damping", type=float, default=0.2)
    ap.add_argument("--anderson", type=int, default=0)
    ap.add_argument("--anderson-beta", type=float, default=0.7)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--label", type=str, default="v3")
    ap.add_argument("--outdir", type=str, default="results/full_ree")
    args = ap.parse_args()

    u_grid = np.linspace(-args.umax, args.umax, args.Gu)
    print(f"Posterior-method v3 (adaptive p-grid):", flush=True)
    print(f"  Gu={args.Gu}, Gp={args.Gp}, umax={args.umax}", flush=True)
    print(f"  τ={args.tau}, γ={args.gamma}", flush=True)
    print(f"  damping={args.damping}, anderson={args.anderson}", flush=True)

    print(f"Computing per-row p-grid (no-learning prices)...", flush=True)
    t0 = time.time()
    p_lo, p_hi, p_grid = init_p_grid(u_grid, args.tau, args.gamma, args.Gp)
    print(f"  p-range at u=0: [{p_lo[args.Gu//2]:.4f}, {p_hi[args.Gu//2]:.4f}]", flush=True)
    print(f"  p-range at u=u_min: [{p_lo[0]:.4f}, {p_hi[0]:.4f}]", flush=True)
    print(f"  p-range at u=u_max: [{p_lo[-1]:.4f}, {p_hi[-1]:.4f}]", flush=True)
    print(f"  ({time.time() - t0:.1f}s)", flush=True)

    # Initialize μ⁰(u, p) = Λ(τu)
    mu0 = np.zeros((args.Gu, args.Gp))
    for i, u in enumerate(u_grid):
        mu0[i, :] = Lam(args.tau * u)

    print(f"\nIterating...", flush=True)
    t0 = time.time()
    mu_final, hist, conv = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, args.tau, args.gamma,
        damping=args.damping, anderson=args.anderson,
        anderson_beta=args.anderson_beta, max_iter=args.max_iter,
        tol=args.tol, progress=args.progress,
    )
    elapsed = time.time() - t0
    print(f"\nFinished: iters={len(hist)}, residual={hist[-1][0]:.4e}, "
          f"active={hist[-1][1]}, converged={conv}, elapsed={elapsed:.1f}s", flush=True)

    # Measure 1-R²
    print(f"\nMeasuring 1-R² over Gu^3 = {args.Gu**3} realizations...", flush=True)
    t1 = time.time()
    R2def, slope, n_samples = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi,
                                         args.tau, args.gamma)
    print(f"  1-R² = {R2def:.6e}", flush=True)
    print(f"  slope = {slope:.4f}", flush=True)
    print(f"  n_samples = {n_samples}", flush=True)
    print(f"  ({time.time() - t1:.1f}s)", flush=True)

    # Save
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    label = f"posterior_v3_Gu{args.Gu}_Gp{args.Gp}_tau{args.tau:g}_gamma{args.gamma:g}_{args.label}"
    np.savez(outdir / f"{label}_prices.npz",
             mu=mu_final, u_grid=u_grid, p_grid=p_grid,
             p_lo=p_lo, p_hi=p_hi)
    summary = {
        "method": "posterior_function_v3",
        "Gu": args.Gu, "Gp": args.Gp,
        "tau": args.tau, "gamma": args.gamma,
        "iterations": len(hist), "converged": conv,
        "residual_inf": hist[-1][0] if hist else None,
        "active_cells": hist[-1][1] if hist else None,
        "revelation_deficit": float(R2def), "slope": float(slope),
        "n_samples": int(n_samples),
        "elapsed_seconds": elapsed,
    }
    with open(outdir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {outdir}/{label}_*", flush=True)


if __name__ == "__main__":
    main()
