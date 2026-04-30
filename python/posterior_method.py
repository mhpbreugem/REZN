#!/usr/bin/env python3
"""Posterior-function method per POSTERIOR_METHOD.md.

Stores μ(u, p) on a (G_u × G_p) grid. The contour falls out from market
clearing using the current μ — no price interpolation, no boundary issues.
Works for symmetric agents (single μ function).

Usage:
  python3 python/posterior_method.py --Gu 20 --Gp 20 --Nsweep 20 \
      --tau 2 --gamma 0.5 --max-iter 100 --tol 1e-10 --progress
"""

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np

EPS = 1e-10


def Lam(z):
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


def f_v(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


def crra_demand(mu, p, gamma, W=1.0):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)


def interp_mu_u(mu_col, u_grid, u_query):
    """1D interpolation of μ(·, p_j) at u_query.

    mu_col: shape (G_u,) — values at u_grid for fixed p_j.
    Linear interpolation; clipped to (EPS, 1-EPS).
    """
    val = np.interp(u_query, u_grid, mu_col)
    return float(np.clip(val, EPS, 1 - EPS))


def root_u3(mu_col, u_grid, p_j, gamma, T3,
            u_lo=None, u_hi=None, tol=1e-12, max_iter=80):
    """Bisect for u₃ in [u_lo, u_hi] such that
    x(μ(u₃, p_j), p_j; γ) = T3.

    The demand x is monotone-increasing in μ, μ monotone-increasing in u
    (for sane mu_col). So x is monotone-increasing in u, and bisection works.
    """
    if u_lo is None: u_lo = u_grid[0]
    if u_hi is None: u_hi = u_grid[-1]

    def f(u):
        mu = interp_mu_u(mu_col, u_grid, u)
        return crra_demand(mu, p_j, gamma) - T3

    flo = f(u_lo)
    fhi = f(u_hi)
    # If T3 is outside the bracket, return the closest endpoint
    if flo > 0:
        return u_lo
    if fhi < 0:
        return u_hi
    # Bisect
    for _ in range(max_iter):
        m = 0.5 * (u_lo + u_hi)
        fm = f(m)
        if fm < 0:
            u_lo = m
        else:
            u_hi = m
        if u_hi - u_lo < tol:
            break
    return 0.5 * (u_lo + u_hi)


def phi_mu(mu_grid, u_grid, p_grid, tau, gamma, N_sweep, two_pass=True):
    """One Φ_μ evaluation. Returns the new μ array (G_u × G_p).

    For each (u_i, p_j):
      D₁ = -x(μ(u_i, p_j), p_j)
      Sweep u₂ in N_sweep points; root-find u₃; sum f_v(u₂) f_v(u₃) → A_v
      μ_new(u_i, p_j) = f1(u_i) A_1 / (f0(u_i) A_0 + f1(u_i) A_1)

    If two_pass=True, also sweep u₃ symmetrically and average.
    """
    G_u = len(u_grid); G_p = len(p_grid)
    mu_new = np.empty_like(mu_grid)

    # Pre-compute signal densities on the sweep grid
    u_sweep = np.linspace(u_grid[0], u_grid[-1], N_sweep)
    f0_sweep = f_v(u_sweep, 0, tau)
    f1_sweep = f_v(u_sweep, 1, tau)

    for j, p_j in enumerate(p_grid):
        mu_col = mu_grid[:, j]
        for i, u_i in enumerate(u_grid):
            mu_i = mu_col[i]
            x1 = crra_demand(mu_i, p_j, gamma)
            D1 = -x1

            A0 = 0.0; A1 = 0.0

            # Pass A: sweep u₂, root-find u₃
            for n, u2 in enumerate(u_sweep):
                mu_2 = interp_mu_u(mu_col, u_grid, u2)
                x2 = crra_demand(mu_2, p_j, gamma)
                T3 = D1 - x2
                u3 = root_u3(mu_col, u_grid, p_j, gamma, T3)
                f0_u3 = float(f_v(np.array([u3]), 0, tau)[0])
                f1_u3 = float(f_v(np.array([u3]), 1, tau)[0])
                A0 += f0_sweep[n] * f0_u3
                A1 += f1_sweep[n] * f1_u3

            if two_pass:
                # Pass B: sweep u₃, root-find u₂ (by symmetry of agents 2 & 3)
                for n, u3 in enumerate(u_sweep):
                    mu_3 = interp_mu_u(mu_col, u_grid, u3)
                    x3 = crra_demand(mu_3, p_j, gamma)
                    T2 = D1 - x3
                    u2 = root_u3(mu_col, u_grid, p_j, gamma, T2)
                    f0_u2 = float(f_v(np.array([u2]), 0, tau)[0])
                    f1_u2 = float(f_v(np.array([u2]), 1, tau)[0])
                    A0 += f0_u2 * f0_sweep[n]
                    A1 += f1_u2 * f1_sweep[n]
                A0 *= 0.5; A1 *= 0.5

            f0_i = float(f_v(np.array([u_i]), 0, tau)[0])
            f1_i = float(f_v(np.array([u_i]), 1, tau)[0])
            denom = f0_i * A0 + f1_i * A1
            if denom <= 0:
                mu_new[i, j] = 0.5
            else:
                mu_new[i, j] = float(np.clip(f1_i * A1 / denom, EPS, 1 - EPS))

    return mu_new


def picard_anderson_mu(mu0, u_grid, p_grid, tau, gamma, N_sweep,
                       damping=0.5, anderson=5, anderson_beta=0.7,
                       max_iter=200, tol=1e-12, progress=False):
    mu = mu0.copy(); history = []
    x_hist = []; f_hist = []
    for it in range(1, max_iter + 1):
        cand = phi_mu(mu, u_grid, p_grid, tau, gamma, N_sweep)
        F = cand - mu
        residual = float(np.max(np.abs(F)))
        history.append(residual)
        if progress and (it % 5 == 0 or it == 1):
            print(f"  iter={it} resid={residual:.4e}", flush=True)
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


def measure_R2(mu, u_grid, p_grid, tau, gamma, N_sample=50, K=3):
    """Sample (u_1, u_2, u_3) realizations, solve market clearing for p,
    measure 1-R² of logit(p) vs T*."""
    rng = np.random.default_rng(0)
    Y = []; X = []; Wts = []

    # use tensor product of u_grid for samples (or random)
    G_u = len(u_grid)
    # Sample on the grid for reproducibility
    triples = [(u_grid[i], u_grid[j], u_grid[k])
               for i in range(G_u) for j in range(G_u) for k in range(G_u)]

    for u1, u2, u3 in triples:
        # Market clear: find p such that sum_k x(μ(u_k, p), p) = 0
        def Z(p):
            mu1 = interp_mu_2d(mu, u_grid, p_grid, u1, p)
            mu2 = interp_mu_2d(mu, u_grid, p_grid, u2, p)
            mu3 = interp_mu_2d(mu, u_grid, p_grid, u3, p)
            return sum(crra_demand(mu_k, p, gamma) for mu_k in (mu1, mu2, mu3))

        lo, hi = 1e-6, 1 - 1e-6
        flo = Z(lo); fhi = Z(hi)
        if flo * fhi >= 0:
            continue
        # Bisect
        for _ in range(80):
            m = 0.5 * (lo + hi)
            if Z(m) > 0:
                lo = m
            else:
                hi = m
        p_star = 0.5 * (lo + hi)
        T = tau * (u1 + u2 + u3)
        w = 0.5 * (f_v(u1, 1, tau) * f_v(u2, 1, tau) * f_v(u3, 1, tau)
                   + f_v(u1, 0, tau) * f_v(u2, 0, tau) * f_v(u3, 0, tau))
        Y.append(logit(p_star)); X.append(T); Wts.append(float(w))

    Y, X, Wts = np.array(Y), np.array(X), np.array(Wts)
    Wts = Wts / Wts.sum()
    Yb = (Wts * Y).sum(); Xb = (Wts * X).sum()
    cov = (Wts * (Y - Yb) * (X - Xb)).sum()
    vy = (Wts * (Y - Yb) ** 2).sum()
    vx = (Wts * (X - Xb) ** 2).sum()
    R2 = cov ** 2 / (vy * vx) if vy * vx > 0 else 0.0
    slope = cov / vx if vx > 0 else 0.0
    return 1.0 - R2, slope, len(Y)


def interp_mu_2d(mu, u_grid, p_grid, u_q, p_q):
    """Bilinear interpolation of mu at (u_q, p_q)."""
    u_q = float(np.clip(u_q, u_grid[0], u_grid[-1]))
    p_q = float(np.clip(p_q, p_grid[0], p_grid[-1]))
    i = int(np.clip(np.searchsorted(u_grid, u_q) - 1, 0, len(u_grid) - 2))
    j = int(np.clip(np.searchsorted(p_grid, p_q) - 1, 0, len(p_grid) - 2))
    du = u_grid[i + 1] - u_grid[i]; dp = p_grid[j + 1] - p_grid[j]
    fu = (u_q - u_grid[i]) / du if du > 0 else 0.0
    fp = (p_q - p_grid[j]) / dp if dp > 0 else 0.0
    v00 = mu[i, j]; v10 = mu[i + 1, j]; v01 = mu[i, j + 1]; v11 = mu[i + 1, j + 1]
    val = (1 - fu) * (1 - fp) * v00 + fu * (1 - fp) * v10 \
        + (1 - fu) * fp * v01 + fu * fp * v11
    return float(np.clip(val, EPS, 1 - EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Gu", type=int, default=20)
    ap.add_argument("--Gp", type=int, default=20)
    ap.add_argument("--Nsweep", type=int, default=20)
    ap.add_argument("--umax", type=float, default=4.0)
    ap.add_argument("--Llogit", type=float, default=6.0,
                    help="p-grid spans Lambda(-L) to Lambda(+L) on a logit grid")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--anderson", type=int, default=5)
    ap.add_argument("--anderson-beta", type=float, default=0.7)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--label", type=str, default="post")
    ap.add_argument("--outdir", type=str, default="results/full_ree")
    args = ap.parse_args()

    u_grid = np.linspace(-args.umax, args.umax, args.Gu)
    # logit-spaced p grid
    logit_p = np.linspace(-args.Llogit, args.Llogit, args.Gp)
    p_grid = Lam(logit_p)

    # Initialize: μ⁰(u, p) = Λ(τu)  (no-learning)
    mu0 = np.zeros((args.Gu, args.Gp))
    for i, u in enumerate(u_grid):
        mu0[i, :] = float(Lam(np.array([args.tau * u]))[0])

    print(f"Posterior-method solver:", flush=True)
    print(f"  Gu={args.Gu}, Gp={args.Gp}, Nsweep={args.Nsweep}", flush=True)
    print(f"  u_grid: [{u_grid[0]:.2f}, {u_grid[-1]:.2f}]", flush=True)
    print(f"  p_grid: [{p_grid[0]:.4f}, {p_grid[-1]:.4f}] (logit-spaced)", flush=True)
    print(f"  τ={args.tau}, γ={args.gamma}", flush=True)
    print(f"  damping={args.damping}, anderson={args.anderson}", flush=True)
    print(f"  max_iter={args.max_iter}, tol={args.tol:.0e}", flush=True)

    t0 = time.time()
    mu_final, hist, conv = picard_anderson_mu(
        mu0, u_grid, p_grid, args.tau, args.gamma, args.Nsweep,
        damping=args.damping, anderson=args.anderson, anderson_beta=args.anderson_beta,
        max_iter=args.max_iter, tol=args.tol, progress=args.progress,
    )
    elapsed = time.time() - t0

    print(f"\nFinished: iters={len(hist)}, residual={hist[-1]:.4e}, "
          f"converged={conv}, elapsed={elapsed:.1f}s", flush=True)

    # Measure 1-R² by sampling u₁,u₂,u₃ on grid
    print(f"\nMeasuring 1-R² by sampling Gu^3 = {args.Gu**3} realizations...", flush=True)
    t1 = time.time()
    R2def, slope, n_samples = measure_R2(mu_final, u_grid, p_grid,
                                          args.tau, args.gamma)
    elapsed_R2 = time.time() - t1
    print(f"  1-R² = {R2def:.6e}", flush=True)
    print(f"  slope = {slope:.4f}", flush=True)
    print(f"  n_samples = {n_samples}", flush=True)
    print(f"  ({elapsed_R2:.1f}s)", flush=True)

    # Save
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    label = f"posterior_Gu{args.Gu}_Gp{args.Gp}_Nsw{args.Nsweep}_tau{args.tau:g}_gamma{args.gamma:g}_{args.label}"
    np.savez(outdir / f"{label}_prices.npz",
             mu=mu_final, u_grid=u_grid, p_grid=p_grid)
    summary = {
        "method": "posterior_function",
        "Gu": args.Gu, "Gp": args.Gp, "Nsweep": args.Nsweep,
        "tau": args.tau, "gamma": args.gamma,
        "iterations": len(hist),
        "converged": conv,
        "residual_inf": hist[-1] if hist else None,
        "revelation_deficit": float(R2def),
        "slope": float(slope),
        "n_samples": int(n_samples),
        "elapsed_seconds": elapsed,
        "elapsed_R2_seconds": elapsed_R2,
    }
    with open(outdir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {outdir}/{label}_*", flush=True)


if __name__ == "__main__":
    main()
