#!/usr/bin/env python3
"""Heterogeneous-gamma solver with KERNEL-SMOOTHED contour evidence.

Replaces the linear-interp sign-change tracer in
full_ree_solver_het.py with a Gaussian-kernel-weighted sum:

  A_v = sum_{i,j} K((P[i,j] - p_target)/h) * f_v(u_i) * f_v(u_j)

  K(x) = exp(-x^2/2)

This is smooth in P (and in p_target), so the residual surface has no
sign-change kinks.  Fixed points can therefore be reached at machine
precision via Picard+Anderson or Newton-Krylov, even without
permutation symmetrization in the heterogeneous-gamma case.

The bandwidth h biases the result by O(h^2) (since the Gaussian
approximates a delta).  Recommend h ~ 0.005-0.02 for G=6.
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


def _f_density(u, v, tau):
    mean = v - 0.5
    return math.sqrt(tau / (2.0 * math.pi)) * np.exp(-0.5 * tau * (np.asarray(u) - mean) ** 2)


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


# ---------- KERNEL-SMOOTHED contour evidence ----------

def contour_evidence_smooth(slice_2d, grid, target_p, tau, h):
    """A_0, A_1 = sum over (a,b) cells of exp(-((P[a,b]-p)/h)^2/2) * f_v(a)*f_v(b).
    Smooth in P; same as Dirac (and original linear-interp tracer up to bias)
    in the limit h -> 0.
    """
    n = len(grid)
    f0_grid = _f_density(grid, 0, tau)
    f1_grid = _f_density(grid, 1, tau)
    # outer products of f_v on (a,b)
    F0 = np.outer(f0_grid, f0_grid)
    F1 = np.outer(f1_grid, f1_grid)
    z = (slice_2d - target_p) / h
    K = np.exp(-0.5 * z * z)
    A0 = float(np.sum(K * F0))
    A1 = float(np.sum(K * F1))
    return A0, A1


def _symmetrize(P):
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    return sum(np.transpose(P, axes=pp) for pp in perms) / len(perms)


def phi_het(P, grid, tau, gammas, h, symmetric=False):
    """One Φ evaluation with kernel-smoothed contour evidence."""
    G = len(grid)
    P_new = np.empty_like(P)
    M1 = np.empty_like(P); M2 = np.empty_like(P); M3 = np.empty_like(P)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = float(P[i, j, k])
                u1 = float(grid[i]); u2 = float(grid[j]); u3 = float(grid[k])
                a0_1, a1_1 = contour_evidence_smooth(P[i, :, :], grid, p, tau, h)
                a0_2, a1_2 = contour_evidence_smooth(P[:, j, :], grid, p, tau, h)
                a0_3, a1_3 = contour_evidence_smooth(P[:, :, k], grid, p, tau, h)
                def post(uo, A0, A1):
                    f0 = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (uo + 0.5) ** 2)
                    f1 = math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (uo - 0.5) ** 2)
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


def deficit(P, grid, tau):
    G = len(grid)
    Y, X, W = [], [], []
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                if not (1e-9 < p < 1 - 1e-9): continue
                T = tau * (grid[i] + grid[j] + grid[k])
                f1 = _f_density(np.array([grid[i], grid[j], grid[k]]), 1, tau)
                f0 = _f_density(np.array([grid[i], grid[j], grid[k]]), 0, tau)
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


def picard_anderson_het(P0, grid, tau, gammas, h, damping=0.3, anderson=5,
                        anderson_beta=1.0, max_iter=600, tol=1e-15,
                        progress=False, symmetric=False):
    P = P0.copy(); history = []
    x_hist, f_hist = [], []
    for it in range(1, max_iter + 1):
        cand, _ = phi_het(P, grid, tau, gammas, h, symmetric=symmetric)
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


def _gmres(matvec, b, max_iter=80, tol=1e-3):
    n = b.size
    beta = float(np.linalg.norm(b))
    if beta == 0:
        return np.zeros_like(b), 0.0, 0
    V = np.zeros((n, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))
    V[:, 0] = b / beta
    g = np.zeros(max_iter + 1); g[0] = beta
    cs = np.zeros(max_iter); sn = np.zeros(max_iter)
    best_y = np.zeros(0); best_resid = beta; best_iter = 0
    for k in range(max_iter):
        w = matvec(V[:, k])
        for j in range(k + 1):
            H[j, k] = float(V[:, j] @ w)
            w = w - H[j, k] * V[:, j]
        H[k+1, k] = float(np.linalg.norm(w))
        if H[k+1, k] < 1e-30:
            best_iter = k + 1
            best_y = np.linalg.lstsq(H[:k+1, :k+1], g[:k+1], rcond=None)[0]
            best_resid = 0.0
            break
        V[:, k+1] = w / H[k+1, k]
        for i in range(k):
            t1 = cs[i]*H[i,k] + sn[i]*H[i+1,k]
            t2 = -sn[i]*H[i,k] + cs[i]*H[i+1,k]
            H[i,k] = t1; H[i+1,k] = t2
        denom = math.hypot(H[k,k], H[k+1,k])
        cs[k] = H[k,k]/denom; sn[k] = H[k+1,k]/denom
        H[k,k] = denom; H[k+1,k] = 0.0
        t1 = cs[k]*g[k]; g[k+1] = -sn[k]*g[k]; g[k] = t1
        resid = abs(g[k+1])
        y = np.linalg.solve(H[:k+1, :k+1], g[:k+1])
        if resid < best_resid:
            best_y = y; best_resid = resid; best_iter = k + 1
        if resid <= tol * beta: break
    step = V[:, :best_iter] @ best_y
    return step, best_resid, best_iter


def newton_krylov_het(P0, grid, tau, gammas, h, max_iter=20, tol=1e-15,
                     gmres_max_iter=80, gmres_tol=1e-3, fd_eps=1e-7,
                     newton_damping=1.0, min_step=1e-4, progress=False,
                     symmetric=False):
    P = P0.copy(); history = []
    for it in range(1, max_iter + 1):
        cand, _ = phi_het(P, grid, tau, gammas, h, symmetric=symmetric)
        F = P - cand
        residual = float(np.max(np.abs(F)))
        history.append(residual)
        if residual < tol:
            return P, history, True
        f_flat = F.ravel(); p_flat = P.ravel()
        jvp_scale = fd_eps * max(1.0, float(np.linalg.norm(p_flat)))
        def matvec(v):
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0: return np.zeros_like(v)
            eps = jvp_scale / v_norm
            P_eps = np.clip((p_flat + eps * v).reshape(P.shape), 1e-8, 1 - 1e-8)
            cand_e, _ = phi_het(P_eps, grid, tau, gammas, h, symmetric=symmetric)
            F_e = P_eps - cand_e
            return (F_e.ravel() - f_flat) / eps
        step, gres, git = _gmres(matvec, -f_flat, max_iter=gmres_max_iter, tol=gmres_tol)
        alpha = min(1.0, newton_damping)
        accepted = False
        best_P = P; best_residual = residual
        while alpha >= min_step:
            trial = np.clip((p_flat + alpha * step).reshape(P.shape), 1e-8, 1 - 1e-8)
            ct, _ = phi_het(trial, grid, tau, gammas, h, symmetric=symmetric)
            tr = float(np.max(np.abs(trial - ct)))
            if tr < best_residual:
                best_P = trial; best_residual = tr; accepted = True
                if tr <= (1 - 1e-4 * alpha) * residual: break
            alpha *= 0.5
        P = best_P
        if progress:
            print(f"  newton={it} resid={residual:.4e} -> {best_residual:.4e}  "
                  f"gmres({git})={gres:.2e}  alpha={alpha:.3e}", flush=True)
        if not accepted:
            return P, history, False
    return P, history, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--umax", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gammas", type=str, required=True)
    ap.add_argument("--seed-array", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--h", type=float, required=True, help="Gaussian kernel bandwidth")
    ap.add_argument("--method", choices=["picard", "newton"], default="picard")
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-14)
    ap.add_argument("--damping", type=float, default=0.3)
    ap.add_argument("--anderson", type=int, default=5)
    ap.add_argument("--anderson-beta", type=float, default=0.7)
    ap.add_argument("--gmres-max-iter", type=int, default=80)
    ap.add_argument("--gmres-tol", type=float, default=1e-3)
    ap.add_argument("--fd-eps", type=float, default=1e-7)
    ap.add_argument("--newton-damping", type=float, default=1.0)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--outdir", type=str, default="results/full_ree")
    args = ap.parse_args()

    gammas = [float(x) for x in args.gammas.split(",")]
    if len(gammas) != 3:
        raise SystemExit("--gammas must be 3 values")

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
    if args.method == "picard":
        P_final, hist, conv = picard_anderson_het(
            P0, grid, args.tau, gammas, args.h,
            damping=args.damping, anderson=args.anderson,
            anderson_beta=args.anderson_beta, max_iter=args.max_iter,
            tol=args.tol, progress=args.progress, symmetric=args.symmetric)
    else:
        P_final, hist, conv = newton_krylov_het(
            P0, grid, args.tau, gammas, args.h,
            max_iter=args.max_iter, tol=args.tol,
            gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
            fd_eps=args.fd_eps, newton_damping=args.newton_damping,
            progress=args.progress, symmetric=args.symmetric)
    elapsed = time.time() - t0

    R2def, slope = deficit(P_final, grid, args.tau)
    i_p1 = int(np.argmin(np.abs(grid - 1.0)))
    i_m1 = int(np.argmin(np.abs(grid + 1.0)))
    T_can = args.tau * (grid[i_p1] + grid[i_m1] + grid[i_p1])
    fr_can = float(logistic(T_can))
    _, posts = phi_het(P_final, grid, args.tau, gammas, args.h, symmetric=args.symmetric)
    M1, M2, M3 = posts
    P_fr = logistic(args.tau * (grid[:, None, None] + grid[None, :, None] + grid[None, None, :]))
    max_fr = float(np.max(np.abs(P_final - P_fr)))

    summary = {
        "G": args.G, "umax": args.umax, "tau": args.tau,
        "gammas": gammas, "h": args.h, "method": args.method,
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
    np.savez(outdir / f"G{args.G}_tau{args.tau:g}_smoothh{args.h:g}_het{g_str}_{args.label}_prices.npz",
             P=P_final, grid=grid)
    with open(outdir / f"G{args.G}_tau{args.tau:g}_smoothh{args.h:g}_het{g_str}_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
