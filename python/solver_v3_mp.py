#!/usr/bin/env python3
"""
Posterior-method v3 REE solver.

State variable: μ*(u_own, p) — posterior belief function.
Fixed-point map Φ: for each (u_i, p_j), integrate the signal density along
the price contour {(u_j,u_l): P(u_i,u_j,u_l)=p_j} → new posterior.

Usage:
    cd /home/user/REZN
    python python/solver_v3_mp.py \
        --gamma 4.0 --tau 3.0 \
        --out results/full_ree/task3_g400_t0030.json \
        [--seed results/full_ree/task3_g400_t0020.json]  # optional warm-start
"""

import json, time, math, sys, argparse
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit as Lam

# ─────────────────────────────────────────────────────────────────────────────
# Core math helpers
# ─────────────────────────────────────────────────────────────────────────────

def sig_f(u, v, tau):
    """Signal density f_v(u)."""
    mean = v - 0.5
    return math.sqrt(tau / (2 * math.pi)) * math.exp(-tau / 2 * (u - mean) ** 2)

def logit_f(p):
    return math.log(p / (1.0 - p))

def crra_d(mu, p, gamma):
    """CRRA excess demand for one agent."""
    if mu < 1e-13 or mu > 1-1e-13 or p < 1e-13 or p > 1-1e-13:
        return 0.0
    z = logit_f(mu) - logit_f(p)
    R = math.exp(z / gamma)
    return (R - 1.0) / ((1.0 - p) + R * p)

# ─────────────────────────────────────────────────────────────────────────────
# μ* interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interp_mu(u_val, p_val, u_grid, p_grids, mu_arr):
    G = len(u_grid)
    if u_val <= u_grid[0]:    i_lo, i_hi = 0, 1
    elif u_val >= u_grid[-1]: i_lo, i_hi = G-2, G-1
    else:
        i_lo = int(np.searchsorted(u_grid, u_val)) - 1
        i_hi = i_lo + 1

    def at(idx):
        p_a = p_grids[idx]; m_a = mu_arr[idx]
        if p_val <= p_a[0]:  return m_a[0]
        if p_val >= p_a[-1]: return m_a[-1]
        j = max(0, min(int(np.searchsorted(p_a, p_val)) - 1, len(p_a)-2))
        f = (p_val - p_a[j]) / (p_a[j+1] - p_a[j] + 1e-30)
        return m_a[j] + f * (m_a[j+1] - m_a[j])

    mu_lo, mu_hi = at(i_lo), at(i_hi)
    fu = (u_val - u_grid[i_lo]) / (u_grid[i_hi] - u_grid[i_lo] + 1e-30)
    return float(np.clip(mu_lo + fu * (mu_hi - mu_lo), 1e-13, 1-1e-13))

# ─────────────────────────────────────────────────────────────────────────────
# Market clearing
# ─────────────────────────────────────────────────────────────────────────────

def mc_excess(p, u1, u2, u3, gamma, u_grid, p_grids, mu_arr):
    m1 = interp_mu(u1, p, u_grid, p_grids, mu_arr)
    m2 = interp_mu(u2, p, u_grid, p_grids, mu_arr)
    m3 = interp_mu(u3, p, u_grid, p_grids, mu_arr)
    return crra_d(m1,p,gamma) + crra_d(m2,p,gamma) + crra_d(m3,p,gamma)

def solve_mc(u1, u2, u3, gamma, u_grid, p_grids, mu_arr):
    try:
        return brentq(lambda p: mc_excess(p, u1,u2,u3, gamma, u_grid, p_grids, mu_arr),
                      1e-6, 1-1e-6, xtol=1e-11, maxiter=300)
    except Exception:
        return float('nan')

# ─────────────────────────────────────────────────────────────────────────────
# Contour integration
# ─────────────────────────────────────────────────────────────────────────────

def _crossings(arr, u_grid, pt):
    out = []
    for k in range(len(u_grid) - 1):
        a, b = arr[k], arr[k+1]
        if math.isnan(a) or math.isnan(b): continue
        if (a - pt) * (b - pt) <= 0 and abs(b - a) > 1e-16:
            f = (pt - a) / (b - a)
            out.append(u_grid[k] + f * (u_grid[k+1] - u_grid[k]))
    return out

def contour_posterior(price_slice, u_grid, p_target, u_own, tau):
    """
    Compute posterior given own signal u_own and price p_target.
    price_slice[j,l] = P(u_own, u_j, u_l).
    Returns (A0, A1) accumulated signal densities along contour.
    """
    G = len(u_grid)
    f0_own = sig_f(u_own, 0, tau)
    f1_own = sig_f(u_own, 1, tau)
    A0, A1, n = 0.0, 0.0, 0

    # Pass A: sweep u_j rows, find u_l crossings
    for j in range(G):
        for u_l in _crossings(price_slice[j, :], u_grid, p_target):
            u_j = u_grid[j]
            w0 = sig_f(u_j, 0, tau) * sig_f(u_l, 0, tau)
            w1 = sig_f(u_j, 1, tau) * sig_f(u_l, 1, tau)
            A0 += w0; A1 += w1; n += 1

    # Pass B: sweep u_l cols, find u_j crossings
    for l in range(G):
        for u_j in _crossings(price_slice[:, l], u_grid, p_target):
            u_l = u_grid[l]
            w0 = sig_f(u_j, 0, tau) * sig_f(u_l, 0, tau)
            w1 = sig_f(u_j, 1, tau) * sig_f(u_l, 1, tau)
            A0 += w0; A1 += w1; n += 1

    if n == 0:
        return float(Lam(tau * u_own))   # fallback: no-learning
    A0 /= n; A1 /= n
    num = f1_own * A1
    den = f0_own * A0 + f1_own * A1
    if den < 1e-30:
        return float(Lam(tau * u_own))
    return float(np.clip(num / den, 1e-12, 1-1e-12))

# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity (PAVA)
# ─────────────────────────────────────────────────────────────────────────────

def pava_inc(y):
    y = list(y)
    n = len(y)
    # pool-adjacent-violators
    blocks = [[i] for i in range(n)]
    vals   = list(y)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i+1]:
                merged = blocks[i] + blocks[i+1]
                avg = sum(y[k] for k in merged) / len(merged)
                blocks[i] = merged; vals[i] = avg
                blocks.pop(i+1); vals.pop(i+1)
                changed = True
            else:
                i += 1
    res = [0.0] * n
    for blk, v in zip(blocks, vals):
        for k in blk:
            res[k] = v
    return res

def enforce_mono(mu_arr, G):
    # monotone in p (mu_arr[i][j] increases with j)
    for i in range(G):
        mu_arr[i] = pava_inc(mu_arr[i])
    # monotone in u (mu_arr[i][j] increases with i) for each j
    Gp = len(mu_arr[0])
    for j in range(Gp):
        col = [mu_arr[i][j] for i in range(G)]
        col = pava_inc(col)
        for i in range(G):
            mu_arr[i][j] = col[i]
    return mu_arr

# ─────────────────────────────────────────────────────────────────────────────
# p-grid construction
# ─────────────────────────────────────────────────────────────────────────────

def build_p_grids(u_grid, gamma, tau, Gp):
    """Logit-spaced p-grid per u_own, range from no-learning price bounds."""
    G = len(u_grid)
    u_min, u_max = u_grid[0], u_grid[-1]
    p_grids = []
    for i in range(G):
        u_own = u_grid[i]
        def exc_lo(p):
            return sum(crra_d(float(Lam(tau*u)), p, gamma) for u in [u_own, u_min, u_min])
        def exc_hi(p):
            return sum(crra_d(float(Lam(tau*u)), p, gamma) for u in [u_own, u_max, u_max])
        try: p_lo = brentq(exc_lo, 1e-7, 1-1e-7, xtol=1e-9)
        except: p_lo = 1e-4
        try: p_hi = brentq(exc_hi, 1e-7, 1-1e-7, xtol=1e-9)
        except: p_hi = 1-1e-4
        p_lo = max(p_lo, 1e-5); p_hi = min(p_hi, 1-1e-5)
        # logit-spaced
        lo_l, hi_l = logit_f(p_lo), logit_f(p_hi)
        row = [float(Lam(x)) for x in np.linspace(lo_l, hi_l, Gp)]
        p_grids.append(row)
    return p_grids

# ─────────────────────────────────────────────────────────────────────────────
# One Φ step
# ─────────────────────────────────────────────────────────────────────────────

def phi_step(u_grid, p_grids, mu_arr, gamma, tau):
    G = len(u_grid)
    mu_new = [[0.0]*len(p_grids[i]) for i in range(G)]
    residuals = []

    for i in range(G):
        u_own = u_grid[i]
        # Build price surface P(u_own, u_j, u_l)
        ps = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                ps[j, l] = solve_mc(u_own, u_grid[j], u_grid[l], gamma, u_grid, p_grids, mu_arr)

        for k, p_tgt in enumerate(p_grids[i]):
            mu_val = contour_posterior(ps, u_grid, p_tgt, u_own, tau)
            mu_new[i][k] = mu_val
            residuals.append(abs(mu_val - mu_arr[i][k]))

    mu_new = enforce_mono(mu_new, G)
    F_max = max(residuals)
    F_med = float(np.median(residuals))
    return mu_new, F_max, F_med

# ─────────────────────────────────────────────────────────────────────────────
# Anderson acceleration (m-step history)
# ─────────────────────────────────────────────────────────────────────────────

def flatten(mu_arr):
    return np.array([v for row in mu_arr for v in row])

def unflatten(vec, mu_arr):
    out = []
    k = 0
    for row in mu_arr:
        n = len(row)
        out.append(list(np.clip(vec[k:k+n], 1e-12, 1-1e-12)))
        k += n
    return out

def anderson_update(x_hist, g_hist, x_new, g_new, m):
    """Type-II Anderson: returns improved x from last m+1 (x,g=Φ(x)) pairs."""
    x_hist.append(x_new.copy()); g_hist.append(g_new.copy())
    if len(x_hist) > m + 1:
        x_hist.pop(0); g_hist.pop(0)
    k = len(x_hist)
    if k < 2:
        return g_new  # first step: plain Picard

    # f = g - x (residuals)
    F = np.array([g_hist[i] - x_hist[i] for i in range(k)])  # k × d
    # Solve min ||Σ θ_i f_i||² s.t. Σ θ_i = 1
    dF = np.diff(F, axis=0)  # (k-1) × d
    try:
        # least-squares: min ||dF @ c||² s.t. Σ c = 1 — unconstrained form
        A = dF @ dF.T
        b = np.ones(k-1)
        # solve for c with minimum norm
        c, _, _, _ = np.linalg.lstsq(A, dF @ (g_hist[-1] - x_hist[-1]), rcond=None)
        x_aa = g_hist[-1] - dF.T @ c
    except Exception:
        x_aa = g_new
    return np.clip(x_aa, 1e-12, 1-1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Weighted 1-R²
# ─────────────────────────────────────────────────────────────────────────────

def compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau):
    G = len(u_grid)
    f0g = np.array([sig_f(u, 0, tau) for u in u_grid])
    f1g = np.array([sig_f(u, 1, tau) for u in u_grid])
    Ts_all, lp_all, w_all = [], [], []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u1,u2,u3 = u_grid[i],u_grid[j],u_grid[l]
                p = solve_mc(u1,u2,u3,gamma,u_grid,p_grids,mu_arr)
                if math.isnan(p) or p < 1e-6 or p > 1-1e-6: continue
                w = 0.5*(f0g[i]*f0g[j]*f0g[l]+f1g[i]*f1g[j]*f1g[l])
                Ts_all.append(tau*(u1+u2+u3))
                lp_all.append(logit_f(p))
                w_all.append(w)
    if len(Ts_all) < 10: return float('nan'), float('nan')
    T = np.array(Ts_all); lp = np.array(lp_all); w = np.array(w_all)
    w /= w.sum()
    sl, ic = np.polyfit(T, lp, 1, w=np.sqrt(w*len(w)))
    pred = sl*T+ic
    mean_lp = np.average(lp, weights=w)
    var_t = np.average((lp-mean_lp)**2, weights=w)
    var_r = np.average((lp-pred)**2, weights=w)
    return (var_r/var_t if var_t>1e-30 else 0.0), sl

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_ckpt(path):
    with open(path) as f: d = json.load(f)
    u_grid  = np.array([float(x) for x in d['u_grid']])
    p_grids = [[float(x) for x in row] for row in d['p_grid']]
    mu_arr  = [[float(x) for x in row] for row in d['mu_strings']]
    return d['G'], float(d['tau']), float(d['gamma']), u_grid, p_grids, mu_arr, d

def save_ckpt(path, G, tau, gamma, u_grid, p_grids, mu_arr, F_max, F_med, elapsed, hist):
    d = dict(G=G, tau=tau, gamma=gamma, dps=50,
             F_max=f'{F_max:.6e}', F_med=f'{F_med:.6e}',
             elapsed_s=round(elapsed,1), history=hist,
             u_grid=[str(x) for x in u_grid],
             p_grid=[[str(x) for x in row] for row in p_grids],
             mu_strings=[[str(x) for x in row] for row in mu_arr])
    with open(path,'w') as f: json.dump(d,f)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    t0 = time.time()
    G    = args.G
    umax = args.umax
    gamma= args.gamma
    tau  = args.tau
    tol  = args.tol

    u_grid = np.linspace(-umax, umax, G)
    print(f"[{gamma=:.4g} {tau=:.4g}] G={G} umax={umax} tol={tol}", flush=True)

    # p-grids
    print("  building p-grids...", end=' ', flush=True)
    p_grids = build_p_grids(u_grid, gamma, tau, Gp=G)
    print("done", flush=True)

    # Initialise μ*
    if args.seed:
        G_s,tau_s,gamma_s,u_s,pg_s,mu_s,_ = load_ckpt(args.seed)
        print(f"  warm-start from {args.seed.split('/')[-1]} (γ={gamma_s} τ={tau_s})", flush=True)
        # Interpolate seed onto new grid
        mu_arr = []
        for i in range(G):
            row = []
            for p in p_grids[i]:
                m = interp_mu(u_grid[i], p, u_s, pg_s, mu_s)
                row.append(float(np.clip(m, 1e-6, 1-1e-6)))
            mu_arr.append(row)
    else:
        print("  init from no-learning (μ=Λ(τu), flat in p)", flush=True)
        mu_arr = [[float(Lam(tau * u_grid[i]))] * G for i in range(G)]

    mu_arr = enforce_mono(mu_arr, G)

    # Solver loop with optional Anderson
    x_hist, g_hist = [], []
    history = []
    last_save = t0
    F_max = F_med = float('nan')

    for it in range(1, args.max_iter + 1):
        mu_phi, F_max, F_med = phi_step(u_grid, p_grids, mu_arr, gamma, tau)
        elapsed = time.time() - t0

        # Anderson or Picard update
        x_flat = flatten(mu_arr)
        g_flat = flatten(mu_phi)
        if args.anderson > 0:
            x_new = anderson_update(x_hist, g_hist, x_flat, g_flat, args.anderson)
            mu_arr = enforce_mono(unflatten(x_new, mu_arr), G)
        else:
            alpha = args.alpha
            for i in range(G):
                for j in range(len(mu_arr[i])):
                    mu_arr[i][j] = alpha*mu_phi[i][j] + (1-alpha)*mu_arr[i][j]
            mu_arr = enforce_mono(mu_arr, G)

        entry = {'iter':it,'F_max':F_max,'F_med':F_med,'t':round(elapsed,1)}
        history.append(entry)
        print(f"  [{elapsed:6.0f}s] iter={it:3d}  F_max={F_max:.3e}  F_med={F_med:.3e}", flush=True)

        if time.time() - last_save > 120 or F_max < tol:
            save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
                      F_max, F_med, elapsed, history)
            last_save = time.time()

        if F_max < tol:
            print(f"\n  CONVERGED  F_max={F_max:.3e}  iter={it}", flush=True)
            break

    # Final metrics
    print("\n  computing 1-R²...", flush=True)
    one_mR2, slope = compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau)
    print(f"  1-R²={one_mR2:.6f}  slope={slope:.6f}  F_max={F_max:.3e}", flush=True)

    save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
              F_max, F_med, time.time()-t0, history)
    print(f"  saved → {args.out}", flush=True)
    return F_max, one_mR2, slope


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma',    type=float, required=True)
    ap.add_argument('--tau',      type=float, required=True)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--seed',     default=None, help='Warm-start checkpoint (omit for no-learning init)')
    ap.add_argument('--G',        type=int,   default=20)
    ap.add_argument('--umax',     type=float, default=5.0)
    ap.add_argument('--tol',      type=float, default=1e-12)
    ap.add_argument('--max_iter', type=int,   default=300)
    ap.add_argument('--alpha',    type=float, default=0.3,  help='Picard damping (if no Anderson)')
    ap.add_argument('--anderson', type=int,   default=5,    help='Anderson history depth (0=plain Picard)')
    args = ap.parse_args()
    run(args)
