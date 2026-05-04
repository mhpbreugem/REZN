#!/usr/bin/env python3
"""
Posterior-method v3 REE solver — two-phase precision.

Phase 1 (float64 + Anderson m=5):  rapid approach to F_max < 1e-10
Phase 2 (mp50   + Picard  α=0.3):  polish to F_max < tol (default 1e-25)

State: μ*(u_own, p) — posterior belief on a G×G grid.
Fixed-point map Φ: for each (u_i, p_j), integrate signal density along the
price contour {(u_j,u_l): P(u_i,u_j,u_l)=p_j} to get new posterior.

Usage:
    python python/solver_v3_mp.py \\
        --gamma 4.0 --tau 2.0 \\
        --out results/full_ree/task3_g400_t0020.json \\
        [--seed <checkpoint>]
"""

import json, time, math, sys, argparse
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit as Lam
import mpmath as mp
from mpmath import mp as mpctx

MP_DPS = 52          # 50 significant digits + 2 guard
mpctx.dps = MP_DPS
F64_TOL = 1e-10      # switch to mp50 phase when F_max < this
F64_MAX  = 1000      # max iterations for float64 phase

# ─────────────────────────────────────────────────────────────────────────────
# Float64 helpers
# ─────────────────────────────────────────────────────────────────────────────

def sig_f(u, v, tau):
    mean = v - 0.5
    return math.sqrt(tau / (2*math.pi)) * math.exp(-tau/2 * (u - mean)**2)

def logit_f(p):
    return math.log(p / (1.0 - p))

def crra_d(mu, p, gamma):
    if mu < 1e-13 or mu > 1-1e-13 or p < 1e-13 or p > 1-1e-13:
        return 0.0
    z = logit_f(mu) - logit_f(p)
    R = math.exp(z / gamma)
    return (R - 1.0) / ((1.0 - p) + R*p)

def interp_mu(u_val, p_val, u_grid, p_grids, mu_arr):
    G = len(u_grid)
    if u_val <= u_grid[0]:    i_lo, i_hi = 0, 1
    elif u_val >= u_grid[-1]: i_lo, i_hi = G-2, G-1
    else:
        i_lo = int(np.searchsorted(u_grid, u_val)) - 1
        i_hi = i_lo + 1

    def at(idx):
        p_a = p_grids[idx]; m_a = mu_arr[idx]
        if p_val <= p_a[0]:  return float(m_a[0])
        if p_val >= p_a[-1]: return float(m_a[-1])
        j = max(0, min(int(np.searchsorted(p_a, p_val)) - 1, len(p_a)-2))
        f = (p_val - p_a[j]) / (p_a[j+1] - p_a[j] + 1e-30)
        return float(m_a[j]) + f * (float(m_a[j+1]) - float(m_a[j]))

    mu_lo, mu_hi = at(i_lo), at(i_hi)
    fu = (u_val - u_grid[i_lo]) / (u_grid[i_hi] - u_grid[i_lo] + 1e-30)
    return float(np.clip(mu_lo + fu*(mu_hi - mu_lo), 1e-13, 1-1e-13))

def solve_mc(u1, u2, u3, gamma, u_grid, p_grids, mu_arr):
    def exc(p):
        return (crra_d(interp_mu(u1,p,u_grid,p_grids,mu_arr), p, gamma) +
                crra_d(interp_mu(u2,p,u_grid,p_grids,mu_arr), p, gamma) +
                crra_d(interp_mu(u3,p,u_grid,p_grids,mu_arr), p, gamma))
    try:
        return brentq(exc, 1e-6, 1-1e-6, xtol=1e-11, maxiter=300)
    except Exception:
        return float('nan')

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
    G = len(u_grid)
    f0o = sig_f(u_own, 0, tau)
    f1o = sig_f(u_own, 1, tau)
    A0, A1, n = 0.0, 0.0, 0
    for j in range(G):
        for u_l in _crossings(price_slice[j,:], u_grid, p_target):
            u_j = u_grid[j]
            A0 += sig_f(u_j,0,tau)*sig_f(u_l,0,tau)
            A1 += sig_f(u_j,1,tau)*sig_f(u_l,1,tau)
            n += 1
    for l in range(G):
        for u_j in _crossings(price_slice[:,l], u_grid, p_target):
            u_l = u_grid[l]
            A0 += sig_f(u_j,0,tau)*sig_f(u_l,0,tau)
            A1 += sig_f(u_j,1,tau)*sig_f(u_l,1,tau)
            n += 1
    if n == 0:
        return float(Lam(tau * u_own))
    A0 /= n; A1 /= n
    den = f0o*A0 + f1o*A1
    if den < 1e-30:
        return float(Lam(tau * u_own))
    return float(np.clip(f1o*A1 / den, 1e-12, 1-1e-12))

# ─────────────────────────────────────────────────────────────────────────────
# mp50 helpers
# ─────────────────────────────────────────────────────────────────────────────

_MP_EPS = mp.mpf('1e-55')
_MP1    = mp.mpf('1')

def sig_f_mp(u, v, tau):
    u = mp.mpf(str(u)); v = mp.mpf(str(v)); tau = mp.mpf(str(tau))
    mean = v - mp.mpf('0.5')
    return mp.sqrt(tau / (2*mp.pi)) * mp.exp(-tau/2 * (u - mean)**2)

def crra_d_mp(mu, p, gamma):
    eps = _MP_EPS
    if mu < eps or mu > _MP1-eps or p < eps or p > _MP1-eps:
        return mp.mpf('0')
    z = mp.log(mu/(_MP1-mu)) - mp.log(p/(_MP1-p))
    R = mp.exp(z / mp.mpf(str(gamma)))
    return (R - _MP1) / ((_MP1-p) + R*p)

def interp_mu_mp(u_val, p_val, u_grid, p_grids, mu_arr_mp):
    """Bilinear interpolation returning mpf."""
    G = len(u_grid)
    uf = float(u_val); pf = float(p_val)
    if uf <= u_grid[0]:    i_lo, i_hi = 0, 1
    elif uf >= u_grid[-1]: i_lo, i_hi = G-2, G-1
    else:
        i_lo = int(np.searchsorted(u_grid, uf)) - 1
        i_hi = i_lo + 1

    def at(idx):
        p_a = p_grids[idx]; m_a = mu_arr_mp[idx]
        if pf <= p_a[0]:  return m_a[0]
        if pf >= p_a[-1]: return m_a[-1]
        j = max(0, min(int(np.searchsorted(p_a, pf)) - 1, len(p_a)-2))
        df = mp.mpf(str(p_a[j+1] - p_a[j]))
        f  = mp.mpf(str(pf - p_a[j])) / (df if df > _MP_EPS else _MP_EPS)
        return m_a[j] + f*(m_a[j+1] - m_a[j])

    mu_lo, mu_hi = at(i_lo), at(i_hi)
    dug = mp.mpf(str(u_grid[i_hi] - u_grid[i_lo]))
    fu  = mp.mpf(str(uf - u_grid[i_lo])) / (dug if dug > _MP_EPS else _MP_EPS)
    v   = mu_lo + fu*(mu_hi - mu_lo)
    return max(_MP_EPS, min(_MP1 - _MP_EPS, v))

def solve_mc_mp(u1, u2, u3, gamma, u_grid, p_grids, mu_arr_mp, p0_hint):
    """mp50 market clearing, using float64 hint as starting point for fast convergence."""
    if math.isnan(p0_hint) or p0_hint <= 0 or p0_hint >= 1:
        return mp.nan
    gmp = mp.mpf(str(gamma))
    def exc(p):
        m1 = interp_mu_mp(u1, p, u_grid, p_grids, mu_arr_mp)
        m2 = interp_mu_mp(u2, p, u_grid, p_grids, mu_arr_mp)
        m3 = interp_mu_mp(u3, p, u_grid, p_grids, mu_arr_mp)
        return crra_d_mp(m1,p,gmp) + crra_d_mp(m2,p,gmp) + crra_d_mp(m3,p,gmp)
    try:
        # Starting from p0_hint (accurate to ~1e-10), Muller converges in ~3 steps
        return mp.findroot(exc, mp.mpf(str(p0_hint)), tol=mp.mpf('1e-50'))
    except Exception:
        try:
            eps = mp.mpf('1e-8')
            p_lo = max(mp.mpf('1e-50'), mp.mpf(str(p0_hint)) - eps)
            p_hi = min(_MP1 - mp.mpf('1e-50'), mp.mpf(str(p0_hint)) + eps)
            return mp.findroot(exc, (p_lo, p_hi), solver='illinois', tol=mp.mpf('1e-50'))
        except Exception:
            return mp.nan

def contour_posterior_mp(ps_f64, ps_mp, u_grid, p_target_mp, u_own, tau):
    """
    mp50 posterior computation.
    ps_f64: float64 price surface (for contour finding — fast)
    ps_mp:  mp50 price surface values at crossing points (computed on-the-fly)
    Contour crossings located by ps_f64; integrals evaluated at mp50.
    """
    G     = len(u_grid)
    pt_f  = float(p_target_mp)
    tau_mp = mp.mpf(str(tau))
    f0o   = sig_f_mp(u_own, 0, tau)
    f1o   = sig_f_mp(u_own, 1, tau)
    A0, A1, n = mp.mpf('0'), mp.mpf('0'), 0

    def add_crossing(u_j, u_l):
        nonlocal A0, A1, n
        A0 += sig_f_mp(u_j, 0, tau) * sig_f_mp(u_l, 0, tau)
        A1 += sig_f_mp(u_j, 1, tau) * sig_f_mp(u_l, 1, tau)
        n  += 1

    for j in range(G):
        for u_l in _crossings(ps_f64[j,:], u_grid, pt_f):
            add_crossing(u_grid[j], u_l)
    for l in range(G):
        for u_j in _crossings(ps_f64[:,l], u_grid, pt_f):
            add_crossing(u_j, u_grid[l])

    if n == 0:
        nl = mp.mpf(str(float(Lam(tau * u_own))))
        return max(_MP_EPS, min(_MP1 - _MP_EPS, nl))
    A0 /= n; A1 /= n
    den = f0o*A0 + f1o*A1
    if den < _MP_EPS:
        nl = mp.mpf(str(float(Lam(tau * u_own))))
        return max(_MP_EPS, min(_MP1 - _MP_EPS, nl))
    v = f1o*A1 / den
    return max(_MP_EPS, min(_MP1 - _MP_EPS, v))

# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity (PAVA) — works for both float and mpf
# ─────────────────────────────────────────────────────────────────────────────

def pava_inc(y):
    y = list(y); n = len(y)
    blocks = [[i] for i in range(n)]
    vals   = list(y)
    changed = True
    while changed:
        changed = False; i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i+1]:
                merged = blocks[i] + blocks[i+1]
                avg = sum(y[k] for k in merged) / len(merged)
                blocks[i] = merged; vals[i] = avg
                blocks.pop(i+1); vals.pop(i+1)
                changed = True
            else:
                i += 1
    res = [type(y[0])(0)] * n
    for blk, v in zip(blocks, vals):
        for k in blk:
            res[k] = v
    return res

def enforce_mono(mu_arr, G):
    for i in range(G):
        mu_arr[i] = pava_inc(mu_arr[i])
    Gp = len(mu_arr[0])
    for j in range(Gp):
        col = pava_inc([mu_arr[i][j] for i in range(G)])
        for i in range(G):
            mu_arr[i][j] = col[i]
    return mu_arr

# ─────────────────────────────────────────────────────────────────────────────
# p-grid construction
# ─────────────────────────────────────────────────────────────────────────────

def build_p_grids(u_grid, gamma, tau, Gp):
    G = len(u_grid)
    u_min, u_max = u_grid[0], u_grid[-1]
    p_grids = []
    for i in range(G):
        u_own = u_grid[i]
        def exc_lo(p):
            return sum(crra_d(float(Lam(tau*u)), p, gamma) for u in [u_own, u_min, u_min])
        def exc_hi(p):
            return sum(crra_d(float(Lam(tau*u)), p, gamma) for u in [u_own, u_max, u_max])
        try:    p_lo = brentq(exc_lo, 1e-7, 1-1e-7, xtol=1e-9)
        except: p_lo = 1e-4
        try:    p_hi = brentq(exc_hi, 1e-7, 1-1e-7, xtol=1e-9)
        except: p_hi = 1-1e-4
        p_lo = max(p_lo, 1e-5); p_hi = min(p_hi, 1-1e-5)
        lo_l = logit_f(p_lo); hi_l = logit_f(p_hi)
        p_grids.append([float(Lam(x)) for x in np.linspace(lo_l, hi_l, Gp)])
    return p_grids

# ─────────────────────────────────────────────────────────────────────────────
# One φ step — float64 version
# ─────────────────────────────────────────────────────────────────────────────

def phi_step_f64(u_grid, p_grids, mu_arr, gamma, tau):
    G = len(u_grid)
    mu_new   = [[0.0]*len(p_grids[i]) for i in range(G)]
    residuals = []
    for i in range(G):
        ps = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                ps[j,l] = solve_mc(u_grid[i], u_grid[j], u_grid[l],
                                   gamma, u_grid, p_grids, mu_arr)
        for k, p_tgt in enumerate(p_grids[i]):
            mv = contour_posterior(ps, u_grid, p_tgt, u_grid[i], tau)
            mu_new[i][k] = mv
            residuals.append(abs(mv - float(mu_arr[i][k])))
    mu_new = enforce_mono(mu_new, G)
    return mu_new, max(residuals), float(np.median(residuals))

# ─────────────────────────────────────────────────────────────────────────────
# One φ step — mp50 version
# ─────────────────────────────────────────────────────────────────────────────

def phi_step_mp50(u_grid, p_grids, mu_arr_mp, mu_arr_f64, gamma, tau):
    """
    mp50 Picard step.
    mu_arr_f64 is kept in sync (float conversions of mu_arr_mp) for fast bracketing.
    """
    G = len(u_grid)
    mu_new   = [[mp.mpf('0')]*len(p_grids[i]) for i in range(G)]
    residuals = []

    for i in range(G):
        u_own = u_grid[i]
        # Float64 price surface for contour finding (fast)
        ps_f64 = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                ps_f64[j,l] = solve_mc(u_own, u_grid[j], u_grid[l],
                                       gamma, u_grid, p_grids, mu_arr_f64)

        for k, p_tgt in enumerate(p_grids[i]):
            p_tgt_mp = mp.mpf(str(p_tgt))
            mv = contour_posterior_mp(ps_f64, None, u_grid, p_tgt_mp, u_own, tau)
            mu_new[i][k] = mv
            residuals.append(abs(mv - mu_arr_mp[i][k]))

    mu_new = enforce_mono(mu_new, G)
    F_max = max(residuals)
    F_med = sorted(residuals)[len(residuals)//2]
    return mu_new, F_max, F_med

# ─────────────────────────────────────────────────────────────────────────────
# Anderson acceleration (float64 only, for Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def flatten(mu_arr):
    return np.array([float(v) for row in mu_arr for v in row])

def unflatten(vec, mu_arr):
    out = []; k = 0
    for row in mu_arr:
        n = len(row)
        out.append(list(np.clip(vec[k:k+n], 1e-12, 1-1e-12)))
        k += n
    return out

def anderson_update(x_hist, g_hist, x_new, g_new, m):
    """
    Anderson Type-1 mixing (Walker & Ni 2011).
    Returns x* = G θ* where θ* minimises ||F θ||² s.t. Σθ=1.
    """
    x_hist.append(x_new.copy()); g_hist.append(g_new.copy())
    if len(x_hist) > m+1: x_hist.pop(0); g_hist.pop(0)
    k = len(x_hist)
    if k < 2: return g_new

    G   = np.array(g_hist)                     # k × d
    F   = G - np.array(x_hist)                 # k × d   (f_i = Φ(x_i) - x_i)
    FTF = F @ F.T + 1e-12*np.eye(k)            # k × k
    try:
        v     = np.linalg.solve(FTF, np.ones(k))
        theta = v / v.sum()
        x_aa  = theta @ G                       # d,
        return np.clip(x_aa, 1e-12, 1-1e-12)
    except Exception:
        return g_new

# ─────────────────────────────────────────────────────────────────────────────
# Weighted 1-R²
# ─────────────────────────────────────────────────────────────────────────────

def compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau):
    G = len(u_grid)
    f0g = [sig_f(u,0,tau) for u in u_grid]
    f1g = [sig_f(u,1,tau) for u in u_grid]
    mu_f64 = [[float(v) for v in row] for row in mu_arr]
    Ts, lp, w = [], [], []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = solve_mc(u_grid[i],u_grid[j],u_grid[l],
                             gamma, u_grid, p_grids, mu_f64)
                if math.isnan(p) or p < 1e-6 or p > 1-1e-6: continue
                wt = 0.5*(f0g[i]*f0g[j]*f0g[l] + f1g[i]*f1g[j]*f1g[l])
                Ts.append(tau*(u_grid[i]+u_grid[j]+u_grid[l]))
                lp.append(logit_f(p))
                w.append(wt)
    if len(Ts) < 10: return float('nan'), float('nan')
    T = np.array(Ts); lp = np.array(lp); w = np.array(w); w /= w.sum()
    sl, ic = np.polyfit(T, lp, 1, w=np.sqrt(w*len(w)))
    mean_lp = np.average(lp, weights=w)
    var_t = np.average((lp-mean_lp)**2, weights=w)
    var_r = np.average((lp - sl*T - ic)**2, weights=w)
    return (var_r/var_t if var_t > 1e-30 else 0.0), sl

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_ckpt(path):
    with open(path) as f: d = json.load(f)
    u_grid  = np.array([float(x) for x in d['u_grid']])
    p_grids = [[float(x) for x in row] for row in d['p_grid']]
    mu_arr  = [[float(x) for x in row] for row in d['mu_strings']]
    return d['G'], float(d['tau']), float(d['gamma']), u_grid, p_grids, mu_arr, d

def save_ckpt(path, G, tau, gamma, u_grid, p_grids, mu_arr,
              F_max, F_med, elapsed, hist, phase):
    F_str = f'{float(F_max):.6e}'
    d = dict(G=G, tau=tau, gamma=gamma, dps=MP_DPS,
             F_max=F_str, F_med=f'{float(F_med):.6e}',
             phase=phase, elapsed_s=round(elapsed,1), history=hist,
             u_grid=[str(x) for x in u_grid],
             p_grid=[[str(x) for x in row] for row in p_grids],
             mu_strings=[[str(float(x)) for x in row] for row in mu_arr])
    with open(path,'w') as f: json.dump(d,f)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    t0    = time.time()
    G     = args.G
    gamma = args.gamma
    tau   = args.tau
    tol   = args.tol

    u_grid = np.linspace(-args.umax, args.umax, G)
    print(f"[gamma={gamma:.4g} tau={tau:.4g}] G={G} umax={args.umax} "
          f"tol={tol:.1e}  dps={MP_DPS}", flush=True)

    print("  building p-grids...", end=' ', flush=True)
    p_grids = build_p_grids(u_grid, gamma, tau, Gp=G)
    print("done", flush=True)

    # ── Init μ* ──────────────────────────────────────────────────────────────
    if args.seed:
        _,_,_,u_s,pg_s,mu_s,_ = load_ckpt(args.seed)
        print(f"  warm-start from {args.seed.split('/')[-1]}", flush=True)
        mu_arr = []
        for i in range(G):
            row = [float(np.clip(interp_mu(u_grid[i], p, u_s, pg_s, mu_s),
                                 1e-6, 1-1e-6))
                   for p in p_grids[i]]
            mu_arr.append(row)
    else:
        print("  init: no-learning μ=Λ(τu)", flush=True)
        mu_arr = [[float(Lam(tau*u_grid[i]))]*G for i in range(G)]

    mu_arr = enforce_mono(mu_arr, G)

    history   = []
    last_save = t0
    F_max = F_med = float('nan')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — float64 + Anderson(m=5)
    # ══════════════════════════════════════════════════════════════════════════
    f64_iters  = min(F64_MAX, args.max_iter)
    ALPHA_P1   = 0.05   # plain Picard damping — proven stable from far-out start
    x_hist, g_hist = [], []   # kept for potential future use

    print(f"\n  ── Phase 1 (float64 Picard α={ALPHA_P1}) "
          f"target F_max<{F64_TOL:.0e} ──", flush=True)

    for it in range(1, f64_iters + 1):
        mu_phi, F_max, F_med = phi_step_f64(u_grid, p_grids, mu_arr, gamma, tau)
        elapsed = time.time() - t0

        # Plain damped Picard — Anderson overshoots badly when far from fixed point
        for i in range(G):
            for j in range(len(mu_arr[i])):
                mu_arr[i][j] = ALPHA_P1*mu_phi[i][j] + (1-ALPHA_P1)*mu_arr[i][j]
        mu_arr = enforce_mono(mu_arr, G)

        entry = {'phase':1,'iter':it,'F_max':F_max,'F_med':F_med,'t':round(elapsed,1)}
        history.append(entry)
        print(f"  [{elapsed:7.0f}s] P1 iter={it:4d}  "
              f"F_max={F_max:.3e}  F_med={F_med:.3e}", flush=True)

        if time.time() - last_save > 120 or F_max < tol:
            save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
                      F_max, F_med, elapsed, history, 1)
            last_save = time.time()

        if F_max < tol:
            print(f"\n  CONVERGED in Phase 1  F_max={F_max:.3e}", flush=True)
            break

        if F_max < F64_TOL:
            print(f"  → Phase 1 done  F_max={F_max:.3e}, switching to mp50", flush=True)
            break

    remaining = args.max_iter - it

    if F_max >= tol and remaining > 0:
        # ════════════════════════════════════════════════════════════════════
        # Phase 2 — mp50 Picard
        # ════════════════════════════════════════════════════════════════════
        print(f"\n  ── Phase 2 (mp{MP_DPS-2} Picard α={args.alpha}) "
              f"target F_max<{tol:.0e} ({remaining} iter left) ──", flush=True)

        # Convert mu_arr to mpf
        mu_arr_mp = [[mp.mpf(str(v)) for v in row] for row in mu_arr]
        mu_arr_f64 = [[float(v) for v in row] for row in mu_arr]

        for it2 in range(1, remaining + 1):
            mu_phi_mp, F_max, F_med = phi_step_mp50(
                u_grid, p_grids, mu_arr_mp, mu_arr_f64, gamma, tau)
            elapsed = time.time() - t0

            # Damped Picard update (args.alpha for Phase 2, default 0.3)
            alpha2 = args.alpha
            for i in range(G):
                for j in range(len(mu_arr_mp[i])):
                    mu_arr_mp[i][j] = (alpha2*mu_phi_mp[i][j] +
                                       (1-alpha2)*mu_arr_mp[i][j])
            mu_arr_mp = enforce_mono(mu_arr_mp, G)
            # Keep float64 shadow in sync
            mu_arr_f64 = [[float(v) for v in row] for row in mu_arr_mp]

            F_str = f'{float(F_max):.3e}'
            entry = {'phase':2,'iter':it2,
                     'F_max':str(F_max),'F_med':str(F_med),
                     't':round(elapsed,1)}
            history.append(entry)
            print(f"  [{elapsed:7.0f}s] P2 iter={it2:4d}  "
                  f"F_max={F_str}  F_med={float(F_med):.3e}", flush=True)

            if time.time() - last_save > 120 or F_max < tol:
                save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr_mp,
                          F_max, F_med, elapsed, history, 2)
                last_save = time.time()

            if F_max < tol:
                print(f"\n  CONVERGED in Phase 2  F_max={F_str}", flush=True)
                break

        mu_arr = mu_arr_mp   # final result is mpf
    else:
        mu_arr_mp = mu_arr   # already float, convert for saving
        mu_arr_f64 = mu_arr

    # ── Final metrics ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n  computing 1-R²...", flush=True)
    one_mR2, slope = compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau)
    print(f"  1-R²={one_mR2:.6f}  slope={slope:.6f}  "
          f"F_max={float(F_max):.3e}", flush=True)

    save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
              F_max, F_med, elapsed, history,
              2 if remaining > 0 else 1)
    print(f"  saved → {args.out}", flush=True)
    return F_max, one_mR2, slope


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma',    type=float, required=True)
    ap.add_argument('--tau',      type=float, required=True)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--seed',     default=None)
    ap.add_argument('--G',        type=int,   default=20)
    ap.add_argument('--umax',     type=float, default=5.0)
    ap.add_argument('--tol',      type=float, default=1e-25)
    ap.add_argument('--max_iter', type=int,   default=3000)
    ap.add_argument('--alpha',    type=float, default=0.3)
    ap.add_argument('--anderson', type=int,   default=5)
    args = ap.parse_args()
    run(args)
