#!/usr/bin/env python3
"""
Posterior-method v3 REE solver — two-phase precision.

Phase 1 (float64 Picard α=0.05):   approach F_max < F64_TOL (~1e-8)
Phase 2 (LM-Newton, no cutoff):    converge to tol=1e-25 via quadratic convergence
  - Jacobian J(x) = I - Φ'(x) computed by float64 finite differences
  - LM step: Δx = -(J^T J + λI)^{-1} J^T F(x)  [λ=0 → pure Newton]
  - Full step applied without cutoff
  - Residual verified with mp50 (dps=52) arithmetic

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
from scipy.linalg import lu_factor, lu_solve as scipy_lu_solve
import mpmath as mp
from mpmath import mp as mpctx

MP_DPS   = 52        # 50 significant decimal digits + 2 guard
F64_TOL  = 1e-8      # Picard phase target before switching to Newton
F64_MAX  = 3000      # max Picard iterations (plain Picard at rate ~0.966/step: ~535 iters)
N_NEWTON = 5         # max LM-Newton outer steps
LM_REG   = 0.0       # λ for LM (0 = pure Newton; small > 0 adds stability)

mpctx.dps = MP_DPS

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
        return float(m_a[j]) + f*(float(m_a[j+1]) - float(m_a[j]))

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
            out.append(u_grid[k] + f*(u_grid[k+1] - u_grid[k]))
    return out

def contour_posterior(price_slice, u_grid, p_target, u_own, tau):
    G = len(u_grid)
    f0o = sig_f(u_own, 0, tau); f1o = sig_f(u_own, 1, tau)
    A0, A1, n = 0.0, 0.0, 0
    for j in range(G):
        for u_l in _crossings(price_slice[j,:], u_grid, p_target):
            A0 += sig_f(u_grid[j],0,tau)*sig_f(u_l,0,tau)
            A1 += sig_f(u_grid[j],1,tau)*sig_f(u_l,1,tau)
            n += 1
    for l in range(G):
        for u_j in _crossings(price_slice[:,l], u_grid, p_target):
            A0 += sig_f(u_j,0,tau)*sig_f(u_grid[l],0,tau)
            A1 += sig_f(u_j,1,tau)*sig_f(u_grid[l],1,tau)
            n += 1
    if n == 0:
        return float(Lam(tau * u_own))
    A0 /= n; A1 /= n
    den = f0o*A0 + f1o*A1
    if den < 1e-30: return float(Lam(tau * u_own))
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
    G   = len(u_grid)
    uf  = float(u_val); pf = float(p_val)
    if uf <= u_grid[0]:    i_lo, i_hi = 0, 1
    elif uf >= u_grid[-1]: i_lo, i_hi = G-2, G-1
    else:
        i_lo = int(np.searchsorted(u_grid, uf)) - 1
        i_hi = i_lo + 1

    def at(idx):
        p_a = p_grids[idx]; m_a = mu_arr_mp[idx]
        if pf <= p_a[0]:  return m_a[0]
        if pf >= p_a[-1]: return m_a[-1]
        j  = max(0, min(int(np.searchsorted(p_a, pf)) - 1, len(p_a)-2))
        df = mp.mpf(str(p_a[j+1] - p_a[j]))
        f  = mp.mpf(str(pf - p_a[j])) / (df if df > _MP_EPS else _MP_EPS)
        return m_a[j] + f*(m_a[j+1] - m_a[j])

    mu_lo, mu_hi = at(i_lo), at(i_hi)
    dug = mp.mpf(str(u_grid[i_hi] - u_grid[i_lo]))
    fu  = mp.mpf(str(uf - u_grid[i_lo])) / (dug if dug > _MP_EPS else _MP_EPS)
    v   = mu_lo + fu*(mu_hi - mu_lo)
    return max(_MP_EPS, min(_MP1 - _MP_EPS, v))

def contour_posterior_mp(ps_f64, u_grid, p_target_mp, u_own, tau):
    """mp50 posterior; contour located via float64 price surface."""
    pt_f = float(p_target_mp)
    f0o  = sig_f_mp(u_own, 0, tau); f1o = sig_f_mp(u_own, 1, tau)
    A0, A1, n = mp.mpf('0'), mp.mpf('0'), 0

    def add(uj, ul):
        nonlocal A0, A1, n
        A0 += sig_f_mp(uj, 0, tau) * sig_f_mp(ul, 0, tau)
        A1 += sig_f_mp(uj, 1, tau) * sig_f_mp(ul, 1, tau)
        n  += 1

    G = len(u_grid)
    for j in range(G):
        for u_l in _crossings(ps_f64[j,:], u_grid, pt_f):
            add(u_grid[j], u_l)
    for l in range(G):
        for u_j in _crossings(ps_f64[:,l], u_grid, pt_f):
            add(u_j, u_grid[l])

    if n == 0:
        v = mp.mpf(str(float(Lam(tau * u_own))))
        return max(_MP_EPS, min(_MP1 - _MP_EPS, v))
    A0 /= n; A1 /= n
    den = f0o*A0 + f1o*A1
    if den < _MP_EPS:
        v = mp.mpf(str(float(Lam(tau * u_own))))
        return max(_MP_EPS, min(_MP1 - _MP_EPS, v))
    v = f1o*A1 / den
    return max(_MP_EPS, min(_MP1 - _MP_EPS, v))

# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity (PAVA)
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
                avg    = sum(y[k] for k in merged) / len(merged)
                blocks[i] = merged; vals[i] = avg
                blocks.pop(i+1); vals.pop(i+1)
                changed = True
            else:
                i += 1
    res = list(y)
    for blk, v in zip(blocks, vals):
        for k in blk: res[k] = v
    return res

def enforce_mono(mu_arr, G):
    for i in range(G): mu_arr[i] = pava_inc(mu_arr[i])
    Gp = len(mu_arr[0])
    for j in range(Gp):
        col = pava_inc([mu_arr[i][j] for i in range(G)])
        for i in range(G): mu_arr[i][j] = col[i]
    return mu_arr

# ─────────────────────────────────────────────────────────────────────────────
# p-grid construction
# ─────────────────────────────────────────────────────────────────────────────

def build_p_grids(u_grid, gamma, tau, Gp):
    G = len(u_grid); u_min, u_max = u_grid[0], u_grid[-1]
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
# φ step — float64
# ─────────────────────────────────────────────────────────────────────────────

def phi_step_f64(u_grid, p_grids, mu_arr, gamma, tau):
    G = len(u_grid)
    mu_new = [[0.0]*len(p_grids[i]) for i in range(G)]
    for i in range(G):
        ps = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                ps[j,l] = solve_mc(u_grid[i], u_grid[j], u_grid[l],
                                   gamma, u_grid, p_grids, mu_arr)
        for k, p_tgt in enumerate(p_grids[i]):
            mu_new[i][k] = contour_posterior(ps, u_grid, p_tgt, u_grid[i], tau)
    mu_new = enforce_mono(mu_new, G)
    # residual computed AFTER monotone projection — true fixed-point residual
    residuals = [abs(float(mu_new[i][k]) - float(mu_arr[i][k]))
                 for i in range(G) for k in range(len(mu_arr[i]))]
    return mu_new, max(residuals), float(np.median(residuals))

# ─────────────────────────────────────────────────────────────────────────────
# φ step — mp50
# Contour locations from float64 price surface; densities in mp50.
# ─────────────────────────────────────────────────────────────────────────────

def phi_step_mp50(u_grid, p_grids, mu_arr_mp, mu_arr_f64, gamma, tau):
    G = len(u_grid)
    mu_new = [[mp.mpf('0')]*len(p_grids[i]) for i in range(G)]
    for i in range(G):
        u_own  = u_grid[i]
        ps_f64 = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                ps_f64[j,l] = solve_mc(u_own, u_grid[j], u_grid[l],
                                       gamma, u_grid, p_grids, mu_arr_f64)
        for k, p_tgt in enumerate(p_grids[i]):
            mu_new[i][k] = contour_posterior_mp(
                ps_f64, u_grid, mp.mpf(str(p_tgt)), u_own, tau)
    mu_new = enforce_mono(mu_new, G)
    # residual computed AFTER monotone projection — true fixed-point residual
    residuals = [abs(mu_new[i][k] - mu_arr_mp[i][k])
                 for i in range(G) for k in range(len(mu_arr_mp[i]))]
    F_max = max(residuals)
    F_med = sorted(residuals)[len(residuals)//2]
    return mu_new, F_max, F_med

# ─────────────────────────────────────────────────────────────────────────────
# Flatten / unflatten
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

# ─────────────────────────────────────────────────────────────────────────────
# Jacobian of F(x) = x - Φ(x) by float64 forward differences
# ─────────────────────────────────────────────────────────────────────────────

def compute_jacobian(u_grid, p_grids, mu_arr, gamma, tau, G, t0, eps=1e-6):
    """
    Returns J (d×d float64), F_f64 (d,), elapsed_s.
    d = total cells = G × len(p_grids[0]).
    Cost: d + 1 phi_step_f64 evaluations ≈ d × 3.5s.
    """
    x0   = flatten(mu_arr)
    phi0, _, _ = phi_step_f64(u_grid, p_grids, mu_arr, gamma, tau)
    g0   = flatten(phi0)
    F_f64 = x0 - g0        # F(x) = x - Φ(x)
    d    = len(x0)
    J_phi = np.zeros((d, d))

    print(f"    computing Jacobian ({d}×{d}, {d} phi-steps)...", flush=True)
    for k in range(d):
        if k % 100 == 0:
            print(f"      col {k}/{d}  elapsed={time.time()-t0:.0f}s", flush=True)
        ek = np.zeros(d); ek[k] = eps
        x_p = np.clip(x0 + ek, 1e-12, 1-1e-12)
        mu_p = enforce_mono(unflatten(x_p, mu_arr), G)
        phi_p, _, _ = phi_step_f64(u_grid, p_grids, mu_p, gamma, tau)
        J_phi[:, k] = (flatten(phi_p) - g0) / eps

    J = np.eye(d) - J_phi   # Jacobian of F = x - Φ(x)
    print(f"    Jacobian done  {time.time()-t0:.0f}s  "
          f"cond={np.linalg.cond(J):.2e}", flush=True)
    return J, F_f64

# ─────────────────────────────────────────────────────────────────────────────
# LM-Newton step (no cutoff)
# ─────────────────────────────────────────────────────────────────────────────

def lm_newton_step(x0, J_f64, F_f64, mu_arr, u_grid, p_grids, gamma, tau, G,
                   lam=LM_REG):
    """
    Solves: (J^T J + λI) Δx = −J^T F  (λ=0 → pure Newton: J Δx = −F)
    Applies full step without cutoff: x_new = x0 + Δx
    Verifies with mp50 residual.
    Returns (mu_arr_new, x_new_f64, F_max_mp50, F_med_mp50, phi_mp50)
    """
    d = len(x0)
    # ── float64 solve ──────────────────────────────────────────────────────
    if lam == 0.0:
        lu, piv = lu_factor(J_f64)
        delta   = scipy_lu_solve((lu, piv), -F_f64)
    else:
        JtJ = J_f64.T @ J_f64 + lam * np.eye(d)
        lu, piv = lu_factor(JtJ)
        delta   = scipy_lu_solve((lu, piv), -(J_f64.T @ F_f64))

    # ── apply full step (no cutoff) ────────────────────────────────────────
    x_new   = np.clip(x0 + delta, 1e-12, 1-1e-12)
    mu_new  = enforce_mono(unflatten(x_new, mu_arr), G)

    # ── mp50 residual ──────────────────────────────────────────────────────
    mu_mp   = [[mp.mpf(str(v)) for v in row] for row in mu_new]
    mu_f64  = [[float(v) for v in row] for row in mu_new]
    phi_mp, F_max_mp, F_med_mp = phi_step_mp50(
        u_grid, p_grids, mu_mp, mu_f64, gamma, tau)

    return mu_new, x_new, F_max_mp, F_med_mp, phi_mp

# ─────────────────────────────────────────────────────────────────────────────
# Weighted 1-R²
# ─────────────────────────────────────────────────────────────────────────────

def compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau):
    G   = len(u_grid)
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
                wt = 0.5*(f0g[i]*f0g[j]*f0g[l]+f1g[i]*f1g[j]*f1g[l])
                Ts.append(tau*(u_grid[i]+u_grid[j]+u_grid[l]))
                lp.append(logit_f(p)); w.append(wt)
    if len(Ts) < 10: return float('nan'), float('nan')
    T  = np.array(Ts); lp = np.array(lp); w = np.array(w); w /= w.sum()
    sl, ic = np.polyfit(T, lp, 1, w=np.sqrt(w*len(w)))
    mean_lp = np.average(lp, weights=w)
    var_t   = np.average((lp-mean_lp)**2, weights=w)
    var_r   = np.average((lp - sl*T - ic)**2, weights=w)
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
             p_grid =[[str(x) for x in row] for row in p_grids],
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
    converged = False

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — float64 Picard α=0.05
    # ══════════════════════════════════════════════════════════════════════════
    f64_iters = min(F64_MAX, args.max_iter)
    ALPHA_P1  = 0.05  # small damping keeps the non-contracting Picard map stable

    f64_threshold = args.f64_tol  # CLI override; set >1 to skip Phase 1
    print(f"\n  ── Phase 1 (float64 Picard α={ALPHA_P1}) "
          f"until F_max < {f64_threshold:.0e} ──", flush=True)

    prev_F_max   = float('inf')
    bad_streak   = 0
    alpha_p1_eff = ALPHA_P1

    for it in range(1, f64_iters + 1):
        mu_phi, F_max, F_med = phi_step_f64(u_grid, p_grids, mu_arr, gamma, tau)
        elapsed = time.time() - t0

        # Adaptive α: if residual increased 3 steps running, halve step
        if F_max > prev_F_max:
            bad_streak += 1
            if bad_streak >= 3:
                alpha_p1_eff = max(0.05, alpha_p1_eff * 0.5)
                bad_streak   = 0
                print(f"  [adaptive] α reduced to {alpha_p1_eff:.3f}", flush=True)
        else:
            bad_streak = 0

        for i in range(G):
            for j in range(len(mu_arr[i])):
                mu_arr[i][j] = alpha_p1_eff*mu_phi[i][j] + (1-alpha_p1_eff)*mu_arr[i][j]
        mu_arr = enforce_mono(mu_arr, G)
        prev_F_max = F_max

        entry = {'phase':1,'iter':it,'F_max':F_max,'F_med':F_med,'t':round(elapsed,1)}
        history.append(entry)
        print(f"  [{elapsed:7.0f}s] P1 iter={it:4d}  "
              f"F_max={F_max:.3e}  F_med={F_med:.3e}  α={alpha_p1_eff:.3f}", flush=True)

        if time.time() - last_save > 120 or F_max < tol:
            save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
                      F_max, F_med, elapsed, history, 1)
            last_save = time.time()

        if F_max < tol:
            print(f"\n  CONVERGED in Phase 1  F_max={F_max:.3e}", flush=True)
            converged = True; break

        if F_max < f64_threshold:
            print(f"  Phase 1 done  F_max={F_max:.3e}", flush=True)
            break

    remaining = args.max_iter - it

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — LM-Newton (no cutoff)
    # ══════════════════════════════════════════════════════════════════════════
    if not converged and remaining > 0:
        lam   = args.lm_reg
        print(f"\n  ── Phase 2 (LM-Newton λ={lam}  max {N_NEWTON} steps) ──",
              flush=True)

        x_curr = flatten(mu_arr)

        for nit in range(1, N_NEWTON + 1):
            elapsed = time.time() - t0
            print(f"\n  [Newton step {nit}/{N_NEWTON}  {elapsed:.0f}s]", flush=True)

            # Jacobian (float64)
            J_f64, F_f64 = compute_jacobian(
                u_grid, p_grids, mu_arr, gamma, tau, G, t0)

            # LM-Newton step — full step, no cutoff
            mu_arr, x_curr, F_max, F_med, phi_mp = lm_newton_step(
                x_curr, J_f64, F_f64, mu_arr, u_grid, p_grids, gamma, tau, G,
                lam=lam)

            elapsed = time.time() - t0
            print(f"  [Newton {nit}] F_max(mp50)={float(F_max):.3e}  "
                  f"F_med(mp50)={float(F_med):.3e}  {elapsed:.0f}s", flush=True)

            entry = {'phase':2,'newton':nit,
                     'F_max':str(F_max),'F_med':str(F_med),'t':round(elapsed,1)}
            history.append(entry)

            save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
                      F_max, F_med, elapsed, history, 2)
            last_save = time.time()

            if F_max < tol:
                print(f"\n  CONVERGED in Phase 2 (Newton {nit})  "
                      f"F_max={float(F_max):.3e}", flush=True)
                converged = True; break

            # Use Newton iterate as new starting point for next step
            # The mp50 phi output is the new Picard update — apply it for cleanup
            alpha2 = args.alpha
            mu_arr_mp = [[mp.mpf(str(v)) for v in row] for row in mu_arr]
            for i in range(G):
                for j in range(len(mu_arr[i])):
                    mu_arr[i][j] = float(
                        alpha2*phi_mp[i][j] + (1-alpha2)*mu_arr_mp[i][j])
            mu_arr = enforce_mono(mu_arr, G)
            x_curr = flatten(mu_arr)

    # ── Final metrics ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n  computing 1-R²...", flush=True)
    one_mR2, slope = compute_1mR2(u_grid, p_grids, mu_arr, gamma, tau)
    print(f"  1-R²={one_mR2:.6f}  slope={slope:.6f}  "
          f"F_max={float(F_max):.3e}", flush=True)

    save_ckpt(args.out, G, tau, gamma, u_grid, p_grids, mu_arr,
              F_max, F_med, elapsed, history, 2 if not converged else 2)
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
    ap.add_argument('--alpha',    type=float, default=0.3,
                    help='Picard damping between Newton steps')
    ap.add_argument('--anderson', type=int,   default=0,
                    help='(unused, kept for CLI compatibility)')
    ap.add_argument('--lm_reg',   type=float, default=0.0,
                    help='LM regularisation λ (0=pure Newton)')
    ap.add_argument('--f64_tol',  type=float, default=1e-8,
                    help='Phase 1 exit threshold (default 1e-8); set >1 to skip Phase 1')
    args = ap.parse_args()
    run(args)
