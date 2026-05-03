#!/usr/bin/env python3
"""
mpmath polisher for posterior-method v3.

Loads a float64 checkpoint produced by run_gamma1_tau_sweep.py and runs
the same Phi map at high mpmath precision (default mp100), polishing the
solution so that ||F||inf < 1e-50 (or whatever the residual floor of the
algorithm allows on a given tau).

Algorithm: Aitken-accelerated damped Picard with PAVA monotonicity
projection.  Aitken's delta-squared on each cell gives super-linear
convergence near a fixed point — combined with a good warm start (the
float64 solution), three or four mpmath iterations is usually enough to
push the bulk residual from 1e-13 to below 1e-50.

The lens-corner cells (negligible signal-density weight) are typically
non-convergent in this formulation and stay at residual ~0.1.  We track
||F||_active = max |F| over rows whose own-signal density exceeds 1e-6
and use that as the pass criterion.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import mpmath as mp
import numpy as np


# ----------------------------------------------------------------------
# mpmath helpers (precision is set by the caller via mp.mp.dps)
# ----------------------------------------------------------------------

def mpf(x):
    return mp.mpf(x)


def lam_mp(z):
    return 1 / (1 + mp.exp(-z))


def logit_mp(p):
    return mp.log(p) - mp.log(1 - p)


def signal_density_mp(u, v, tau):
    mean = mpf(v) - mp.mpf("0.5")
    return mp.sqrt(tau / (2 * mp.pi)) * mp.exp(-tau / 2 * (u - mean) ** 2)


def crra_demand_mp(mu, p, gamma):
    """CRRA demand in mpmath. mu, p in (0,1)."""
    z = (logit_mp(mu) - logit_mp(p)) / gamma
    R = mp.exp(z)
    return (R - 1) / ((1 - p) + R * p)


def crra_demand_no_logit_mp(z, p, gamma):
    """CRRA demand given pre-computed (logit(mu)-logit(p))/gamma = z."""
    R = mp.exp(z)
    return (R - 1) / ((1 - p) + R * p)


# ----------------------------------------------------------------------
# Linear interpolation in mpmath
# ----------------------------------------------------------------------

def interp_row_p_mp(mu_row, p_row, p_query):
    """Linear interp of mu_row over p_row at p_query (mpmath).
    p_row is monotone increasing.  Clamps at boundaries."""
    n = len(p_row)
    if p_query <= p_row[0]:
        return mu_row[0]
    if p_query >= p_row[n - 1]:
        return mu_row[n - 1]
    # Bisection to find bracket
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_row[mid] <= p_query:
            lo = mid
        else:
            hi = mid
    frac = (p_query - p_row[lo]) / (p_row[lo + 1] - p_row[lo])
    return mu_row[lo] + frac * (mu_row[lo + 1] - mu_row[lo])


def extract_column_mp(mu, p_grids, p_query):
    """For each row i, interpolate mu[i, :] over p_grids[i, :] at p_query."""
    G = len(mu)
    return [interp_row_p_mp(mu[i], p_grids[i], p_query) for i in range(G)]


def interp_monotone_mp(targets, d_sorted, u_sorted):
    """Vector linear interp.  targets and result are 1-D lists (mpmath)."""
    n = len(d_sorted)
    out = [None] * len(targets)
    for k, t in enumerate(targets):
        if t <= d_sorted[0]:
            out[k] = u_sorted[0]
        elif t >= d_sorted[n - 1]:
            out[k] = u_sorted[n - 1]
        else:
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if d_sorted[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            frac = (t - d_sorted[lo]) / (d_sorted[lo + 1] - d_sorted[lo])
            out[k] = u_sorted[lo] + frac * (u_sorted[lo + 1] - u_sorted[lo])
    return out


# ----------------------------------------------------------------------
# mpmath Phi step
# ----------------------------------------------------------------------

def phi_step_mp(mu, u_grid, p_grids, tau, gamma, f0_grid, f1_grid):
    """One iteration of Phi at full mpmath precision.

    Inputs and outputs are nested lists of mpf, shape (G, G).

    f0_grid, f1_grid are precomputed signal densities at u_grid for v=0/1.
    """
    G = len(mu)
    u_min = u_grid[0]
    u_max = u_grid[G - 1]
    one = mpf(1)

    mu_new = [list(row) for row in mu]

    for i in range(G):
        u_i = u_grid[i]
        f0_i = f0_grid[i]
        f1_i = f1_grid[i]
        for j in range(G):
            p = p_grids[i][j]
            mu_col = extract_column_mp(mu, p_grids, p)
            # Demand vector
            log_p = logit_mp(p)
            d = []
            for k in range(G):
                z = (logit_mp(mu_col[k]) - log_p) / gamma
                d.append(crra_demand_no_logit_mp(z, p, gamma))

            # d is monotone increasing in u (built that way by Bayes monotonicity)
            # Sort to match the float64 Phi exactly (np.argsort).
            order = sorted(range(G), key=lambda k: d[k])
            d_sorted = [d[k] for k in order]
            u_sorted = [u_grid[k] for k in order]

            d_i = d[i]
            targets = [-(d_i + dk) for dk in d]

            # Validity: target must be inside d's range
            d_lo = d_sorted[0]
            d_hi = d_sorted[G - 1]
            valid = [(t >= d_lo and t <= d_hi) for t in targets]

            if sum(valid) < 2:
                continue

            u3_star = interp_monotone_mp(targets, d_sorted, u_sorted)

            A1 = mpf(0)
            A0 = mpf(0)
            for k in range(G):
                if not valid[k]:
                    continue
                u3 = u3_star[k]
                f1u3 = signal_density_mp(u3, 1, tau)
                f0u3 = signal_density_mp(u3, 0, tau)
                A1 += f1_grid[k] * f1u3
                A0 += f0_grid[k] * f0u3

            denom = f0_i * A0 + f1_i * A1
            if denom <= 0:
                continue
            mu_new[i][j] = (f1_i * A1) / denom

    return mu_new


# ----------------------------------------------------------------------
# PAVA in mpmath (simple O(n) pool-adjacent-violators)
# ----------------------------------------------------------------------

def pava_mp(values):
    """In-place PAVA on a list of mpmath values (monotone non-decreasing)."""
    n = len(values)
    if n < 2:
        return list(values)
    # Use blocks
    block_vals = []
    block_lens = []
    for v in values:
        block_vals.append(v)
        block_lens.append(1)
        # Merge while violation
        while len(block_vals) >= 2 and block_vals[-2] > block_vals[-1]:
            v2 = block_vals.pop()
            l2 = block_lens.pop()
            v1 = block_vals.pop()
            l1 = block_lens.pop()
            merged = (v1 * l1 + v2 * l2) / (l1 + l2)
            block_vals.append(merged)
            block_lens.append(l1 + l2)
    out = []
    for v, l in zip(block_vals, block_lens):
        for _ in range(l):
            out.append(v)
    return out


def pava_project_mp(mu, u_grid, p_grids):
    """Two-pass PAVA: monotone in u (per column j), then in p (per row i)."""
    G = len(mu)
    out = [list(row) for row in mu]
    # Pass 1: u-direction
    for j in range(G):
        col = [out[i][j] for i in range(G)]
        col = pava_mp(col)
        for i in range(G):
            out[i][j] = col[i]
    # Pass 2: p-direction
    for i in range(G):
        out[i] = pava_mp(out[i])
    return out


# ----------------------------------------------------------------------
# Build p-grids in mpmath from u_grid (uses no-learning bracketing)
# ----------------------------------------------------------------------

def build_p_grids_mp(u_grid, tau, gamma, G_p):
    """Return p_lo, p_hi, p_grids in mpmath using mpmath findroot bisection."""
    G = len(u_grid)
    u_min, u_max = u_grid[0], u_grid[G - 1]

    def excess(p, u1, u2, u3):
        m1 = lam_mp(tau * u1)
        m2 = lam_mp(tau * u2)
        m3 = lam_mp(tau * u3)
        return (crra_demand_mp(m1, p, gamma)
                + crra_demand_mp(m2, p, gamma)
                + crra_demand_mp(m3, p, gamma))

    def bisect(u1, u2, u3):
        lo, hi = mpf("1e-100"), mpf(1) - mpf("1e-100")
        for _ in range(400):
            mid = (lo + hi) / 2
            f = excess(mid, u1, u2, u3)
            if abs(f) < mpf("1e-" + str(mp.mp.dps - 5)):
                return mid
            f_lo = excess(lo, u1, u2, u3)
            if f_lo * f < 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    p_lo = [bisect(u_grid[i], u_min, u_min) for i in range(G)]
    p_hi = [bisect(u_grid[i], u_max, u_max) for i in range(G)]

    p_grids = [[None] * G_p for _ in range(G)]
    for i in range(G):
        lp_lo = logit_mp(p_lo[i])
        lp_hi = logit_mp(p_hi[i])
        for j in range(G_p):
            t = mpf(j) / mpf(G_p - 1)
            lp = lp_lo + t * (lp_hi - lp_lo)
            p_grids[i][j] = lam_mp(lp)
    return p_lo, p_hi, p_grids


# ----------------------------------------------------------------------
# Polishing loop
# ----------------------------------------------------------------------

def to_mp_array(arr_f64, dps):
    """Convert numpy array (or list of lists) to mpmath nested list."""
    arr_f64 = np.asarray(arr_f64)
    if arr_f64.ndim == 1:
        return [mpf(repr(float(x))) for x in arr_f64]
    return [[mpf(repr(float(x))) for x in row] for row in arr_f64]


def from_mp_array(arr_mp):
    """Convert nested mpmath list to numpy float64 array."""
    if isinstance(arr_mp[0], list):
        return np.array([[float(x) for x in row] for row in arr_mp])
    return np.array([float(x) for x in arr_mp])


def to_mp_str(x, dps):
    return mp.nstr(x, dps, strip_zeros=False)


def f_max_active(F_mp, active_mask):
    """Max |F| over active cells (active_mask is np bool array)."""
    G = len(F_mp)
    fmax = mpf(0)
    for i in range(G):
        for j in range(G):
            if active_mask[i, j]:
                a = abs(F_mp[i][j])
                if a > fmax:
                    fmax = a
    return fmax


def F_func_flat(mu_flat, u_grid_mp, p_grids, tau, gamma, f0_grid, f1_grid, G):
    """Flatten 2D mpmath mu, run phi step, return F = Phi(mu) - mu (flat list)."""
    mu = [[mu_flat[i * G + j] for j in range(G)] for i in range(G)]
    mu_phi = phi_step_mp(mu, u_grid_mp, p_grids, tau, gamma, f0_grid, f1_grid)
    out = [None] * (G * G)
    for i in range(G):
        for j in range(G):
            out[i * G + j] = mu_phi[i][j] - mu[i][j]
    return out


def jvp_fd(mu_flat, v, eps, F0_cache, u_grid_mp, p_grids, tau, gamma,
           f0_grid, f1_grid, G):
    """Finite-difference Jacobian-vector product:  J*v ~= (F(mu+eps*v) - F0) / eps.
    F0_cache is the precomputed F at mu_flat (one phi eval saved per call)."""
    mu_pert = [mu_flat[k] + eps * v[k] for k in range(G * G)]
    F1 = F_func_flat(mu_pert, u_grid_mp, p_grids, tau, gamma,
                     f0_grid, f1_grid, G)
    return [(F1[k] - F0_cache[k]) / eps for k in range(G * G)]


def gmres_mp(matvec, b, restart=30, tol=None, maxiter=60, verbose=False):
    """Minimal GMRES in mpmath. Solves A x = b where matvec(v) returns A*v.
    Returns (x, residual_norm, n_matvec)."""
    n = len(b)
    if tol is None:
        tol = mpf("1e-" + str(mp.mp.dps - 5))

    x = [mpf(0)] * n
    r = list(b)
    beta = mp.sqrt(sum(rk * rk for rk in r))
    if beta == 0:
        return x, mpf(0), 0

    n_matvec = 0

    for outer in range(maxiter // restart + 1):
        V = [[rk / beta for rk in r]]
        H = [[mpf(0)] * (restart + 1) for _ in range(restart + 1)]
        g = [mpf(0)] * (restart + 1)
        g[0] = beta
        cs = [mpf(0)] * restart
        sn = [mpf(0)] * restart

        m_used = 0
        for j in range(restart):
            w = matvec(V[j])
            n_matvec += 1
            for i in range(j + 1):
                H[i][j] = sum(w[k] * V[i][k] for k in range(n))
                w = [w[k] - H[i][j] * V[i][k] for k in range(n)]
            H[j + 1][j] = mp.sqrt(sum(wk * wk for wk in w))
            if H[j + 1][j] != 0:
                V.append([wk / H[j + 1][j] for wk in w])
            else:
                V.append([mpf(0)] * n)

            for i in range(j):
                temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j]
                H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j]
                H[i][j] = temp
            denom = mp.sqrt(H[j][j] ** 2 + H[j + 1][j] ** 2)
            if denom == 0:
                cs[j], sn[j] = mpf(1), mpf(0)
            else:
                cs[j] = H[j][j] / denom
                sn[j] = H[j + 1][j] / denom
            H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j]
            H[j + 1][j] = mpf(0)
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            m_used = j + 1
            if verbose and (j + 1) % 5 == 0:
                print(f"      gmres j={j+1}: |r|={mp.nstr(abs(g[j+1]), 4)}",
                      flush=True)
            if abs(g[j + 1]) < tol:
                break

        m = m_used
        y = [mpf(0)] * m
        for i in range(m - 1, -1, -1):
            s = g[i]
            for k in range(i + 1, m):
                s = s - H[i][k] * y[k]
            y[i] = s / H[i][i]

        for k in range(n):
            for i in range(m):
                x[k] = x[k] + V[i][k] * y[i]

        Ax = matvec(x)
        n_matvec += 1
        r = [b[k] - Ax[k] for k in range(n)]
        beta = mp.sqrt(sum(rk * rk for rk in r))
        if beta < tol:
            break

    return x, beta, n_matvec


def polish_one(checkpoint_path, dps=100, target=mp.mpf("1e-50"),
               max_iter=10, weight_floor=1e-6,
               u_grid_mp=None, p_grids_mp=None,
               gmres_restart=30, gmres_maxiter=80):
    """Polish a single checkpoint via Newton-Krylov in mpmath."""
    mp.mp.dps = dps
    target = mpf(target) if not isinstance(target, mp.mpf) else target

    with open(checkpoint_path) as f:
        d = json.load(f)
    G = d["G"]
    tau = mpf(repr(d["tau"]))
    gamma = mpf(repr(d["gamma"]))
    UMAX = d["UMAX"]

    print(f"=== {os.path.basename(checkpoint_path)} ===")
    print(f"  G={G}, tau={d['tau']}, gamma={d['gamma']}, UMAX={UMAX}, dps={dps}")

    if u_grid_mp is None:
        u_grid_mp = [mpf(-UMAX) + mpf(2 * UMAX) * mpf(i) / mpf(G - 1)
                     for i in range(G)]

    p_grids = [[mpf(s) for s in row] for row in d["p_grid"]]

    # Initial mu from checkpoint, flattened
    mu_flat = []
    for i in range(G):
        for j in range(G):
            mu_flat.append(mpf(d["mu_strings"][i][j]))

    # Active mask
    u_arr = np.array([float(u) for u in u_grid_mp])
    f0 = np.sqrt(float(tau) / (2 * np.pi)) * np.exp(-float(tau) / 2 * (u_arr + 0.5) ** 2)
    f1 = np.sqrt(float(tau) / (2 * np.pi)) * np.exp(-float(tau) / 2 * (u_arr - 0.5) ** 2)
    row_w = 0.5 * (f0 + f1)
    active_rows = row_w > weight_floor
    active_mask = np.zeros((G, G), dtype=bool)
    active_mask[active_rows, :] = True
    print(f"  active rows: {np.where(active_rows)[0].tolist()}")

    f0_grid = [signal_density_mp(u, 0, tau) for u in u_grid_mp]
    f1_grid = [signal_density_mp(u, 1, tau) for u in u_grid_mp]

    history = []
    eps_jvp = mpf("10") ** (-(dps // 2))

    def stats(F_flat):
        F_max = max(abs(F_flat[k]) for k in range(G * G))
        F_med_sorted = sorted([abs(F_flat[k]) for k in range(G * G)])
        F_med = F_med_sorted[len(F_med_sorted) // 2]
        F_act = mpf(0)
        for i in range(G):
            for j in range(G):
                if active_mask[i, j]:
                    a = abs(F_flat[i * G + j])
                    if a > F_act:
                        F_act = a
        return F_max, F_med, F_act

    t0 = time.time()
    F_act_prev = None
    for it in range(1, max_iter + 1):
        F_flat = F_func_flat(mu_flat, u_grid_mp, p_grids, tau, gamma,
                              f0_grid, f1_grid, G)
        F_max, F_med, F_act = stats(F_flat)
        elapsed = time.time() - t0
        print(f"  iter {it:2d}  F_max={mp.nstr(F_max, 6)}  "
              f"F_act={mp.nstr(F_act, 6)}  F_med={mp.nstr(F_med, 6)}  "
              f"({elapsed:.1f}s)")
        history.append({
            "iter": it,
            "F_max": to_mp_str(F_max, dps),
            "F_med": to_mp_str(F_med, dps),
            "F_active": to_mp_str(F_act, dps),
        })

        if F_act < target:
            print(f"  CONVERGED at iter {it}")
            break

        if F_act_prev is not None and F_act >= F_act_prev * mpf("0.9"):
            print(f"  STALLED")
            break
        F_act_prev = F_act

        # Newton-Krylov step:  J * delta = -F,  mu <- mu + delta
        # Cache F0 = F(mu_flat) (already computed above) for jvp_fd reuse
        F0_cache = F_flat

        def matvec(v):
            return jvp_fd(mu_flat, v, eps_jvp, F0_cache, u_grid_mp, p_grids,
                          tau, gamma, f0_grid, f1_grid, G)

        rhs = [-Fk for Fk in F_flat]
        gmres_tol = max(F_act * mpf("1e-3"), mpf("10") ** (-(dps - 5)))
        delta, gmres_res, n_mv = gmres_mp(matvec, rhs, restart=gmres_restart,
                                            tol=gmres_tol, maxiter=gmres_maxiter,
                                            verbose=True)
        delta_norm = max(abs(d) for d in delta)
        print(f"    GMRES residual={mp.nstr(gmres_res, 4)}, "
              f"||delta||inf={mp.nstr(delta_norm, 4)}, n_matvec={n_mv}",
              flush=True)

        # Take the step (no damping; near a root Newton has α=1)
        mu_flat = [mu_flat[k] + delta[k] for k in range(G * G)]
        # Clip to safely off boundary
        eps_clip = mpf("10") ** (-(dps - 5))
        for k in range(G * G):
            if mu_flat[k] < eps_clip:
                mu_flat[k] = eps_clip
            elif mu_flat[k] > 1 - eps_clip:
                mu_flat[k] = 1 - eps_clip

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. "
          f"||F||inf={mp.nstr(F_max, 4)}, "
          f"||F||active={mp.nstr(F_act, 4)}, "
          f"||F||med={mp.nstr(F_med, 4)}")

    mu = [[mu_flat[i * G + j] for j in range(G)] for i in range(G)]
    return {
        "G": G,
        "UMAX": UMAX,
        "tau": float(tau),
        "gamma": float(gamma),
        "trim": d.get("trim", 0.0),
        "dps": dps,
        "F_max": to_mp_str(F_max, dps),
        "F_med": to_mp_str(F_med, dps),
        "F_active": to_mp_str(F_act, dps),
        "u_grid": [to_mp_str(u, dps) for u in u_grid_mp],
        "p_grid": [[to_mp_str(x, dps) for x in row] for row in p_grids],
        "mu_strings": [[to_mp_str(x, dps) for x in row] for row in mu],
        "history": history,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=100)
    ap.add_argument("--tol", type=str, default="1e-50")
    ap.add_argument("--maxiter", type=int, default=15)
    ap.add_argument("--only", type=float, default=None,
                    help="Polish only this tau (in units of tau, e.g. 2.0)")
    args = ap.parse_args()

    indir = "results/full_ree"
    files = sorted([f for f in os.listdir(indir)
                    if f.startswith("task3_g100_t") and f.endswith("_mp50.json")])
    if args.only is not None:
        token = f"t{int(round(args.only * 100)):04d}"
        files = [f for f in files if token in f]

    target = mp.mpf(args.tol)
    print(f"Polishing {len(files)} checkpoints at dps={args.dps}, target={args.tol}")

    for fname in files:
        path = os.path.join(indir, fname)
        try:
            payload = polish_one(path, dps=args.dps, target=target,
                                  max_iter=args.maxiter)
        except Exception as exc:
            print(f"  FAILED on {fname}: {exc}")
            continue
        out_name = fname.replace("_mp50.json", f"_mp{args.dps}.json")
        out_path = os.path.join(indir, out_name)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"  wrote {out_name}\n")


if __name__ == "__main__":
    main()
