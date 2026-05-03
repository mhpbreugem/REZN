"""mpmath K=3 phi map for high-precision Picard sharpening.

Pure-Python mpmath port of code/contour_K3_halo.phi_K3_halo (the
linear-scan kernel). Used to sharpen float64 checkpoints to
||F||_inf < 1e-50 at dps=100.

Design:
- mpmath context dps controlled at module level via `set_dps`.
- All elementary ops go through mpmath (mpf type).
- Halo cells (boundary padding) are held FIXED at the no-learning
  seed; only inner cells are updated.

Speed-up tricks:
- Cache f_signal evaluations on the u-grid (reused many times).
- For each agent, pre-extract the 2D slice as a flat list of mpf for
  cache efficiency.
"""

from __future__ import annotations

import math
from typing import List

import mpmath as mp


def set_dps(dps: int = 100) -> None:
    mp.mp.dps = dps


# ------------------------------- primitives -------------------------------

def lam_mp(z):
    """Logistic 1/(1+exp(-z)) in mpmath."""
    z = mp.mpf(z)
    if z >= 0:
        e = mp.exp(-z)
        return 1 / (1 + e)
    e = mp.exp(z)
    return e / (1 + e)


def logit_mp(p):
    p = mp.mpf(p)
    return mp.log(p) - mp.log(1 - p)


def f_signal_mp(u, v: int, tau, *, sqrt_tau_over_2pi=None):
    """Signal density f_v(u) = sqrt(tau/2pi) * exp(-tau/2 (u - mean)^2)."""
    u = mp.mpf(u)
    tau = mp.mpf(tau)
    mean = mp.mpf("0.5") if v == 1 else mp.mpf("-0.5")
    d = u - mean
    if sqrt_tau_over_2pi is None:
        sqrt_tau_over_2pi = mp.sqrt(tau / (2 * mp.pi))
    return sqrt_tau_over_2pi * mp.exp(-tau * d * d / 2)


# ---- demand and market clearing in mpmath -------------------------------

EPS_PRICE_MP = None  # set after dps


def _ensure_eps():
    global EPS_PRICE_MP
    EPS_PRICE_MP = mp.mpf("1e-12")


def x_crra_mp(mu, p, gamma, W=mp.mpf("1")):
    z = (logit_mp(mu) - logit_mp(p)) / gamma
    if z >= 0:
        e = mp.exp(-z)
        return W * (1 - e) / ((1 - p) * e + p)
    e = mp.exp(z)
    return W * (e - 1) / ((1 - p) + p * e)


def clear_crra_mp(mu_vec, gamma_vec, W_vec, *, p_init=None,
                  tol=None):
    """Newton (mpmath findroot) for sum x_k(mu_k, p) = 0 in p in (0,1).

    Aggregate excess demand is strictly decreasing in p; Newton from
    a sensible interior starting point converges quadratically.
    """
    if tol is None:
        tol = mp.mpf(10) ** -(mp.mp.dps - 5)

    def excess(p):
        s = mp.mpf("0")
        for k in range(len(mu_vec)):
            s += x_crra_mp(mu_vec[k], p, gamma_vec[k], W_vec[k])
        return s

    if p_init is None:
        # Use the average of mu_k as a decent starting guess.
        p_init = sum(mu_vec) / len(mu_vec)
        p_init = max(mp.mpf("1e-6"), min(1 - mp.mpf("1e-6"), p_init))

    # mpmath's findroot with secant or Newton. We use 'anderson' (no
    # derivatives needed). Falls back to bisection guard if it strays.
    try:
        p = mp.findroot(excess, p_init, solver='anderson',
                        tol=tol, maxsteps=80, verify=False)
    except Exception:
        # Fall back to bracketed bisection if Newton diverges.
        a = mp.mpf("1e-50")
        b = 1 - mp.mpf("1e-50")
        fa = excess(a)
        fb = excess(b)
        if fa <= 0:
            return a
        if fb >= 0:
            return b
        for _ in range(400):
            c = (a + b) / 2
            fc = excess(c)
            if fc >= 0:
                a = c
            else:
                b = c
            if (b - a) < tol:
                break
        return (a + b) / 2

    # Clamp into (0,1)
    if p < mp.mpf("1e-50"):
        p = mp.mpf("1e-50")
    elif p > 1 - mp.mpf("1e-50"):
        p = 1 - mp.mpf("1e-50")
    return p


# ------------------------- contour-evidence scan -------------------------

def _scan_axis_2d(P_slice_2d, p_target, axis: int, a_idx: int,
                  u_full, tau_a, tau_off, f1_grid, f0_grid, sqrt_tt2pi,
                  acc):
    """Scan one axis of a (G_full, G_full) 2D slice for level-set
    crossings of p_target. Equivalent to _scan_axis_K3 in the float64
    kernel. acc is a list [acc0, acc1] mutated in place.

    P_slice_2d[m][n] is mpf (m=axis-0, n=axis-1).
    """
    G_full = len(u_full)
    u_a = u_full[a_idx]
    f0_a = f0_grid[a_idx]
    f1_a = f1_grid[a_idx]

    if axis == 0:
        get = lambda i: P_slice_2d[i][a_idx]
    else:
        get = lambda i: P_slice_2d[a_idx][i]

    prev_v = get(0)
    for i in range(G_full - 1):
        next_v = get(i + 1)
        d_prev = prev_v - p_target
        d_next = next_v - p_target
        if d_prev * d_next <= 0:
            denom = next_v - prev_v
            if denom == 0:
                prev_v = next_v
                continue
            t = (p_target - prev_v) / denom
            if t < 0:
                t = mp.mpf(0)
            elif t > 1:
                t = mp.mpf(1)
            u_off = (1 - t) * u_full[i] + t * u_full[i + 1]
            f0_off = f_signal_mp(u_off, 0, tau_off,
                                 sqrt_tau_over_2pi=sqrt_tt2pi[1])
            f1_off = f_signal_mp(u_off, 1, tau_off,
                                 sqrt_tau_over_2pi=sqrt_tt2pi[1])
            acc[0] += f0_a * f0_off
            acc[1] += f1_a * f1_off
        prev_v = next_v


def agent_evidence_2d(P_slice_2d, p_target, u_full, tau_a, tau_off,
                      f1_grid_a, f0_grid_a, f1_grid_off, f0_grid_off,
                      sqrt_tt2pi):
    """Compute (A0, A1) by scanning the 2D slice along both axes
    and averaging. Matches _agent_evidence_K3 in the float64 kernel.

    f1_grid_a, f0_grid_a are signal densities under tau_a evaluated
    on u_full (used for the on-grid axis).
    sqrt_tt2pi is a tuple (sqrt_a, sqrt_off) of sqrt(tau/2pi) constants.
    """
    G_full = len(u_full)
    a0_total = mp.mpf("0")
    a1_total = mp.mpf("0")
    for axis in (0, 1):
        acc = [mp.mpf("0"), mp.mpf("0")]
        for a_idx in range(G_full):
            _scan_axis_2d(P_slice_2d, p_target, axis, a_idx,
                          u_full, tau_a, tau_off, f1_grid_a, f0_grid_a,
                          sqrt_tt2pi, acc)
        a0_total += acc[0]
        a1_total += acc[1]
    return a0_total / 2, a1_total / 2


def bayes_mp(u_own, tau_own, A0, A1, f1_own=None, f0_own=None,
             sqrt_tt2pi_own=None):
    if f0_own is None:
        f0_own = f_signal_mp(u_own, 0, tau_own,
                             sqrt_tau_over_2pi=sqrt_tt2pi_own)
    if f1_own is None:
        f1_own = f_signal_mp(u_own, 1, tau_own,
                             sqrt_tau_over_2pi=sqrt_tt2pi_own)
    num = f1_own * A1
    den = f0_own * A0 + num
    if den <= 0:
        return mp.mpf("0.5")
    mu = num / den
    eps = mp.mpf("1e-50")
    if mu < eps:
        return eps
    if mu > 1 - eps:
        return 1 - eps
    return mu


# -------------------------- full phi K=3 (mpmath) --------------------------

def phi_K3_mp(P_full, u_full, inner_lo, inner_hi, tau_vec, gamma_vec,
              W_vec, *, f1_grid=None, f0_grid=None, sqrt_tt2pi=None):
    """Full phi map at mp dps. P_full is a 3D nested list of mpf.

    Returns a NEW 3D nested list with the inner cells updated; halo
    (cells with any index outside [inner_lo, inner_hi)) are unchanged.

    Args:
        u_full: list of mpf, length G_full
        inner_lo, inner_hi: ints, inner index range
        tau_vec: 3-tuple of mpf
        gamma_vec: 3-tuple of mpf
        W_vec: 3-tuple of mpf
        f1_grid, f0_grid: optional precomputed list-of-list f_v(u_full)
                          shape (3, G_full); if None, computed.
        sqrt_tt2pi: optional 3-tuple of sqrt(tau/2pi) constants.
    """
    G_full = len(u_full)
    if sqrt_tt2pi is None:
        sqrt_tt2pi = tuple(mp.sqrt(tau_vec[k] / (2 * mp.pi))
                           for k in range(3))
    if f1_grid is None or f0_grid is None:
        f1_grid = [[f_signal_mp(u_full[i], 1, tau_vec[k],
                                sqrt_tau_over_2pi=sqrt_tt2pi[k])
                    for i in range(G_full)] for k in range(3)]
        f0_grid = [[f_signal_mp(u_full[i], 0, tau_vec[k],
                                sqrt_tau_over_2pi=sqrt_tt2pi[k])
                    for i in range(G_full)] for k in range(3)]

    # Deep copy
    P_new = [[[P_full[i][j][l] for l in range(G_full)]
              for j in range(G_full)] for i in range(G_full)]

    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i][j][l]

                # Agent 0: own = i; slice P_full[i, :, :] (axes 1,2)
                slice_0 = P_full[i]   # [j_idx][l_idx]
                A0_0, A1_0 = agent_evidence_2d(
                    slice_0, p, u_full,
                    tau_a=tau_vec[1], tau_off=tau_vec[2],
                    f1_grid_a=f1_grid[1], f0_grid_a=f0_grid[1],
                    f1_grid_off=f1_grid[2], f0_grid_off=f0_grid[2],
                    sqrt_tt2pi=(sqrt_tt2pi[1], sqrt_tt2pi[2]),
                )
                mu0 = bayes_mp(u_full[i], tau_vec[0], A0_0, A1_0,
                               f1_own=f1_grid[0][i],
                               f0_own=f0_grid[0][i])

                # Agent 1: own = j; slice P_full[:, j, :] (axes 0,2)
                slice_1 = [[P_full[ii][j][ll] for ll in range(G_full)]
                           for ii in range(G_full)]
                A0_1, A1_1 = agent_evidence_2d(
                    slice_1, p, u_full,
                    tau_a=tau_vec[0], tau_off=tau_vec[2],
                    f1_grid_a=f1_grid[0], f0_grid_a=f0_grid[0],
                    f1_grid_off=f1_grid[2], f0_grid_off=f0_grid[2],
                    sqrt_tt2pi=(sqrt_tt2pi[0], sqrt_tt2pi[2]),
                )
                mu1 = bayes_mp(u_full[j], tau_vec[1], A0_1, A1_1,
                               f1_own=f1_grid[1][j],
                               f0_own=f0_grid[1][j])

                # Agent 2: own = l; slice P_full[:, :, l] (axes 0,1)
                slice_2 = [[P_full[ii][jj][l] for jj in range(G_full)]
                           for ii in range(G_full)]
                A0_2, A1_2 = agent_evidence_2d(
                    slice_2, p, u_full,
                    tau_a=tau_vec[0], tau_off=tau_vec[1],
                    f1_grid_a=f1_grid[0], f0_grid_a=f0_grid[0],
                    f1_grid_off=f1_grid[1], f0_grid_off=f0_grid[1],
                    sqrt_tt2pi=(sqrt_tt2pi[0], sqrt_tt2pi[1]),
                )
                mu2 = bayes_mp(u_full[l], tau_vec[2], A0_2, A1_2,
                               f1_own=f1_grid[2][l],
                               f0_own=f0_grid[2][l])

                p_new = clear_crra_mp([mu0, mu1, mu2], gamma_vec, W_vec)
                P_new[i][j][l] = p_new

    return P_new


def residual_inf(P_a, P_b, inner_lo, inner_hi):
    """||P_a - P_b||_inf over the inner cube (using mpmath max)."""
    out = mp.mpf("0")
    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                d = P_a[i][j][l] - P_b[i][j][l]
                ad = abs(d)
                if ad > out:
                    out = ad
    return out


def init_no_learning_K3_mp(u_full, tau_vec, gamma_vec, W_vec, *,
                           sqrt_tt2pi=None):
    """No-learning halo seed: P[i,j,l] = clear with mu_k = lam(tau_k * u_k)."""
    G_full = len(u_full)
    if sqrt_tt2pi is None:
        sqrt_tt2pi = tuple(mp.sqrt(tau_vec[k] / (2 * mp.pi))
                           for k in range(3))
    P = [[[mp.mpf("0")] * G_full for _ in range(G_full)]
         for _ in range(G_full)]
    for i in range(G_full):
        m0 = lam_mp(tau_vec[0] * u_full[i])
        for j in range(G_full):
            m1 = lam_mp(tau_vec[1] * u_full[j])
            for l in range(G_full):
                m2 = lam_mp(tau_vec[2] * u_full[l])
                P[i][j][l] = clear_crra_mp([m0, m1, m2], gamma_vec,
                                           W_vec)
    return P


# -------------------------- helpers for I/O --------------------------

def f64_array_to_mp(P_3d_float, dps=None):
    """Convert nested list/np-array of float to nested list of mpf."""
    if dps is not None:
        mp.mp.dps = dps
    G = len(P_3d_float)
    out = [[[mp.mpf(repr(float(P_3d_float[i][j][l])))
             for l in range(G)] for j in range(G)] for i in range(G)]
    return out


def mp_array_to_strings(P_3d_mp):
    """Convert mp 3D nested list to nested list of decimal strings
    (preserving full precision)."""
    G = len(P_3d_mp)
    return [[[mp.nstr(P_3d_mp[i][j][l], mp.mp.dps, strip_zeros=False)
              for l in range(G)] for j in range(G)] for i in range(G)]


def strings_to_mp(P_3d_strings, dps=None):
    if dps is not None:
        mp.mp.dps = dps
    G = len(P_3d_strings)
    return [[[mp.mpf(P_3d_strings[i][j][l])
              for l in range(G)] for j in range(G)] for i in range(G)]
