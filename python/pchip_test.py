"""PCHIP (monotone cubic Hermite) contour experiment.

This is a one-shot precision test, not a production solver. The goal is to
re-solve a KNOWN converged (τ, γ) configuration with a C¹-smooth row/column
interpolant (PCHIP) in place of the piecewise-linear one used by
rezn_het._contour_sum. If the PhiI floor actually comes from kinks at grid
vertices, replacing the interpolant with a monotone cubic should let
Picard drive PhiI several orders of magnitude lower, while NOT introducing
overshoot of a non-monotone cubic spline (classical pitfall).

Strategy:
  row(u_b) → PchipInterpolator(u, row)
  crossings of row(u_b) = p_obs → for each segment [u[k], u[k+1]] in which
    (row[k]-p_obs) and (row[k+1]-p_obs) have opposite signs, use brentq on
    the PCHIP interpolant to find the off-grid root.

Same for column sweeps. Everything else is unchanged from rezn_het.

Slow — scipy + Python loops, no numba.
"""
from __future__ import annotations
import sys
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import rezn_het as rh

G_DEFAULT = 9
UMAX_DEFAULT = 2.0
EPS = 1e-12


# ---------------- PCHIP contour sum -----------------------------------

def _contour_sum_pchip(slice_, u, tau_A, tau_B, p_obs):
    """Analogue of rezn_het._contour_sum using PCHIP along each row/column."""
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0

    # Pass A: sweep row
    for a in range(G):
        row = slice_[a]
        ua = u[a]
        pc = PchipInterpolator(u, row)
        d = row - p_obs
        for k in range(G - 1):
            if d[k] * d[k + 1] < 0.0:
                try:
                    ub = brentq(lambda x: pc(x) - p_obs, u[k], u[k + 1],
                                xtol=1e-12, rtol=1e-14, maxiter=100)
                except Exception:
                    continue
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif d[k] == 0.0 and d[k + 1] != 0.0:
                ub = u[k]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)
            elif k == G - 2 and d[k + 1] == 0.0:
                ub = u[k + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub, tau_B)

    # Pass B: sweep column
    for b in range(G):
        col = slice_[:, b]
        ub_grid = u[b]
        pc = PchipInterpolator(u, col)
        d = col - p_obs
        for k in range(G - 1):
            if d[k] * d[k + 1] < 0.0:
                try:
                    ua = brentq(lambda x: pc(x) - p_obs, u[k], u[k + 1],
                                xtol=1e-12, rtol=1e-14, maxiter=100)
                except Exception:
                    continue
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)
            elif d[k] == 0.0 and d[k + 1] != 0.0:
                ua = u[k]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)
            elif k == G - 2 and d[k + 1] == 0.0:
                ua = u[k + 1]
                A0 += rh._f0(ua, tau_A) * rh._f0(ub_grid, tau_B)
                A1 += rh._f1(ua, tau_A) * rh._f1(ub_grid, tau_B)

    return 0.5 * A0, 0.5 * A1


def _agent_posterior_pchip(ag, i, j, l, p_obs, Pg, u, taus):
    G = u.shape[0]
    taus = np.asarray(taus, dtype=float)
    if ag == 0:
        u_own = u[i]; tau_own = taus[0]
        tau_A = taus[1]; tau_B = taus[2]
        slice_ = Pg[i, :, :].copy()
    elif ag == 1:
        u_own = u[j]; tau_own = taus[1]
        tau_A = taus[0]; tau_B = taus[2]
        slice_ = Pg[:, j, :].copy()
    else:
        u_own = u[l]; tau_own = taus[2]
        tau_A = taus[0]; tau_B = taus[1]
        slice_ = Pg[:, :, l].copy()
    A0, A1 = _contour_sum_pchip(slice_, u, tau_A, tau_B, p_obs)
    g0 = rh._f0(u_own, tau_own); g1 = rh._f1(u_own, tau_own)
    den = g0 * A0 + g1 * A1
    if den <= 0:
        return 1.0 / (1.0 + np.exp(-tau_own * u_own))
    return g1 * A1 / den


def _posteriors_at_pchip(i, j, l, p_obs, Pg, u, taus):
    return (
        _agent_posterior_pchip(0, i, j, l, p_obs, Pg, u, taus),
        _agent_posterior_pchip(1, i, j, l, p_obs, Pg, u, taus),
        _agent_posterior_pchip(2, i, j, l, p_obs, Pg, u, taus),
    )


# ---------------- Phi, residual, Picard with PCHIP -------------------

def _demand_crra(mu, p, gamma, W=1.0):
    mu_c = float(np.clip(mu, EPS, 1 - EPS))
    p_c  = float(np.clip(p,  EPS, 1 - EPS))
    R = np.exp((np.log(mu_c/(1-mu_c)) - np.log(p_c/(1-p_c))) / gamma)
    return W * (R - 1.0) / ((1.0 - p_c) + R * p_c)


def _clearing_residual(mus, p, gammas, Ws):
    return sum(_demand_crra(m, p, g, w) for m, g, w in zip(mus, gammas, Ws))


def _clear_price(mus, gammas, Ws):
    def f(p): return _clearing_residual(mus, p, gammas, Ws)
    return brentq(f, 1e-9, 1 - 1e-9, xtol=1e-13, rtol=1e-15, maxiter=200)


def phi_map_pchip(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    Pnew = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_cur = float(Pg[i, j, l])
                mus = _posteriors_at_pchip(i, j, l, p_cur, Pg, u, taus)
                Pnew[i, j, l] = _clear_price(list(mus), list(gammas), list(Ws))
    return Pnew


def residual_array_pchip(Pg, u, taus, gammas, Ws):
    G = u.shape[0]
    F = np.empty_like(Pg)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = float(Pg[i, j, l])
                mus = _posteriors_at_pchip(i, j, l, p, Pg, u, taus)
                F[i, j, l] = _clearing_residual(list(mus), p, list(gammas), list(Ws))
    return F


def solve_picard_pchip(G, taus, gammas, umax=UMAX_DEFAULT, Ws=(1, 1, 1),
                        P_init=None, maxiters=3000, abstol=1e-14, alpha=1.0):
    u = np.linspace(-umax, umax, G)
    taus = np.asarray(taus, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    Ws = np.asarray(Ws, dtype=float)
    if P_init is None:
        # cold start from no-learning via existing rezn_het (linear is fine for seed)
        P0 = rh._nolearning_price(u, taus, gammas, Ws)
    else:
        P0 = np.clip(P_init, 1e-9, 1 - 1e-9).copy()
    Pcur = P0.copy()
    history = []
    for it in range(maxiters):
        Pnew = phi_map_pchip(Pcur, u, taus, gammas, Ws)
        diff = float(np.abs(Pnew - Pcur).max())
        Pcur = alpha * Pnew + (1 - alpha) * Pcur
        history.append(diff)
        if diff < abstol:
            break
    F = residual_array_pchip(Pcur, u, taus, gammas, Ws)
    return dict(P_star=Pcur, P0=P0, u=u, residual=F,
                history=history,
                converged=bool(history and history[-1] < abstol),
                taus=taus, gammas=gammas)


# ---------------- Experiment -----------------------------------------

def experiment(taus, gammas, umax=2.0, maxiters=3000):
    """Solve one config with PCHIP and print convergence history."""
    print(f"\n=== PCHIP Picard on τ={taus}, γ={gammas} ===")
    sys.stdout.flush()

    # seed from rezn_het's cold start (fast)
    u = np.linspace(-umax, umax, G_DEFAULT)
    taus_a = np.asarray(taus, dtype=float)
    gammas_a = np.asarray(gammas, dtype=float)
    Ws_a = np.asarray((1, 1, 1), dtype=float)
    P_seed = rh._nolearning_price(u, taus_a, gammas_a, Ws_a)

    t0 = time.time()
    res = solve_picard_pchip(G_DEFAULT, taus, gammas, umax=umax,
                              P_init=P_seed, maxiters=maxiters,
                              abstol=1e-14, alpha=1.0)
    dt = time.time() - t0

    PhiI = res["history"][-1] if res["history"] else float("nan")
    Finf = float(np.abs(res["residual"]).max())
    print(f"  iters   : {len(res['history'])}")
    print(f"  time    : {dt:.1f} s")
    print(f"  ‖Φ-I‖∞  : {PhiI:.3e}")
    print(f"  ‖F‖∞    : {Finf:.3e}")
    # show last few history entries to see the asymptote
    print(f"  last 5  : {['%.2e' % x for x in res['history'][-5:]]}")
    sys.stdout.flush()
    return res


if __name__ == "__main__":
    # Known configs from MainRun1:
    known = [
        ((3, 3, 3), (50, 50, 50)),     # lowest-PhiI leader
        ((3, 3, 3), (3, 3, 3)),        # homo middle
        ((3, 3, 3), (0.3, 50, 50)),    # PR leader (66%)
    ]
    for t, g in known:
        experiment(t, g, maxiters=2000)
