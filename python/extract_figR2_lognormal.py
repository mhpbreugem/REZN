#!/usr/bin/env python3
"""Fig R2: 2-state lognormal payoff, no-learning weighted 1-R^2 vs tau.
mpmath dps=100, tol 1e-50.

Asset: v in {v_L, v_H} = {exp(-sigma/2), exp(+sigma/2)}, sigma=1, prior 0.5.
Signal: u_k = log(v) + eps/sqrt(tau), so mu_k = Lambda(tau u_k).

CRRA FOC for theta:
    mu (vH-p)(W + th(vH-p))^{-gamma} + (1-mu)(vL-p)(W + th(vL-p))^{-gamma} = 0
Move the (vL-p) term to the right (it is negative when p>vL):
    mu (vH-p)(W + th(vH-p))^{-gamma} = (1-mu)(p-vL)(W + th(vL-p))^{-gamma}
=>  ((W + th(vL-p)) / (W + th(vH-p)))^gamma = (1-mu)(p-vL) / (mu(vH-p))
Closed form: with C = (1-mu)(p-vL) / (mu(vH-p)) > 0 and X = C^(1/gamma),
    theta = W (X - 1) / [(vL - p) - X (vH - p)]
Reservation (no-trade) price: theta=0 iff X=1 iff p* = (1-mu) vL + mu vH.
This eliminates the inner root-finder (only outer bisection on p remains).

Regression target zeta(p) = log((p-vL)/(vH-p)). Under CARA this is exactly
T*/K, so weighted 1-R^2 of zeta on T* generalises the binary {0,1} logit case.

Grid: G=20 from u in [-5, +5]; tau in 16 log-spaced points from 0.1 to 10.
gamma in {0.5, 1.0, 4.0}.

Outputs:
    results/full_ree/figR2_G20_pgfplots.tex     (filename kept for paper)
    results/full_ree/figR2_G20_lognormal.json   (filename kept for paper)
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mpmath as mp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _mp_helpers import (
    bisect_market_clear,
    fmt_mp,
    lam,
    EPS_PRICE,
    TOL_BISECT,
)

mp.mp.dps = 100

N_WORKERS = 4

OUT_TEX = Path("results/full_ree/figR2_G20_pgfplots.tex")
OUT_JSON = Path("results/full_ree/figR2_G20_lognormal.json")

ZERO = mp.mpf(0)
ONE = mp.mpf(1)
HALF = mp.mpf("0.5")

SIGMA = mp.mpf(1)
V_L = mp.exp(-SIGMA / 2)
V_H = mp.exp(SIGMA / 2)
W_END = mp.mpf(1)

GAMMAS = [mp.mpf("0.5"), mp.mpf(1), mp.mpf(4)]
N_TAU = 16
TAU_LO = mp.mpf("0.1")
TAU_HI = mp.mpf(10)

G = 20
UMAX = mp.mpf(5)


def make_u_grid():
    if G == 1:
        return [ZERO]
    step = (2 * UMAX) / (G - 1)
    return [-UMAX + step * i for i in range(G)]


def signal_density(u, v_state, tau):
    """f(u | v=state) for state in {0,1} -> mean shift -1/2, +1/2; precision tau."""
    mean = HALF if v_state == 1 else -HALF
    d = u - mean
    return mp.sqrt(tau / (2 * mp.pi)) * mp.exp(-tau * d * d / 2)


def crra_demand_lognormal(mu, p, gamma, vL, vH, W):
    """Closed-form theta for 2-state asset with payoff in {vL, vH}, prob {1-mu, mu}."""
    if p <= vL:
        return mp.mpf("1e30")
    if p >= vH:
        return mp.mpf("-1e30")
    if mu <= 0:
        # Sure low payoff: short the asset, capped by wealth bound
        return -W / (vH - p) * (1 - mp.mpf("1e-30"))
    if mu >= 1:
        # Sure high payoff: buy, capped by wealth bound
        return W / (p - vL) * (1 - mp.mpf("1e-30"))
    C = ((ONE - mu) * (p - vL)) / (mu * (vH - p))
    X = C ** (ONE / gamma)
    denom = (vL - p) - X * (vH - p)
    if denom == 0:
        return mp.mpf("1e30")
    return W * (X - ONE) / denom


def clear_NL(u_triple, tau, gamma):
    """No-learning: each agent uses mu_k = Lambda(tau u_k); solve sum theta_k = 0 in p."""
    mus = [lam(tau * u) for u in u_triple]
    eps = (V_H - V_L) * mp.mpf("1e-40")
    a = V_L + eps
    b = V_H - eps

    def excess(p):
        return sum(crra_demand_lognormal(m, p, gamma, V_L, V_H, W_END) for m in mus)

    fa = excess(a)
    fb = excess(b)
    if fa <= 0:
        return a
    if fb >= 0:
        return b
    for _ in range(2000):
        c = (a + b) / 2
        fc = excess(c)
        if abs(fc) < TOL_BISECT:
            return c
        if fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc
        if (b - a) < TOL_BISECT * (V_H - V_L):
            return (a + b) / 2
    return (a + b) / 2


def compute_one(gamma, tau, u_grid):
    """Weighted (1-R^2) of zeta(p) on T* under no-learning, lognormal asset."""
    f1 = [signal_density(u, 1, tau) for u in u_grid]
    f0 = [signal_density(u, 0, tau) for u in u_grid]

    Wsum = ZERO
    sum_T = ZERO
    sum_Z = ZERO
    sum_TT = ZERO
    sum_ZZ = ZERO
    sum_TZ = ZERO
    n_pts = 0
    half = HALF

    Gn = len(u_grid)
    for i in range(Gn):
        for j in range(Gn):
            wij1 = f1[i] * f1[j]
            wij0 = f0[i] * f0[j]
            for k in range(Gn):
                w = half * (wij1 * f1[k] + wij0 * f0[k])
                if w < mp.mpf("1e-300"):
                    continue
                Tstar = tau * (u_grid[i] + u_grid[j] + u_grid[k])
                p = clear_NL((u_grid[i], u_grid[j], u_grid[k]), tau, gamma)
                if p <= V_L or p >= V_H:
                    continue
                zeta = mp.log((p - V_L) / (V_H - p))
                Wsum += w
                sum_T += w * Tstar
                sum_Z += w * zeta
                sum_TT += w * Tstar * Tstar
                sum_ZZ += w * zeta * zeta
                sum_TZ += w * Tstar * zeta
                n_pts += 1

    if Wsum == 0:
        return mp.nan
    T_mean = sum_T / Wsum
    Z_mean = sum_Z / Wsum
    var_T = sum_TT / Wsum - T_mean * T_mean
    var_Z = sum_ZZ / Wsum - Z_mean * Z_mean
    cov_TZ = sum_TZ / Wsum - T_mean * Z_mean
    if var_T <= 0 or var_Z <= 0:
        return mp.nan
    R2 = (cov_TZ * cov_TZ) / (var_T * var_Z)
    return ONE - R2


def _worker(args):
    """Top-level worker for ProcessPoolExecutor. Reconfigures mpmath dps in child."""
    gamma_str, tau_str = args
    mp.mp.dps = 100
    gamma = mp.mpf(gamma_str)
    tau = mp.mpf(tau_str)
    u_grid = make_u_grid()
    val = compute_one(gamma, tau, u_grid)
    return gamma_str, tau_str, fmt_mp(val, 30)


def main():
    print(f"Lognormal payoff (mpmath dps={mp.mp.dps}, tol 1e-50, workers={N_WORKERS}):")
    print(f"  v in {{{fmt_mp(V_L,16)}, {fmt_mp(V_H,16)}}},  W = {fmt_mp(W_END,4)}")
    print(f"  G={G}, UMAX={UMAX}, K=3")
    print()

    u_grid = make_u_grid()
    print(f"u_grid = [{', '.join(fmt_mp(u,8) for u in u_grid[:3])} ... "
          f"{', '.join(fmt_mp(u,8) for u in u_grid[-3:])}]")
    print()

    # Log-spaced taus
    taus = [TAU_LO * (TAU_HI / TAU_LO) ** (mp.mpf(k) / (N_TAU - 1)) for k in range(N_TAU)]

    results = {
        "sigma": float(SIGMA),
        "v_L": fmt_mp(V_L, 30),
        "v_H": fmt_mp(V_H, 30),
        "W": float(W_END),
        "G": G,
        "UMAX": float(UMAX),
        "K": 3,
        "dps": mp.mp.dps,
        "tol": "1e-50",
        "n_workers": N_WORKERS,
        "curves": [],
    }

    # Submit all (gamma, tau) tasks; workers receive string-encoded mpf so they
    # rebuild at full precision in their own process.
    tasks = []
    for gamma in GAMMAS:
        for tau in taus:
            tasks.append((fmt_mp(gamma, 30), fmt_mp(tau, 30)))
    print(f"Submitting {len(tasks)} tasks to {N_WORKERS} workers...")

    by_pair = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        n_done = 0
        for fut in as_completed(futures):
            gamma_s, tau_s, val_s = fut.result()
            by_pair[(gamma_s, tau_s)] = val_s
            n_done += 1
            elapsed = time.time() - t_start
            print(f"  [{n_done:2d}/{len(tasks)}] gamma={gamma_s[:8]:<8} "
                  f"tau={tau_s[:14]:<14}  ->  1-R^2 = {val_s[:24]}   "
                  f"(elapsed {elapsed:.0f}s)", flush=True)

    # Reassemble in submission order so the JSON / TeX preserves the curve order
    for gamma in GAMMAS:
        gamma_s = fmt_mp(gamma, 30)
        pts = []
        for tau in taus:
            tau_s = fmt_mp(tau, 30)
            pts.append({"tau": tau_s, "1-R2": by_pair[(gamma_s, tau_s)]})
        results["curves"].append({"gamma": gamma_s, "points": pts})

    OUT_JSON.write_text(json.dumps(results, indent=2))

    lines = [
        f"% Fig R2 (lognormal payoff): no-learning weighted 1-R^2 vs tau",
        f"% mpmath dps={mp.mp.dps}, market-clearing tol 1e-50",
        f"% v in {{{fmt_mp(V_L,18)}, {fmt_mp(V_H,18)}}} (2-state lognormal, sigma=1), W={fmt_mp(W_END,4)}",
        f"% Regression target zeta(p) = log((p-vL)/(vH-p)) on T* = tau*(u1+u2+u3)",
        f"% Weights w = 1/2 (prod f_1 + prod f_0)",
        f"% G={G}, UMAX={fmt_mp(UMAX,4)}, K=3, gammas in {{{', '.join(fmt_mp(g,8) for g in GAMMAS)}}}",
        "",
    ]
    for curve in results["curves"]:
        gamma_s = curve["gamma"]
        coords = "".join(
            f"({pt['tau']},{pt['1-R2']})" for pt in curve["points"]
        )
        lines.append(f"% gamma = {gamma_s}")
        lines.append(f"\\addplot coordinates {{{coords}}};")
        lines.append("")

    coords0 = "".join(f"({pt['tau']},0)" for pt in results["curves"][0]["points"])
    lines.append(f"% CARA reference (identically zero)")
    lines.append(f"\\addplot coordinates {{{coords0}}};")
    lines.append("")

    OUT_TEX.write_text("\n".join(lines))
    print(f"\nWrote {OUT_TEX}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
