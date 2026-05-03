#!/usr/bin/env python3
"""Fig R2: knife-edge with lognormal payoff, no-learning benchmark.

Two-state lognormal asset:
    v in {v_L, v_H} = {exp(-sigma/2), exp(+sigma/2)}, sigma = 1, prior 0.5
Signal: u_k = log(v) + eps_k/sqrt(tau), so likelihood ratio = exp(tau u),
posterior mu_k = Lambda(tau u_k) (same form as binary {0,1}).

No-learning benchmark: each agent uses private prior mu_k only.
Market clearing: sum_k x_k(mu_k, p; gamma, W) = 0 in p in (v_L, v_H).

Regression target (CARA-linear analogue of logit): zeta(p) = log((p-v_L)/(v_H-p)).
Under CARA with this asset, zeta(p) = T*/K exactly. Hence weighted 1-R^2 of
zeta(p) on T* measures the deviation from the CARA-FR benchmark, in direct
analogy with the logit(p)-on-T* regression for the binary {0,1} case.

Saves: results/full_ree/figR2_G20_pgfplots.tex
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path("results/full_ree/figR2_G20_pgfplots.tex")
OUT_JSON = Path("results/full_ree/figR2_G20_lognormal.json")

SIGMA = 1.0
V_L = math.exp(-SIGMA / 2.0)
V_H = math.exp(+SIGMA / 2.0)
W_END = 1.0  # endowment (wealth before trade)

GAMMAS = [0.5, 1.0, 4.0]
TAUS = list(np.geomspace(0.1, 10.0, 16))  # 16 log-spaced tau values

G = 20
UMAX = 5.0
u_grid = np.linspace(-UMAX, UMAX, G)


def lam(z):
    return 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))


def crra_demand_2state(mu, p, gamma, vL, vH, W):
    """Solve for theta s.t. mu(vH-p)(W+th(vH-p))^-g + (1-mu)(vL-p)(W+th(vL-p))^-g = 0.

    Bracket: theta_min = -(W - 1e-12)/(vH - p) (so W + th(vH-p) > 0)
             theta_max = (W - 1e-12)/(p - vL) (so W + th(vL-p) > 0)
    """
    if p <= vL + 1e-14:
        return 1e9  # huge buy demand
    if p >= vH - 1e-14:
        return -1e9  # huge sell demand

    th_min = -(W - 1e-9) / (vH - p) + 1e-12
    th_max = (W - 1e-9) / (p - vL) - 1e-12

    def foc(theta):
        wH = W + theta * (vH - p)
        wL = W + theta * (vL - p)
        if wH <= 0 or wL <= 0:
            return float("nan")
        # FOC: mu(vH-p) wH^-g + (1-mu)(vL-p) wL^-g
        return mu * (vH - p) * wH ** (-gamma) + (1.0 - mu) * (vL - p) * wL ** (-gamma)

    fa = foc(th_min)
    fb = foc(th_max)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        # Fall back to small bracket
        th_min = -W * 0.99 / (vH - p)
        th_max = W * 0.99 / (p - vL)
        fa = foc(th_min)
        fb = foc(th_max)
    if fa * fb > 0:
        return th_min if abs(fa) < abs(fb) else th_max
    return brentq(foc, th_min, th_max, xtol=1e-12, rtol=1e-12)


def clear_NL(u_triple, tau, gamma, vL, vH, W):
    """No-learning market clearing: sum_k x_k(mu_k, p) = 0, mu_k = Lambda(tau u_k)."""
    mu = [lam(tau * u) for u in u_triple]

    def excess(p):
        return sum(crra_demand_2state(mu[k], p, gamma, vL, vH, W) for k in range(3))

    eps = 1e-10
    a = vL + eps * (vH - vL)
    b = vH - eps * (vH - vL)
    fa = excess(a)
    fb = excess(b)
    if fa <= 0:
        return a
    if fb >= 0:
        return b
    return brentq(excess, a, b, xtol=1e-12, rtol=1e-12)


def signal_density(u, v_state, tau):
    """f(u | v=v_state) for state in {0, 1} -> mean shift -1/2, +1/2."""
    mean = 0.5 if v_state == 1 else -0.5
    return math.sqrt(tau / (2 * math.pi)) * math.exp(-0.5 * tau * (u - mean) ** 2)


def compute_one(gamma, tau):
    """Return weighted (1-R^2) for the no-learning regression at (gamma, tau)."""
    vL, vH = V_L, V_H
    Tstars = []
    zetas = []
    weights = []

    f1 = np.array([signal_density(u, 1, tau) for u in u_grid])
    f0 = np.array([signal_density(u, 0, tau) for u in u_grid])

    for i in range(G):
        for j in range(G):
            for l in range(G):
                u = (u_grid[i], u_grid[j], u_grid[l])
                Tstar = tau * (u[0] + u[1] + u[2])
                w = 0.5 * (f1[i] * f1[j] * f1[l] + f0[i] * f0[j] * f0[l])
                if w < 1e-300:
                    continue
                p = clear_NL(u, tau, gamma, vL, vH, W_END)
                # zeta(p) = log((p-vL)/(vH-p))
                if p <= vL or p >= vH:
                    continue
                zeta = math.log((p - vL) / (vH - p))
                Tstars.append(Tstar)
                zetas.append(zeta)
                weights.append(w)

    Ts = np.array(Tstars)
    Zs = np.array(zetas)
    Ws = np.array(weights)
    Wsum = Ws.sum()
    if Wsum <= 0:
        return float("nan")
    T_mean = (Ws * Ts).sum() / Wsum
    Z_mean = (Ws * Zs).sum() / Wsum
    var_T = (Ws * (Ts - T_mean) ** 2).sum() / Wsum
    var_Z = (Ws * (Zs - Z_mean) ** 2).sum() / Wsum
    cov = (Ws * (Ts - T_mean) * (Zs - Z_mean)).sum() / Wsum
    if var_T <= 0 or var_Z <= 0:
        return float("nan")
    R2 = (cov * cov) / (var_T * var_Z)
    return float(1.0 - R2)


def main():
    print(f"Lognormal payoff: v in {{{V_L:.4f}, {V_H:.4f}}}, sigma={SIGMA}")
    print(f"Endowment W={W_END}, K=3, G={G}, UMAX={UMAX}")
    print(f"Signal model: u_k | state ~ N(state-0.5, 1/tau)")
    print()

    results = {"sigma": SIGMA, "v_L": V_L, "v_H": V_H, "W": W_END,
               "G": G, "UMAX": UMAX, "K": 3, "curves": []}

    for gamma in GAMMAS:
        pts = []
        print(f"gamma = {gamma}:")
        for tau in TAUS:
            one_minus_R2 = compute_one(gamma, tau)
            pts.append({"tau": float(tau), "1-R2": one_minus_R2})
            print(f"  tau = {tau:7.3f}  ->  1 - R^2 = {one_minus_R2:.5f}")
        results["curves"].append({"gamma": gamma, "points": pts})

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Build pgfplots
    lines = [
        f"% Fig R2 (lognormal payoff): no-learning weighted 1-R^2 vs tau",
        f"% v in {{{V_L:.4f}, {V_H:.4f}}} (2-state lognormal, sigma={SIGMA}), W={W_END}",
        f"% Regression target zeta(p) = log((p-vL)/(vH-p)) on T* = tau*(u1+u2+u3)",
        f"% Weights w = 1/2 (prod f_1 + prod f_0) (ex-ante signal density)",
        f"% G={G} UMAX={UMAX} K=3, gammas in {{{', '.join(str(g) for g in GAMMAS)}}}",
        "",
    ]
    for curve in results["curves"]:
        gamma = curve["gamma"]
        coords = "".join(f"({pt['tau']:.4f},{pt['1-R2']:.6f})" for pt in curve["points"])
        lines.append(f"% gamma = {gamma}")
        lines.append(f"\\addplot coordinates {{{coords}}};")
        lines.append("")

    # CARA reference line: identically zero
    coords0 = "".join(f"({tau:.4f},0.000000)" for tau in TAUS)
    lines.append(f"% CARA reference (identically zero)")
    lines.append(f"\\addplot coordinates {{{coords0}}};")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"\nWrote {OUT}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
