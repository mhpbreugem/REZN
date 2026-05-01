"""Figure data generation: must-haves at G=15.

This script produces JSON + pgfplots .tex output for each figure.
Strict G=15 strict checkpoints used as the canonical REE answer.

Figures generated:
  Fig 1: knife-edge no-learning at γ=[0.25, 1, 4]
  Fig 6A: analytical CARA posterior (no compute)
  Fig 5: REE vs NL price function at G=15, γ=0.5, τ=2
  Fig 6B: posteriors fan-out at G=15, γ=0.5, τ=2
  Fig 10: convergence path at G=15, γ=0.5, τ=2
"""
import json
import os
import time
import warnings
import numpy as np
from scipy.optimize import brentq, newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, crra_demand_vec, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 15
RESULTS_DIR = "results/full_ree"

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs(RESULTS_DIR, exist_ok=True)


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def pgf_coords(pts, fmt="{:.6g}"):
    """Convert list of (x, y) tuples to pgfplots coordinates string."""
    parts = []
    for x, y in pts:
        parts.append(f"({fmt.format(x)},{fmt.format(y)})")
    return "".join(parts)


# =========================================================================
# Fig 1: knife-edge no-learning 1-R² vs τ at γ=[0.25, 1, 4]
# =========================================================================
print("=" * 60)
print("Fig 1: knife-edge no-learning")
print("=" * 60, flush=True)


def f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u + 0.5)**2)


def f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u - 0.5)**2)


def no_learning_R2(tau, gamma, G_eval=15):
    """Compute no-learning 1-R² at given (tau, gamma)."""
    u_grid = np.linspace(-UMAX, UMAX, G_eval)
    Y, X, W = [], [], []
    for i in range(G_eval):
        for j in range(G_eval):
            for k in range(G_eval):
                u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]
                mus = [Lam(tau * u) for u in (u1, u2, u3)]
                def Z(p):
                    return sum(crra_demand_vec(np.array([m]),
                                                  np.array([p]), gamma)[0]
                               for m in mus)
                try:
                    p_star = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
                except ValueError:
                    continue
                T = tau * (u1 + u2 + u3)
                w = 0.5 * (f1(u1, tau) * f1(u2, tau) * f1(u3, tau)
                           + f0(u1, tau) * f0(u2, tau) * f0(u3, tau))
                Y.append(np.log(p_star / (1 - p_star)))
                X.append(T)
                W.append(float(w))
    Y = np.array(Y); X = np.array(X); W = np.array(W)
    W = W / W.sum()
    Yb = (W * Y).sum(); Xb = (W * X).sum()
    cov = (W * (Y - Yb) * (X - Xb)).sum()
    vy = (W * (Y - Yb)**2).sum()
    vx = (W * (X - Xb)**2).sum()
    R2 = cov**2 / (vy * vx) if vy * vx > 0 else 0.0
    slope = cov / vx if vx > 0 else 0.0
    return 1.0 - R2, slope


GAMMAS_PAPER = [0.25, 1.0, 4.0]
TAUS_FIG1 = np.exp(np.linspace(np.log(0.1), np.log(20.0), 30))
fig1_curves = []
for gamma in GAMMAS_PAPER:
    print(f"  γ = {gamma}", flush=True)
    points = []
    for tau in TAUS_FIG1:
        r2, slope = no_learning_R2(tau, gamma, G_eval=15)
        points.append({"tau": float(tau), "1-R2": float(r2),
                        "slope": float(slope)})
    fig1_curves.append({"gamma": gamma, "points": points})

with open(f"{RESULTS_DIR}/fig_knife_edge_data.json", "w") as f:
    json.dump({"figure": "fig_knife_edge",
                "params": {"G": 15, "tau_range": [0.1, 20.0],
                           "gammas": GAMMAS_PAPER},
                "curves": fig1_curves}, f, indent=2)

with open(f"{RESULTS_DIR}/fig_knife_edge_pgfplots.tex", "w") as f:
    for c in fig1_curves:
        f.write(f"% gamma={c['gamma']}\n")
        pts = [(p["tau"], p["1-R2"]) for p in c["points"]]
        f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n\n")
print(f"  Saved {RESULTS_DIR}/fig_knife_edge_*.{{json,tex}}", flush=True)


# =========================================================================
# Fig 6A: analytical CARA posterior μ = Λ(T*/3)
# =========================================================================
print("\n" + "=" * 60)
print("Fig 6A: CARA posterior (analytical)")
print("=" * 60, flush=True)
T_range = np.linspace(-15, 15, 200)
mu_CARA = Lam(T_range / 3.0)
points = list(zip(T_range.tolist(), mu_CARA.tolist()))
with open(f"{RESULTS_DIR}/fig_posteriors_CARA_data.json", "w") as f:
    json.dump({"figure": "fig_posteriors_CARA",
                "formula": "mu = Lambda(T*/3)",
                "points": [{"T_star": x, "mu": y} for x, y in points]},
              f, indent=2)
with open(f"{RESULTS_DIR}/fig_posteriors_CARA_pgfplots.tex", "w") as f:
    f.write("% CARA: mu = Lambda(T*/3)\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(points)}}};\n")
print(f"  Saved fig_posteriors_CARA_*.{{json,tex}}", flush=True)


# =========================================================================
# Fig 5: REE vs NL price function at G=15, γ=0.5, τ=2
# =========================================================================
print("\n" + "=" * 60)
print("Fig 5: REE vs NL vs FR price function")
print("=" * 60, flush=True)
TAU = 2.0
GAMMA = 0.5
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]


def mu_at_uv(u, p):
    """Bivariate interpolation of μ at (u, p)."""
    if u <= u_grid_R[0]: idx = 0
    elif u >= u_grid_R[-1]: idx = len(u_grid_R) - 1
    else:
        ia = np.searchsorted(u_grid_R, u)
        ib = ia - 1
        wu = (u - u_grid_R[ib]) / (u_grid_R[ia] - u_grid_R[ib])
        p_b = np.clip(p, p_grid_R[ib, 0], p_grid_R[ib, -1])
        m_b = np.interp(p_b, p_grid_R[ib, :], mu_REE[ib, :])
        p_a = np.clip(p, p_grid_R[ia, 0], p_grid_R[ia, -1])
        m_a = np.interp(p_a, p_grid_R[ia, :], mu_REE[ia, :])
        return (1 - wu) * m_b + wu * m_a
    p_clamp = np.clip(p, p_grid_R[idx, 0], p_grid_R[idx, -1])
    return float(np.interp(p_clamp, p_grid_R[idx, :], mu_REE[idx, :]))


T_list, p_REE, p_NL, p_FR, w_list = [], [], [], [], []
for i in range(G):
    for j in range(G):
        for k in range(G):
            u1, u2, u3 = u_grid_R[i], u_grid_R[j], u_grid_R[k]
            mus_NL = [Lam(TAU * u) for u in (u1, u2, u3)]
            def Z_NL(p):
                return sum(crra_demand_vec(np.array([m]),
                                              np.array([p]), GAMMA)[0]
                           for m in mus_NL)
            try:
                p_nl = brentq(Z_NL, 1e-6, 1 - 1e-6, xtol=1e-12)
            except ValueError:
                continue
            def Z_REE(p):
                return sum(
                    crra_demand_vec(np.array([mu_at_uv(u, p)]),
                                      np.array([p]), GAMMA)[0]
                    for u in (u1, u2, u3))
            try:
                p_ree = brentq(Z_REE, 1e-6, 1 - 1e-6, xtol=1e-12)
            except ValueError:
                continue
            T = TAU * (u1 + u2 + u3)
            T_list.append(T); p_REE.append(p_ree); p_NL.append(p_nl)
            p_FR.append(Lam(T))
            w = 0.5 * (f1(u1, TAU) * f1(u2, TAU) * f1(u3, TAU)
                       + f0(u1, TAU) * f0(u2, TAU) * f0(u3, TAU))
            w_list.append(float(w))

T_arr = np.array(T_list); p_REE = np.array(p_REE); p_NL = np.array(p_NL)
p_FR = np.array(p_FR); w_arr = np.array(w_list)

# Bin by T*
n_bins = 40
T_bins = np.linspace(-10, 10, n_bins + 1)
T_centers, p_REE_b, p_NL_b, p_FR_b = [], [], [], []
for ki in range(n_bins):
    mask = (T_arr >= T_bins[ki]) & (T_arr < T_bins[ki + 1])
    if mask.sum() < 1: continue
    ww = w_arr[mask]; wn = ww.sum()
    if wn == 0: continue
    T_centers.append(0.5 * (T_bins[ki] + T_bins[ki + 1]))
    p_REE_b.append(float((p_REE[mask] * ww).sum() / wn))
    p_NL_b.append(float((p_NL[mask] * ww).sum() / wn))
    p_FR_b.append(float((p_FR[mask] * ww).sum() / wn))

# Save Fig 5
fig5_data = {
    "figure": "fig_ree_vs_nolearning",
    "params": {"G": G, "tau": TAU, "gamma": GAMMA, "n_bins": n_bins},
    "REE": [{"T_star": x, "p": y} for x, y in zip(T_centers, p_REE_b)],
    "no_learning": [{"T_star": x, "p": y} for x, y in zip(T_centers, p_NL_b)],
    "FR": [{"T_star": x, "p": y} for x, y in zip(T_centers, p_FR_b)],
}
with open(f"{RESULTS_DIR}/fig_ree_vs_nolearning_data.json", "w") as f:
    json.dump(fig5_data, f, indent=2)
with open(f"{RESULTS_DIR}/fig_ree_vs_nolearning_pgfplots.tex", "w") as f:
    f.write("% REE\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(zip(T_centers, p_REE_b))}}};\n\n")
    f.write("% no-learning\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(zip(T_centers, p_NL_b))}}};\n\n")
    f.write("% FR\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(zip(T_centers, p_FR_b))}}};\n")
print(f"  Saved fig_ree_vs_nolearning_*.{{json,tex}}", flush=True)


# =========================================================================
# Fig 6B: posteriors fan-out at G=15, γ=0.5, τ=2
# =========================================================================
print("\n" + "=" * 60)
print("Fig 6B: posteriors fan-out (REE)")
print("=" * 60, flush=True)
# At each T*, pick representative configurations:
#  - low-spread: u1=u2=u3=T*/(3τ)
#  - high-spread: extreme +1 vs others
T_grid = np.linspace(-12, 12, 25)
fig6B = {"agent_high": [], "agent_low": [], "price": []}
for T_target in T_grid:
    # Symmetric config: all three signals = T_target/(3*τ)
    # If T*=2u, then u=T_target/3 each → 1 +1 = 1, 1 -1 = -1
    # We want a "high-low" split for fan-out — pick u_high=+1, u_low=-1 + u3
    u_high = +1.0
    u_low = -1.0
    u3 = T_target / TAU - u_high - u_low
    if u3 < -UMAX or u3 > UMAX:
        continue
    def Z_REE(p):
        return (
            crra_demand_vec(np.array([mu_at_uv(u_high, p)]),
                              np.array([p]), GAMMA)[0]
            + crra_demand_vec(np.array([mu_at_uv(u_low, p)]),
                                np.array([p]), GAMMA)[0]
            + crra_demand_vec(np.array([mu_at_uv(u3, p)]),
                                np.array([p]), GAMMA)[0])
    try:
        p_REE_T = brentq(Z_REE, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        continue
    mu_high = mu_at_uv(u_high, p_REE_T)
    mu_low = mu_at_uv(u_low, p_REE_T)
    fig6B["agent_high"].append({"T_star": float(T_target),
                                  "mu": float(mu_high)})
    fig6B["agent_low"].append({"T_star": float(T_target),
                                 "mu": float(mu_low)})
    fig6B["price"].append({"T_star": float(T_target),
                             "p": float(p_REE_T)})

fig6B["params"] = {"G": G, "tau": TAU, "gamma": GAMMA,
                    "u_high": 1.0, "u_low": -1.0}
with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_data.json", "w") as f:
    json.dump(fig6B, f, indent=2)
with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_pgfplots.tex", "w") as f:
    f.write("% mu_high (u=+1)\n")
    pts = [(p["T_star"], p["mu"]) for p in fig6B["agent_high"]]
    f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n\n")
    f.write("% mu_low (u=-1)\n")
    pts = [(p["T_star"], p["mu"]) for p in fig6B["agent_low"]]
    f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n\n")
    f.write("% price\n")
    pts = [(p["T_star"], p["p"]) for p in fig6B["price"]]
    f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n")
print(f"  Saved fig_posteriors_CRRA_*.{{json,tex}}", flush=True)


print("\n" + "=" * 60)
print("DONE: Fig 1, 5, 6A, 6B")
print("=" * 60)
