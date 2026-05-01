"""Redo Fig 6B (old style: high u, low u, price) with high sampling rate."""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0; GAMMA = 0.5
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)

ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]


def mu_at(u, p):
    if u <= u_grid_R[0]: idx = 0
    elif u >= u_grid_R[-1]: idx = len(u_grid_R) - 1
    else:
        ia = np.searchsorted(u_grid_R, u); ib = ia - 1
        w = (u - u_grid_R[ib]) / (u_grid_R[ia] - u_grid_R[ib])
        p_b = np.clip(p, p_grid_R[ib, 0], p_grid_R[ib, -1])
        m_b = np.interp(p_b, p_grid_R[ib, :], mu_REE[ib, :])
        p_a = np.clip(p, p_grid_R[ia, 0], p_grid_R[ia, -1])
        m_a = np.interp(p_a, p_grid_R[ia, :], mu_REE[ia, :])
        return (1 - w) * m_b + w * m_a
    p_c = np.clip(p, p_grid_R[idx, 0], p_grid_R[idx, -1])
    return float(np.interp(p_c, p_grid_R[idx, :], mu_REE[idx, :]))


# High-resolution sampling: 200 T* points
T_RANGE = np.linspace(-12, 12, 200)
fig6B = {"agent_high": [], "agent_low": [], "price": []}
for T_target in T_RANGE:
    u_high = +1.0; u_low = -1.0
    u3 = T_target / TAU - u_high - u_low
    if abs(u3) > 4.0:
        continue
    def Z(p):
        return (
            crra_demand_vec(np.array([mu_at(u_high, p)]), np.array([p]),
                              GAMMA)[0]
            + crra_demand_vec(np.array([mu_at(u_low, p)]), np.array([p]),
                                GAMMA)[0]
            + crra_demand_vec(np.array([mu_at(u3, p)]), np.array([p]),
                                GAMMA)[0])
    try:
        p_REE = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        continue
    mu_high = mu_at(u_high, p_REE); mu_low = mu_at(u_low, p_REE)
    fig6B["agent_high"].append({"T_star": float(T_target),
                                  "mu": float(mu_high)})
    fig6B["agent_low"].append({"T_star": float(T_target),
                                 "mu": float(mu_low)})
    fig6B["price"].append({"T_star": float(T_target),
                             "p": float(p_REE)})

fig6B["params"] = {"G": 15, "tau": TAU, "gamma": GAMMA,
                    "u_high": 1.0, "u_low": -1.0,
                    "n_points": len(T_RANGE)}
with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_data.json", "w") as f:
    json.dump(fig6B, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_pgfplots.tex", "w") as f:
    f.write("% mu_high (u=+1)\n")
    pts = [(p["T_star"], p["mu"]) for p in fig6B["agent_high"]]
    f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
    f.write("% mu_low (u=-1)\n")
    pts = [(p["T_star"], p["mu"]) for p in fig6B["agent_low"]]
    f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
    f.write("% price\n")
    pts = [(p["T_star"], p["p"]) for p in fig6B["price"]]
    f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n")
print("Saved fig_posteriors_CRRA_*.{json,tex}", flush=True)

# Render
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
Ts = [p["T_star"] for p in fig6B["agent_high"]]
mu_high = [p["mu"] for p in fig6B["agent_high"]]
mu_low = [p["mu"] for p in fig6B["agent_low"]]
Tsp = [p["T_star"] for p in fig6B["price"]]
prices = [p["p"] for p in fig6B["price"]]
ax.plot(Ts, mu_high, color=RED, lw=2, label=f"$\\mu_1$ ($u=+1$)")
ax.plot(Ts, mu_low, color=GREEN, lw=2, label=f"$\\mu_2$ ($u=-1$)")
ax.plot(Tsp, prices, color=BLUE, ls="--", lw=2, label="price $p$")
ax.set_xlabel("$T^*$"); ax.set_ylabel("$\\mu$, $p$")
ax.set_xlim(min(Ts), max(Ts)); ax.set_ylim(0, 1)
ax.set_title(f"Fig 6B: CRRA posteriors (REE)\n"
              f"$G=15$, $\\tau={TAU}$, $\\gamma={GAMMA}$, $n={len(Ts)}$",
              fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CRRA_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CRRA_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved preview ({len(Ts)} points)", flush=True)
