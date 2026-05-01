"""Redo Fig 1: knife-edge no-learning at γ=[0.5, 1, 4], τ ∈ [0.1, 100]."""
import json
import warnings
import numpy as np
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec

warnings.filterwarnings("ignore", category=RuntimeWarning)

UMAX = 4.0
G = 15
RESULTS_DIR = "results/full_ree"


def f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u + 0.5)**2)


def f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u - 0.5)**2)


def no_learning_R2(tau, gamma, G_eval=15):
    # Cap u·τ to avoid μ-saturation at large τ
    u_eff_cap = min(UMAX, 6.0 / max(tau, 1e-6))
    u_grid = np.linspace(-u_eff_cap, u_eff_cap, G_eval)
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


GAMMAS = [0.5, 1.0, 4.0]
TAUS = np.exp(np.linspace(np.log(0.1), np.log(100.0), 40))
curves = []
for gamma in GAMMAS:
    print(f"γ = {gamma}", flush=True)
    points = []
    for tau in TAUS:
        r2, slope = no_learning_R2(tau, gamma, G_eval=G)
        points.append({"tau": float(tau), "1-R2": float(r2),
                        "slope": float(slope)})
        print(f"  τ={tau:6.3f}: 1-R²={r2:.4e}", flush=True)
    curves.append({"gamma": gamma, "points": points})

with open(f"{RESULTS_DIR}/fig_knife_edge_data.json", "w") as f:
    json.dump({"figure": "fig_knife_edge",
                "params": {"G": G, "tau_range": [0.1, 100.0],
                           "gammas": GAMMAS},
                "curves": curves}, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_knife_edge_pgfplots.tex", "w") as f:
    for c in curves:
        f.write(f"% gamma={c['gamma']}\n")
        pts = [(p["tau"], p["1-R2"]) for p in c["points"]]
        f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
print("Saved fig_knife_edge_*.{json,tex}", flush=True)

# Render preview
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
GREEN = (0.11, 0.35, 0.02); RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42)
for c, color, style in zip(curves, [GREEN, RED, BLUE], ["-", "--", ":"]):
    ts = [p["tau"] for p in c["points"]]
    rs = [p["1-R2"] for p in c["points"]]
    ax.plot(ts, rs, color=color, ls=style, lw=2,
              label=f"$\\gamma={c['gamma']}$")
ax.set_xscale("log")
ax.set_xlabel("$\\tau$"); ax.set_ylabel("$1-R^2$")
ax.set_title(f"Fig 1: knife-edge (no-learning), G={G}", fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_knife_edge_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_knife_edge_preview.png", dpi=150,
              bbox_inches="tight")
plt.close(fig)
print("Saved previews", flush=True)
