"""Redo Fig 6B: CRRA posteriors fan-out at G=15, γ=0.5, τ=2.

Shows multiple agents (different u_k) across a range of T*. Under FR
all curves collapse to μ=Λ(T*); under PR they fan out.

Construction: fix own-signal u₁, sweep partner-signals u₂, u₃ to span T*.
For each (u₁, u₂, u₃), solve REE market clearing and extract μ*(u₁, p*).
"""
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

# Load converged μ
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]


def mu_at(u, p):
    if u <= u_grid_R[0]: idx = 0; w = 0
    elif u >= u_grid_R[-1]: idx = len(u_grid_R) - 1; w = 1
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


U_OWN = [-2.0, -1.0, 0.0, 1.0, 2.0]
T_RANGE = np.linspace(-12, 12, 25)

curves = {}
prices = {}
for u1 in U_OWN:
    pts_mu = []; pts_p = []
    for T_target in T_RANGE:
        # Need u2 + u3 = T_target/τ - u1
        S = T_target / TAU - u1
        # Pick the symmetric split u2 = u3 = S/2
        u2 = S / 2.0; u3 = S / 2.0
        if abs(u2) > 4.0 or abs(u3) > 4.0:
            continue
        def Z(p):
            return (
                crra_demand_vec(np.array([mu_at(u1, p)]), np.array([p]),
                                  GAMMA)[0]
                + crra_demand_vec(np.array([mu_at(u2, p)]), np.array([p]),
                                    GAMMA)[0]
                + crra_demand_vec(np.array([mu_at(u3, p)]), np.array([p]),
                                    GAMMA)[0])
        try:
            p_star = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
        except ValueError:
            continue
        mu_own = mu_at(u1, p_star)
        pts_mu.append({"T_star": float(T_target), "mu": float(mu_own)})
        pts_p.append({"T_star": float(T_target), "p": float(p_star)})
    curves[u1] = pts_mu
    prices[u1] = pts_p

# Save Fig 6B
fig6B_data = {
    "figure": "fig_posteriors_CRRA",
    "params": {"G": 15, "tau": TAU, "gamma": GAMMA, "u_own": U_OWN},
    "curves": {f"u={u:+g}": pts for u, pts in curves.items()},
    "prices": {f"u={u:+g}": pts for u, pts in prices.items()},
}
with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_data.json", "w") as f:
    json.dump(fig6B_data, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_pgfplots.tex", "w") as f:
    for u in U_OWN:
        f.write(f"% mu_own at u={u:+g}\n")
        pts = [(p["T_star"], p["mu"]) for p in curves[u]]
        f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
    for u in U_OWN:
        f.write(f"% price at u_own={u:+g}\n")
        pts = [(p["T_star"], p["p"]) for p in prices[u]]
        f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
print("Saved fig_posteriors_CRRA_*.{json,tex}", flush=True)


# Render preview
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.coolwarm
for k, u in enumerate(U_OWN):
    color = cmap(k / max(len(U_OWN) - 1, 1))
    pts = curves[u]
    Ts = [p["T_star"] for p in pts]; ms = [p["mu"] for p in pts]
    ax.plot(Ts, ms, color=color, lw=2,
              label=f"$u={u:+g}$")

# Add p=Λ(T*) (FR benchmark) as dashed line
T_ref = np.linspace(-12, 12, 100)
ax.plot(T_ref, Lam(T_ref), "k:", lw=1.5, label="$\\mu=p=\\Lambda(T^*)$ (FR)")

# Add price curve (averaged over u_own — they all should match in REE clearing)
# Actually each u_own has a different p* (different triple).  Plot one for u=0:
pts_p0 = prices[0.0]
T_p = [p["T_star"] for p in pts_p0]
p_vals = [p["p"] for p in pts_p0]
ax.plot(T_p, p_vals, "g--", lw=1.5, alpha=0.5, label="$p^*$ (at $u_1=0$)")

ax.set_xlabel("$T^* = \\tau(u_1 + u_2 + u_3)$")
ax.set_ylabel("$\\mu_1$ (own posterior)")
ax.set_xlim(-12, 12); ax.set_ylim(0, 1)
ax.set_title(f"Fig 6B: CRRA posteriors fan-out (REE)\n"
              f"$G=15$, $\\tau={TAU}$, $\\gamma={GAMMA}$",
              fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CRRA_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CRRA_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved preview", flush=True)
