"""Generate Fig 3A (CARA multicontour), Fig 3B (CRRA multicontour),
and Fig 11 (mechanisms bar chart).

Fig 3: at u_1=1, τ=2 — extract level set { (u_2, u_3) : P(1, u_2, u_3) = p }
       at 7 price levels evenly spaced in logit (covering [0.2, 0.8]).
       (A) CARA no-learning prices.
       (B) CRRA REE prices using G=15 strict μ*.

Fig 11: 6 bars showing 1-R² at no-learning under different mechanisms:
       Baseline CARA, Het α (CARA), CRRA γ=0.5, Het γ, Het τ, K=4.
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)
BLACK = "black"


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


def f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u + 0.5)**2)


def f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u - 0.5)**2)


# =========================================================================
# Fig 3A: CARA multicontour
# =========================================================================
print("=" * 50)
print("Fig 3A: CARA multicontour (no-learning)")
print("=" * 50, flush=True)

# CARA: agents use μ_k = Λ(τu_k). Market clears: Σ x_k = 0.
# CARA demand: x = (logit(μ) - logit(p))/γ. Sum=0 → logit(p)=avg(logit(μ_k))
# = avg(τu_k) = T*/K = T*/3.
# So P_CARA(u_1, u_2, u_3) = Λ(T*/3).
u1_fixed = 1.0
P_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
u_test = np.linspace(-3.5, 3.5, 80)

CARA_contours = {}
for p_target in P_LEVELS:
    # logit(p) = T*/3 → T* = 3 logit(p) → u_2 + u_3 = 3 logit(p)/τ - u_1
    T_target = 3 * np.log(p_target / (1 - p_target))
    u2u3_sum = T_target / TAU - u1_fixed
    pts = []
    for u2 in u_test:
        u3 = u2u3_sum - u2
        if -3.5 <= u3 <= 3.5:
            pts.append((float(u2), float(u3)))
    CARA_contours[f"{p_target:g}"] = pts
    print(f"  p={p_target}: {len(pts)} points (line: u_2 + u_3 = {u2u3_sum:.3f})",
          flush=True)

# Save
with open(f"{RESULTS_DIR}/fig_multicontour_A_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_A",
                "params": {"u1": u1_fixed, "tau": TAU, "model": "CARA_no_learning"},
                "p_levels": P_LEVELS,
                "contours": CARA_contours}, f, indent=2)
with open(f"{RESULTS_DIR}/fig_multicontour_A_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        pts = CARA_contours[f"{p:g}"]
        f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
print("  Saved fig_multicontour_A_*.{json,tex}", flush=True)

# Preview
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
for k, p in enumerate(P_LEVELS):
    color = cmap(k / max(len(P_LEVELS) - 1, 1))
    pts = CARA_contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.5, label=f"$p={p}$")
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Fig 3A: CARA contour lines (no learning)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$",
              fontsize=10)
ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_A_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_A_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved preview", flush=True)


# =========================================================================
# Fig 3B: CRRA multicontour at REE μ*, γ=0.5
# =========================================================================
print("\n" + "=" * 50)
print("Fig 3B: CRRA multicontour at REE")
print("=" * 50, flush=True)

GAMMA = 0.5
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]


def mu_at(u, p):
    if u <= u_grid_R[0]: idx = 0; w = 0.0
    elif u >= u_grid_R[-1]: idx = len(u_grid_R) - 1; w = 1.0
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


def market_clear_REE(u1, u2, u3):
    def Z(p):
        return sum(crra_demand_vec(np.array([mu_at(u, p)]), np.array([p]),
                                      GAMMA)[0]
                   for u in (u1, u2, u3))
    try:
        return brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        return None


# Compute P(u1=1, u2, u3) on a grid
print("  Computing P(1, u2, u3)...", flush=True)
u_test_3B = np.linspace(-3.5, 3.5, 50)
P_slice = np.full((len(u_test_3B), len(u_test_3B)), np.nan)
for j, u2 in enumerate(u_test_3B):
    for k, u3 in enumerate(u_test_3B):
        p = market_clear_REE(u1_fixed, u2, u3)
        if p is not None:
            P_slice[j, k] = p
print(f"  Computed slice ({(~np.isnan(P_slice)).sum()}/{P_slice.size} valid)",
      flush=True)

# Extract contours via matplotlib
fig_tmp, ax_tmp = plt.subplots()
cs = ax_tmp.contour(u_test_3B, u_test_3B, P_slice.T, levels=P_LEVELS)
plt.close(fig_tmp)

CRRA_contours = {}
for k, p in enumerate(P_LEVELS):
    pts = []
    if k < len(cs.allsegs):
        for c in cs.allsegs[k]:
            pts.extend([(float(x), float(y)) for x, y in c])
    CRRA_contours[f"{p:g}"] = pts
    print(f"  p={p}: {len(pts)} points", flush=True)

with open(f"{RESULTS_DIR}/fig_multicontour_B_data.json", "w") as f:
    json.dump({"figure": "fig_multicontour_B",
                "params": {"u1": u1_fixed, "tau": TAU, "gamma": GAMMA,
                           "model": "CRRA_REE", "G": 15},
                "p_levels": P_LEVELS,
                "contours": CRRA_contours}, f, indent=2)
with open(f"{RESULTS_DIR}/fig_multicontour_B_pgfplots.tex", "w") as f:
    for p in P_LEVELS:
        f.write(f"% p={p}\n")
        pts = CRRA_contours[f"{p:g}"]
        f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n\n")
print("  Saved fig_multicontour_B_*.{json,tex}", flush=True)

# Preview
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
cmap = plt.cm.viridis
for ki, p in enumerate(P_LEVELS):
    color = cmap(ki / max(len(P_LEVELS) - 1, 1))
    pts = CRRA_contours[f"{p:g}"]
    if pts:
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.5, label=f"$p={p}$")
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Fig 3B: CRRA contour lines (REE)\n"
              f"$u_1={u1_fixed}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, $G=15$",
              fontsize=10)
ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_multicontour_B_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved preview", flush=True)


# =========================================================================
# Fig 11: mechanisms bar chart (no-learning 1-R² for various configs)
# =========================================================================
print("\n" + "=" * 50)
print("Fig 11: mechanisms bar chart")
print("=" * 50, flush=True)

# Configs (all at τ=2 unless noted, K=3 unless noted)
def no_learning_R2(tau, gammas, taus=None, K=3, G_eval=15):
    """Compute no-learning 1-R² with possibly heterogeneous (γ_k, τ_k)."""
    if taus is None:
        taus = [tau] * K
    if not isinstance(gammas, list):
        gammas = [gammas] * K
    u_eff_caps = [min(4.0, 6.0 / max(t, 1e-6)) for t in taus]
    # Use same grid for all agents
    G_eff = G_eval
    grids = [np.linspace(-cap, cap, G_eff) for cap in u_eff_caps]

    Y, X, W = [], [], []
    # Use first agent's tau as reference for T*
    tau_ref = taus[0]

    def gen_indices():
        for i in range(G_eff):
            for j in range(G_eff):
                for k in range(G_eff):
                    yield i, j, k

    for i, j, kk in gen_indices():
        u_vec = [grids[0][i], grids[1][j], grids[2][kk]] if K == 3 else None
        if K != 3:
            continue
        u1, u2, u3 = u_vec
        mus = [Lam(taus[m] * u_vec[m]) for m in range(K)]
        def Z(p):
            return sum(crra_demand_vec(np.array([mus[m]]),
                                          np.array([p]),
                                          gammas[m])[0]
                       for m in range(K))
        try:
            p_star = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
        except ValueError:
            continue
        T = sum(taus[m] * u_vec[m] for m in range(K))
        # Joint density (mixture under v=0,1)
        f1_prod = 1.0; f0_prod = 1.0
        for m in range(K):
            f1_prod *= f1(u_vec[m], taus[m])
            f0_prod *= f0(u_vec[m], taus[m])
        w = 0.5 * (f1_prod + f0_prod)
        Y.append(np.log(p_star / (1 - p_star))); X.append(T); W.append(float(w))
    Y = np.array(Y); X = np.array(X); W = np.array(W)
    if W.sum() == 0:
        return float("nan")
    W = W / W.sum()
    Yb = (W * Y).sum(); Xb = (W * X).sum()
    cov = (W * (Y - Yb) * (X - Xb)).sum()
    vy = (W * (Y - Yb)**2).sum()
    vx = (W * (X - Xb)**2).sum()
    R2 = cov**2 / (vy * vx) if vy * vx > 0 else 0.0
    return 1.0 - R2


configs = [
    ("CARA, sym",     {"tau": 2.0, "gammas": [50.0, 50.0, 50.0]}),
    ("CARA het α",    {"tau": 2.0, "gammas": [50.0, 50.0, 50.0]}),  # α heterogeneity not modelled here; use same as CARA-sym
    ("CRRA γ=0.5",    {"tau": 2.0, "gammas": [0.5, 0.5, 0.5]}),
    ("Het γ (5,3,1)", {"tau": 2.0, "gammas": [5.0, 3.0, 1.0]}),
    ("Het τ (4,2,1)", {"tau": 2.0, "gammas": [0.5, 0.5, 0.5],
                         "taus": [4.0, 2.0, 1.0]}),
    ("CRRA γ=2",      {"tau": 2.0, "gammas": [2.0, 2.0, 2.0]}),
]

bars = []
for label, cfg in configs:
    print(f"  {label}: ", end="", flush=True)
    if "taus" in cfg:
        r2 = no_learning_R2(cfg["tau"], cfg["gammas"], taus=cfg["taus"])
    else:
        r2 = no_learning_R2(cfg["tau"], cfg["gammas"])
    print(f"1-R²={r2:.4e}", flush=True)
    bars.append({"label": label, "1-R2": float(r2)})

with open(f"{RESULTS_DIR}/fig11_mechanisms_data.json", "w") as f:
    json.dump({"figure": "fig11_mechanisms",
                "params": {"K": 3, "G": 15},
                "configs": bars}, f, indent=2)
with open(f"{RESULTS_DIR}/fig11_mechanisms_pgfplots.tex", "w") as f:
    f.write("% mechanisms bar chart\n")
    for b in bars:
        # Replace special chars in label for pgfplots
        f.write(f"% {b['label']}: 1-R²={b['1-R2']:.4e}\n")
print("  Saved fig11_mechanisms_*.{json,tex}", flush=True)

# Preview
fig, ax = plt.subplots(figsize=(10/2.54, 7/2.54))
labels = [b["label"] for b in bars]
vals = [b["1-R2"] for b in bars]
colors = [BLUE, GREEN, RED, RED, RED, RED]
ax.bar(range(len(labels)), vals, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
ax.set_ylabel(r"$1-R^2$ (no learning)")
ax.set_title("Fig 11: mechanisms bar chart\n(no-learning $1-R^2$, $K=3$, $G=15$)",
              fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig11_mechanisms_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig11_mechanisms_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved preview", flush=True)

print("\n=== DONE: Fig 3A, 3B, 11 ===")
