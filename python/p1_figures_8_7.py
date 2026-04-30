"""P1 Tasks 7 (Fig 3 contour) and 8 (Fig 5 convergence).

Task 8: Convergence figure data — log residual history of Picard+NK at G=14
Task 7: Contour figure data — at converged μ*, reconstruct P, find level set
"""
import json, time, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence, brentq

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, crra_demand_vec, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 14
TAU = 2.0
GAMMA = 0.5


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


# ===== Task 8: Convergence figure data =====
print("=== Task 8: Convergence figure data ===\n", flush=True)
print("Running Picard-PAVA + NK at G=14, γ=0.5, τ=2, logging ||Φ-I||∞ every iter\n",
      flush=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
mu = np.zeros((G, G))
for i, u in enumerate(u_grid):
    mu[i, :] = Lam(TAU * u)
mu = pava_2d(mu)

# 5000 Picard at α=0.05
hist_picard1 = []
N1 = 5000; ALPHA1 = 0.05
for it in range(N1):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    cand = pava_2d(cand)
    F = np.abs(cand - mu)
    res = float(F[active].max()) if active.any() else float("nan")
    hist_picard1.append(res)
    mu = ALPHA1 * cand + (1 - ALPHA1) * mu
    mu = np.clip(mu, EPS, 1 - EPS)
mu = pava_2d(mu)

# 5000 Picard at α=0.01 (with Cesaro average)
hist_picard2 = []
N2 = 5000; ALPHA2 = 0.01; N_AVG = 2500
mu_sum = np.zeros_like(mu); n_collected = 0
for it in range(N2):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    cand = pava_2d(cand)
    F = np.abs(cand - mu)
    res = float(F[active].max()) if active.any() else float("nan")
    hist_picard2.append(res)
    mu = ALPHA2 * cand + (1 - ALPHA2) * mu
    mu = np.clip(mu, EPS, 1 - EPS)
    if it >= N2 - N_AVG:
        mu_sum += mu; n_collected += 1
mu = pava_2d(mu_sum / n_collected)
cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
res_after_cesaro = float(np.abs(cand - mu)[active].max())
print(f"After Cesaro: max residual = {res_after_cesaro:.3e}", flush=True)

# NK polish, logging residuals
hist_nk = []
def callback(x, f):
    hist_nk.append(float(np.max(np.abs(f))))

def F_phi_residual(x_flat, shape):
    mu_ = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu_, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F = (cand - mu_); F[~active] = 0.0
    return F.ravel()

try:
    sol = newton_krylov(
        lambda x: F_phi_residual(x, mu.shape), mu.ravel(),
        f_tol=1e-14, maxiter=300, method="lgmres",
        verbose=False, callback=callback)
    mu_final = sol.reshape(mu.shape)
except NoConvergence as e:
    mu_final = e.args[0].reshape(mu.shape)

cand, active, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
res_final = float(np.abs(cand - mu_final)[active].max())
print(f"Final NK residual: {res_final:.3e}", flush=True)

# Save iteration history
conv_data = {
    "params": {"G": G, "tau": TAU, "gamma": GAMMA, "umax": UMAX},
    "picard_phase1": {
        "n_iter": N1, "alpha": ALPHA1,
        "residual_history": hist_picard1,
    },
    "picard_phase2": {
        "n_iter": N2, "alpha": ALPHA2, "cesaro_avg_last": N_AVG,
        "residual_history": hist_picard2,
        "residual_after_cesaro": res_after_cesaro,
    },
    "nk_polish": {
        "residual_history": hist_nk,
        "final_residual": res_final,
    },
}
with open("results/full_ree/fig5_convergence_data.json", "w") as f:
    json.dump(conv_data, f, indent=2)
print(f"Saved: results/full_ree/fig5_convergence_data.json (NK iters: {len(hist_nk)})",
      flush=True)


# ===== Task 7: Contour figure data (Fig 3) =====
print("\n=== Task 7: Contour figure data (Fig 3) ===\n", flush=True)
# Use converged strict μ at G=14, γ=0.5, τ=2
ck = np.load("results/full_ree/posterior_v3_strict_G14_gamma0.5.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]
p_lo_R = ck["p_lo"]; p_hi_R = ck["p_hi"]


def mu_at_uv(u, p, mu_arr=mu_REE, u_g=u_grid_R, p_g=p_grid_R):
    """Bivariate interpolation of μ at (u, p)."""
    if u <= u_g[0]: idx = 0
    elif u >= u_g[-1]: idx = len(u_g) - 1
    else:
        i_above = np.searchsorted(u_g, u)
        i_below = i_above - 1
        wu = (u - u_g[i_below]) / (u_g[i_above] - u_g[i_below])
        m_b = np.interp(np.clip(p, p_g[i_below, 0], p_g[i_below, -1]),
                          p_g[i_below, :], mu_arr[i_below, :])
        m_a = np.interp(np.clip(p, p_g[i_above, 0], p_g[i_above, -1]),
                          p_g[i_above, :], mu_arr[i_above, :])
        return (1-wu)*m_b + wu*m_a
    p_clamp = np.clip(p, p_g[idx, 0], p_g[idx, -1])
    return float(np.interp(p_clamp, p_g[idx, :], mu_arr[idx, :]))


def market_clear_REE(u1, u2, u3):
    def Z(p):
        return sum(crra_demand_vec(np.array([mu_at_uv(u, p)]), np.array([p]),
                                     GAMMA)[0]
                   for u in (u1, u2, u3))
    try:
        return brentq(Z, 1e-6, 1-1e-6, xtol=1e-12)
    except ValueError:
        return None


# Reconstruct P(u1, u2, u3) on (G+1)^3 grid (use slight off-grid to avoid kinks)
u_test = np.linspace(-3.5, 3.5, 21)
P_REE = np.full((21, 21, 21), np.nan)
for i, u1 in enumerate(u_test):
    for j, u2 in enumerate(u_test):
        for k, u3 in enumerate(u_test):
            p = market_clear_REE(u1, u2, u3)
            if p is not None:
                P_REE[i, j, k] = p


# At fixed u1=1.0, find level set { (u2, u3) : P(1, u2, u3) = p_target }
u1_fixed = 1.0
p_target = market_clear_REE(u1_fixed, -1.0, 1.0)
print(f"At u1={u1_fixed}, level p_target = P(1,-1,1) = {p_target:.4f}", flush=True)

# Build slice P(1, u2, u3) and trace contour
slice_P = np.full((len(u_test), len(u_test)), np.nan)
for j, u2 in enumerate(u_test):
    for k, u3 in enumerate(u_test):
        p = market_clear_REE(u1_fixed, u2, u3)
        if p is not None:
            slice_P[j, k] = p

# Save contour data — list of (u2, u3) crossings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7))
cs = ax.contour(u_test, u_test, slice_P.T,
                  levels=[p_target], colors='red')
crrra_pts = []
for c in cs.allsegs[0]:
    crrra_pts.extend([(float(x), float(y)) for x, y in c])

# CARA contour: linear, T* = const (since Λ(T*) = p_target)
T_star = np.log(p_target / (1 - p_target))
# u2 + u3 = T_star/τ - u1
T_target = T_star / TAU - u1_fixed
cara_pts = []
for u2 in np.linspace(-3.5, 3.5, 50):
    u3 = T_target - u2
    if -3.5 <= u3 <= 3.5:
        cara_pts.append((float(u2), float(u3)))

# Save
contour_data = {
    "params": {"G": G, "tau": TAU, "gamma": GAMMA},
    "u1_fixed": u1_fixed,
    "p_target": float(p_target),
    "T_star_logit": float(T_star),
    "CRRA_contour_pts": crrra_pts,
    "CARA_contour_pts": cara_pts,
}
with open("results/full_ree/fig3_contour_data.json", "w") as f:
    json.dump(contour_data, f, indent=2)
print(f"Saved: results/full_ree/fig3_contour_data.json", flush=True)
print(f"  CRRA contour: {len(crrra_pts)} points")
print(f"  CARA contour: {len(cara_pts)} points")

# Make a quick plot
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 8))
if crrra_pts:
    arr = np.array(crrra_pts)
    ax.plot(arr[:, 0], arr[:, 1], color=(0.7, 0.11, 0.11), lw=2.5,
              label="CRRA REE level set")
if cara_pts:
    arr = np.array(cara_pts)
    ax.plot(arr[:, 0], arr[:, 1], color=(0, 0.20, 0.42), lw=2,
              ls="--", label="CARA / FR (straight, $u_2+u_3=$ const)")
ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.set_title(f"Level set $P(u_1, u_2, u_3) = p^*$ at $u_1={u1_fixed}$\n"
              f"$p^* = {p_target:.4f}$, $\\tau={TAU}$, $\\gamma={GAMMA}$, "
              f"$G={G}$")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.3); ax.set_aspect("equal")
plt.savefig("results/full_ree/fig3_contour_plot.png", dpi=150,
              bbox_inches="tight")
plt.savefig("results/full_ree/fig3_contour_plot.pdf",
              bbox_inches="tight")
print("Saved plots: results/full_ree/fig3_contour_plot.{png,pdf}")
print("\n=== Tasks 7 & 8 done ===")
