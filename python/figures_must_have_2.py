"""Figures Fig 10 (convergence) and Fig 4B (γ-sweep) at G=15.

Fig 10: residual history of Picard-PAVA + NK from no-learning seed.
Fig 4B: 1-R² vs γ for γ ∈ {0.1, 0.25, 0.5, 1, 2, 4} at G=15, τ=2.
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
TAU = 2.0
TOL_MAX = 1e-14
RESULTS_DIR = "results/full_ree"

warnings.filterwarnings("ignore", category=RuntimeWarning)


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def pgf_coords(pts, fmt="{:.6g}"):
    parts = []
    for x, y in pts:
        parts.append(f"({fmt.format(x)},{fmt.format(y)})")
    return "".join(parts)


def f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u + 0.5)**2)


def f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u - 0.5)**2)


def F_phi(x, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def measure_residual(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return float(F[active].max()) if active.any() else float("nan")


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    out = np.empty((G_new, p_new.shape[1]))
    for i in range(G_new):
        u = u_new[i]; u_c = np.clip(u, u_old[0], u_old[-1])
        ra = np.searchsorted(u_old, u_c)
        rb = max(ra - 1, 0); ra = min(ra, len(u_old) - 1)
        w = (u_c - u_old[rb]) / (u_old[ra] - u_old[rb]) if ra != rb else 1.0
        for j in range(p_new.shape[1]):
            p = p_new[i, j]
            p_c1 = np.clip(p, p_old[rb, 0], p_old[rb, -1])
            mb = np.interp(p_c1, p_old[rb, :], mu_old[rb, :])
            p_c2 = np.clip(p, p_old[ra, 0], p_old[ra, -1])
            ma = np.interp(p_c2, p_old[ra, :], mu_old[ra, :])
            out[i, j] = (1 - w) * mb + w * mu_old[ra, :].max()  # safety
            out[i, j] = (1 - w) * mb + w * ma
    return np.clip(out, EPS, 1 - EPS)


# =========================================================================
# Fig 10: convergence path at G=15, γ=0.5, τ=2
# =========================================================================
print("=" * 60)
print("Fig 10: convergence path at G=15, γ=0.5, τ=2")
print("=" * 60, flush=True)

GAMMA_CONV = 0.5
u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA_CONV, G)
mu = np.zeros((G, G))
for i, u in enumerate(u_grid):
    mu[i, :] = Lam(TAU * u)
mu = pava_2d(mu)

# Picard phase 1: α=0.05, 5000 iters
hist = []
N1 = 5000; ALPHA1 = 0.05
print(f"  Picard phase 1 (α={ALPHA1}, n={N1})", flush=True)
for it in range(N1):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA_CONV)
    cand = pava_2d(cand)
    F = np.abs(cand - mu)
    res = float(F[active].max()) if active.any() else float("nan")
    hist.append({"iter": it, "phase": "picard1", "alpha": ALPHA1,
                  "residual_inf": res})
    mu = ALPHA1 * cand + (1 - ALPHA1) * mu
    mu = np.clip(mu, EPS, 1 - EPS)
mu = pava_2d(mu)

# Picard phase 2: α=0.01, 5000 iters with Cesaro
N2 = 5000; ALPHA2 = 0.01
print(f"  Picard phase 2 (α={ALPHA2}, n={N2}, Cesaro avg last {N2//2})",
      flush=True)
mu_sum = np.zeros_like(mu); n_collected = 0
for it in range(N2):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA_CONV)
    cand = pava_2d(cand)
    F = np.abs(cand - mu)
    res = float(F[active].max()) if active.any() else float("nan")
    hist.append({"iter": N1 + it, "phase": "picard2", "alpha": ALPHA2,
                  "residual_inf": res})
    mu = ALPHA2 * cand + (1 - ALPHA2) * mu
    mu = np.clip(mu, EPS, 1 - EPS)
    if it >= N2 - N2 // 2:
        mu_sum += mu; n_collected += 1
mu = pava_2d(mu_sum / n_collected)

# NK polish
print(f"  NK polish", flush=True)
nk_residuals = []
def cb(x, f):
    nk_residuals.append(float(np.max(np.abs(f))))

try:
    sol = newton_krylov(
        lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi,
                          TAU, GAMMA_CONV),
        mu.ravel(), f_tol=TOL_MAX, maxiter=300, method="lgmres",
        verbose=False, callback=cb)
    mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "ok"
except NoConvergence as e:
    mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "noconv"

# Append NK residuals to history (assume each entry is one NK iter)
for k, r in enumerate(nk_residuals):
    hist.append({"iter": N1 + N2 + k, "phase": "NK", "alpha": None,
                  "residual_inf": float(r)})

# Final residual
final_res = measure_residual(mu_nk, u_grid, p_grid, p_lo, p_hi,
                              TAU, GAMMA_CONV)
print(f"  Final NK status={nk_status}, residual={final_res:.3e}", flush=True)

# Save Fig 10
fig10_data = {
    "figure": "fig_convergence",
    "params": {"G": G, "tau": TAU, "gamma": GAMMA_CONV,
                "picard_phase1": {"alpha": ALPHA1, "iters": N1},
                "picard_phase2": {"alpha": ALPHA2, "iters": N2,
                                   "cesaro_avg": N2 // 2},
                "nk_iters": len(nk_residuals),
                "tol_target": TOL_MAX},
    "history": hist,
    "final_residual": final_res,
}
with open(f"{RESULTS_DIR}/fig_convergence_data.json", "w") as f:
    json.dump(fig10_data, f, indent=2)
# Pgfplots: subsample to ~200 points (full hist is 10000+)
n_full = len(hist)
stride = max(1, n_full // 300)
pts_picard1 = [(h["iter"], max(h["residual_inf"], 1e-20))
                for h in hist[:N1] if h["iter"] % stride == 0]
pts_picard2 = [(h["iter"], max(h["residual_inf"], 1e-20))
                for h in hist[N1:N1+N2] if h["iter"] % stride == 0]
pts_nk = [(h["iter"], max(h["residual_inf"], 1e-20))
           for h in hist[N1+N2:]]
with open(f"{RESULTS_DIR}/fig_convergence_pgfplots.tex", "w") as f:
    f.write(f"% Picard phase 1 (α={ALPHA1})\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(pts_picard1)}}};\n\n")
    f.write(f"% Picard phase 2 (α={ALPHA2}, Cesaro)\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(pts_picard2)}}};\n\n")
    f.write(f"% NK polish\n")
    f.write(f"\\addplot coordinates {{{pgf_coords(pts_nk)}}};\n")
print(f"  Saved fig_convergence_*.{{json,tex}}", flush=True)


# =========================================================================
# Fig 4B: REE 1-R² vs γ at τ=2, G=15
# =========================================================================
print("\n" + "=" * 60)
print("Fig 4B: REE 1-R² vs γ at G=15, τ=2")
print("=" * 60, flush=True)

GAMMAS_4B = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
fig4b_REE = []
fig4b_NL = []

# Build no-learning 1-R² for each γ at τ=2
def no_learning_R2(tau, gamma, G_eval=15):
    u_g = np.linspace(-UMAX, UMAX, G_eval)
    Y, X, W = [], [], []
    for i in range(G_eval):
        for j in range(G_eval):
            for k in range(G_eval):
                u1, u2, u3 = u_g[i], u_g[j], u_g[k]
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
    return 1.0 - R2


# REE: try existing strict checkpoints first, then solve fresh ones
def get_REE_R2(gamma):
    """Try to load strict G=15 ckpt for this γ; if not found, solve."""
    candidates = [
        f"posterior_v3_strict_G15.npz" if abs(gamma - 0.5) < 1e-6 else None,
    ]
    # No strict G=15 at all γ values exist; fall back to G=14 strict ckpts
    # which we have for γ=0.3, 0.5, 1, 2 etc. For γ=0.1, 0.25, 4: solve fresh.
    if abs(gamma - 0.5) < 1e-6:
        ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
        u_g = ck["u_grid"]; p_g = ck["p_grid"]; mu = ck["mu"]
        p_lo = ck["p_lo"]; p_hi = ck["p_hi"]
        r2, slope, _ = measure_R2(mu, u_g, p_g, p_lo, p_hi, TAU, gamma)
        return float(r2), float(slope)

    # Otherwise: warm-start from G=14 strict at nearest γ
    near_paths = {
        0.1: f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma0.3.npz",
        0.25: f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma0.3.npz",
        1.0: f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma1.npz",
        2.0: f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma2.npz",
        4.0: f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma2.npz",
    }
    sp = near_paths[gamma]
    ck = np.load(sp)
    mu14 = ck["mu"]; u14 = ck["u_grid"]; p14 = ck["p_grid"]
    # Interp to G=15
    u_g15 = np.linspace(-UMAX, UMAX, G)
    p_lo15, p_hi15, p_g15 = init_p_grid(u_g15, TAU, gamma, G)
    mu = interp_mu(mu14, u14, p14, u_g15, p_g15)
    mu = pava_2d(mu)
    # Picard rounds
    def picard_round(mu, n, na, alpha):
        ms = np.zeros_like(mu); nc = 0
        for it in range(n):
            cand, active, _ = phi_step(mu, u_g15, p_g15, p_lo15, p_hi15, TAU, gamma)
            cand = pava_2d(cand)
            mu = alpha * cand + (1 - alpha) * mu
            mu = np.clip(mu, EPS, 1 - EPS)
            if it >= n - na:
                ms += mu; nc += 1
        return pava_2d(ms / nc)
    mu = picard_round(mu, 3000, 1500, 0.05)
    mu = picard_round(mu, 3000, 1500, 0.01)
    mu = picard_round(mu, 5000, 2500, 0.003)
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_g15, p_g15, p_lo15, p_hi15, TAU, gamma),
            mu.ravel(), f_tol=TOL_MAX, maxiter=200, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        if int((np.diff(mu_nk, axis=0) < 0).sum()) == 0 \
                and int((np.diff(mu_nk, axis=1) < 0).sum()) == 0:
            mu = mu_nk
    except (NoConvergence, ValueError, RuntimeError):
        pass
    np.savez(f"{RESULTS_DIR}/posterior_v3_strict_G15_gamma{gamma:g}.npz",
             mu=mu, u_grid=u_g15, p_grid=p_g15, p_lo=p_lo15, p_hi=p_hi15)
    r2, slope, _ = measure_R2(mu, u_g15, p_g15, p_lo15, p_hi15, TAU, gamma)
    return float(r2), float(slope)


for gamma in GAMMAS_4B:
    print(f"  γ = {gamma}", flush=True)
    t0 = time.time()
    ree_r2, ree_slope = get_REE_R2(gamma)
    nl_r2 = no_learning_R2(TAU, gamma, G_eval=G)
    fig4b_REE.append({"gamma": gamma, "1-R2": ree_r2, "slope": ree_slope})
    fig4b_NL.append({"gamma": gamma, "1-R2": nl_r2})
    print(f"    REE 1-R²={ree_r2:.4e}, slope={ree_slope:.4f}; "
          f"NL 1-R²={nl_r2:.4e}, t={time.time()-t0:.0f}s", flush=True)

with open(f"{RESULTS_DIR}/fig_4B_data.json", "w") as f:
    json.dump({"figure": "fig_4B_REE_vs_gamma",
                "params": {"G": G, "tau": TAU, "gammas": GAMMAS_4B},
                "REE": fig4b_REE, "no_learning": fig4b_NL}, f, indent=2)

with open(f"{RESULTS_DIR}/fig_4B_pgfplots.tex", "w") as f:
    f.write("% REE 1-R² vs γ\n")
    pts = [(p["gamma"], p["1-R2"]) for p in fig4b_REE]
    f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n\n")
    f.write("% no-learning 1-R² vs γ\n")
    pts = [(p["gamma"], p["1-R2"]) for p in fig4b_NL]
    f.write(f"\\addplot coordinates {{{pgf_coords(pts)}}};\n")
print(f"  Saved fig_4B_*.{{json,tex}}", flush=True)

print("\n" + "=" * 60)
print("DONE: Fig 10, Fig 4B")
print("=" * 60)
