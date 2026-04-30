"""P0 SOLVER_TODO tasks.

1. Strict at paper γ values 0.25 and 4.0 (γ=1 already done)
2. CARA baseline at γ=50 (close to ∞)
3. Survival ratios: no-learning 1-R² at G=14, identical methodology
4. Posteriors table at (u₁,u₂,u₃) = (1,-1,1)
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
TOL_MAX = 1e-14


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        u_clamped = np.clip(u_target, u_old[0], u_old[-1])
        r_above = np.searchsorted(u_old, u_clamped)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        w = ((u_clamped - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
             if r_above != r_below else 1.0)
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_b, p_old[r_below, :], mu_old[r_below, :])
            p_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                      n_iter, n_avg, alpha):
    mu_sum = np.zeros_like(mu)
    n_collected = 0
    for it in range(n_iter):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n_iter - n_avg:
            mu_sum += mu
            n_collected += 1
    return pava_2d(mu_sum / n_collected)


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def F_phi_residual(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu)
    F[~active] = 0.0
    return F.ravel()


def nk_polish(mu_warm, u_grid, p_grid, p_lo, p_hi, tau, gamma,
              f_tol=TOL_MAX, maxiter=300):
    shape = mu_warm.shape
    try:
        sol = newton_krylov(
            lambda x: F_phi_residual(x, shape, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma),
            mu_warm.ravel(),
            f_tol=f_tol, maxiter=maxiter,
            method="lgmres", verbose=False,
        )
        return np.clip(sol.reshape(shape), EPS, 1 - EPS), "ok"
    except NoConvergence as e:
        mu_nk = (np.clip(e.args[0].reshape(shape), EPS, 1 - EPS)
                  if e.args else mu_warm)
        return mu_nk, "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        return mu_warm, f"err:{type(exc).__name__}"


def strict_solve(tau, gamma, mu_warm, u_warm, p_warm, label=""):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    if mu_warm is None:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(tau * u)
    else:
        mu = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    rounds = [(5000, 2500, 0.05), (5000, 2500, 0.01), (10000, 5000, 0.003)]
    t0 = time.time()
    last_med = float("inf")
    mu_picard_best = mu
    for r_idx, (n, na, a) in enumerate(rounds):
        mu = picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                                 n, na, a)
        d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        elapsed = time.time() - t0
        print(f"  [{label}] picard r{r_idx+1}: max={d['max']:.3e}, "
              f"med={d['med']:.3e}, u={d['u_viol']}, p={d['p_viol']}, "
              f"t={elapsed:.0f}s", flush=True)
        mu_picard_best = mu
        if d["max"] < TOL_MAX and d["u_viol"] == 0 and d["p_viol"] == 0:
            return mu, d, u_grid, p_grid, p_lo, p_hi, "strict_conv", elapsed
        if d["med"] > last_med * 0.5 and r_idx > 0:
            mu = mu + np.random.RandomState(42 + r_idx).normal(0, 1e-5, mu.shape)
            mu = np.clip(mu, EPS, 1 - EPS)
            mu = pava_2d(mu)
        last_med = d["med"]
    print(f"  [{label}] NK polish", flush=True)
    mu_nk, nk_status = nk_polish(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                                   f_tol=TOL_MAX, maxiter=300)
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    elapsed = time.time() - t0
    print(f"  [{label}] post-NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
          f"u={d_nk['u_viol']}, p={d_nk['p_viol']}, NK={nk_status}", flush=True)
    if (d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0
        and d_nk["p_viol"] == 0):
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_conv", elapsed
    if d_nk["u_viol"] > 0 or d_nk["p_viol"] > 0:
        return (mu_picard_best, measure(mu_picard_best, u_grid, p_grid,
                                          p_lo, p_hi, tau, gamma),
                u_grid, p_grid, p_lo, p_hi, "fallback_picard", elapsed)
    return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "no_strict", elapsed


def no_learning_R2(tau, gamma, u_grid):
    """Compute 1-R² for the no-learning equilibrium at the same G,
    using identical methodology to measure_R2."""
    G_u = len(u_grid)
    Y, X, W = [], [], []
    for i in range(G_u):
        for j in range(G_u):
            for k in range(G_u):
                u1, u2, u3 = u_grid[i], u_grid[j], u_grid[k]
                # No-learning: μ_k = Λ(τu_k), then market-clear
                mus = [Lam(tau * u) for u in (u1, u2, u3)]
                def Z(p):
                    return sum(crra_demand_vec(np.array([m]), np.array([p]), gamma)[0]
                               for m in mus)
                try:
                    p_star = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
                except ValueError:
                    continue
                T = tau * (u1 + u2 + u3)
                f1 = lambda u: np.sqrt(tau/(2*np.pi)) * np.exp(-0.5*tau*(u-0.5)**2)
                f0 = lambda u: np.sqrt(tau/(2*np.pi)) * np.exp(-0.5*tau*(u+0.5)**2)
                w = 0.5 * (f1(u1)*f1(u2)*f1(u3) + f0(u1)*f0(u2)*f0(u3))
                Y.append(np.log(p_star/(1-p_star))); X.append(T); W.append(float(w))
    Y, X, W = np.array(Y), np.array(X), np.array(W)
    W = W / W.sum()
    Yb = (W*Y).sum(); Xb = (W*X).sum()
    cov = (W*(Y-Yb)*(X-Xb)).sum()
    vy = (W*(Y-Yb)**2).sum()
    vx = (W*(X-Xb)**2).sum()
    R2 = cov**2/(vy*vx) if vy*vx > 0 else 0.0
    slope = cov/vx if vx > 0 else 0.0
    return 1.0 - R2, slope, len(Y)


warnings.filterwarnings("ignore", category=RuntimeWarning)
results = {"paper_gammas": [], "cara": None, "no_learning": [],
           "posteriors_table": None}

# ===== Task 1: Strict at γ=0.25 and γ=4 =====
print("=== Task 1: Paper γ values ===\n", flush=True)
# γ=0.25 warm from γ=0.3 strict
ck = np.load(f"results/full_ree/posterior_v3_strict_G14_gamma0.3.npz")
mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
print("--- γ=0.25 (warm from γ=0.3) ---", flush=True)
mu, d, ug, pg, plo, phi_, status, t = strict_solve(2.0, 0.25, mu_w, u_w, p_w,
                                                     label="γ=0.25")
r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, 0.25)
results["paper_gammas"].append({
    "gamma": 0.25, "tau": 2.0, "G": G, "max": d["max"], "med": d["med"],
    "u_viol": d["u_viol"], "p_viol": d["p_viol"],
    "1-R^2": float(r2), "slope": float(slope), "status": status,
})
print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}", flush=True)
if status == "strict_conv":
    np.savez("results/full_ree/posterior_v3_strict_G14_gamma0.25.npz",
             mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# γ=4.0 warm from γ=2 strict
ck = np.load(f"results/full_ree/posterior_v3_strict_G14_gamma2.npz")
mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
print("--- γ=4.0 (warm from γ=2) ---", flush=True)
mu, d, ug, pg, plo, phi_, status, t = strict_solve(2.0, 4.0, mu_w, u_w, p_w,
                                                     label="γ=4.0")
r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, 4.0)
results["paper_gammas"].append({
    "gamma": 4.0, "tau": 2.0, "G": G, "max": d["max"], "med": d["med"],
    "u_viol": d["u_viol"], "p_viol": d["p_viol"],
    "1-R^2": float(r2), "slope": float(slope), "status": status,
})
print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}", flush=True)
if status == "strict_conv":
    np.savez("results/full_ree/posterior_v3_strict_G14_gamma4.npz",
             mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# ===== Task 2: CARA baseline (γ=50) =====
print("\n=== Task 2: CARA baseline (γ=50) ===\n", flush=True)
ck = np.load(f"results/full_ree/posterior_v3_strict_G14_gamma2.npz")
mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
print("--- γ=50 (warm from γ=2) ---", flush=True)
mu, d, ug, pg, plo, phi_, status, t = strict_solve(2.0, 50.0, mu_w, u_w, p_w,
                                                     label="γ=50")
r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, 50.0)
results["cara"] = {
    "gamma": 50.0, "tau": 2.0, "G": G, "max": d["max"], "med": d["med"],
    "u_viol": d["u_viol"], "p_viol": d["p_viol"],
    "1-R^2": float(r2), "slope": float(slope), "status": status,
}
print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}", flush=True)
if status == "strict_conv":
    np.savez("results/full_ree/posterior_v3_strict_G14_gamma50.npz",
             mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# Save partial progress
with open("results/full_ree/p0_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ===== Task 3: No-learning 1-R² at G=14, all (γ, τ) =====
print("\n=== Task 3: No-learning 1-R² at G=14 ===\n", flush=True)
u_grid = np.linspace(-UMAX, UMAX, G)
gammas = [0.1, 0.25, 0.3, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 30.0, 50.0]
taus = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
print("γ-line at τ=2:", flush=True)
for gamma in gammas:
    r2_nl, sl_nl, n = no_learning_R2(2.0, gamma, u_grid)
    results["no_learning"].append({
        "tau": 2.0, "gamma": gamma, "no_learning_1-R^2": r2_nl,
        "no_learning_slope": sl_nl,
    })
    print(f"  γ={gamma:>5.2f}: 1-R²={r2_nl:.4e}, slope={sl_nl:.4f}", flush=True)
print("τ-line at γ=0.5:", flush=True)
for tau in taus:
    r2_nl, sl_nl, n = no_learning_R2(tau, 0.5, u_grid)
    results["no_learning"].append({
        "tau": tau, "gamma": 0.5, "no_learning_1-R^2": r2_nl,
        "no_learning_slope": sl_nl,
    })
    print(f"  τ={tau:>5.2f}: 1-R²={r2_nl:.4e}, slope={sl_nl:.4f}", flush=True)

# ===== Task 4: Posteriors table at (u1,u2,u3)=(1,-1,1) =====
print("\n=== Task 4: Posteriors table at (1,-1,1) ===\n", flush=True)
ck = np.load("results/full_ree/posterior_v3_strict_G14_gamma0.5.npz")
mu_REE = ck["mu"]; u_grid_R = ck["u_grid"]; p_grid_R = ck["p_grid"]
p_lo_R = ck["p_lo"]; p_hi_R = ck["p_hi"]
TAU = 2.0; GAMMA = 0.5
u1, u2, u3 = 1.0, -1.0, 1.0
# CRRA REE: solve self-consistently
def mu_at_REE(i_or_u, p):
    """Interpolate μ-row at signal u, evaluate at price p."""
    # Find index closest to u
    idx = int(np.argmin(np.abs(u_grid_R - i_or_u)))
    p_clamp = np.clip(p, p_grid_R[idx, 0], p_grid_R[idx, -1])
    return float(np.interp(p_clamp, p_grid_R[idx, :], mu_REE[idx, :]))
# Or interpolate in u then in p — full bivariate
def mu_at_uv(u, p):
    """Bivariate interpolation of μ at (u, p)."""
    # u-interp first: linear
    if u <= u_grid_R[0]:
        return mu_at_REE(u_grid_R[0], p)
    if u >= u_grid_R[-1]:
        return mu_at_REE(u_grid_R[-1], p)
    i_above = np.searchsorted(u_grid_R, u)
    i_below = i_above - 1
    w = (u - u_grid_R[i_below]) / (u_grid_R[i_above] - u_grid_R[i_below])
    return ((1-w) * mu_at_REE(u_grid_R[i_below], p)
            + w * mu_at_REE(u_grid_R[i_above], p))

def Z_REE(p):
    m1 = mu_at_uv(u1, p)
    m2 = mu_at_uv(u2, p)
    m3 = mu_at_uv(u3, p)
    return sum(crra_demand_vec(np.array([m]), np.array([p]), GAMMA)[0]
               for m in (m1, m2, m3))
try:
    p_REE = brentq(Z_REE, 1e-6, 1 - 1e-6, xtol=1e-12)
    mu_REE_1 = mu_at_uv(u1, p_REE); mu_REE_2 = mu_at_uv(u2, p_REE)
    mu_REE_3 = mu_at_uv(u3, p_REE)
    p_REE_str = f"{p_REE:.6f}"
except ValueError:
    p_REE = float("nan")
    mu_REE_1 = mu_REE_2 = mu_REE_3 = float("nan")
    p_REE_str = "no root"
# CARA at same (u1,u2,u3): logit(p)=T*=2(1-1+1)=2, p=Λ(2)=0.881
T_star = TAU * (u1 + u2 + u3)
mu_CARA = Lam(T_star)   # under FR everyone has μ=p=Λ(T*)
# Prior posterior μ_k = Λ(τu_k) (no learning)
prior_1 = Lam(TAU*u1); prior_2 = Lam(TAU*u2); prior_3 = Lam(TAU*u3)
# No-learning price (same code as Task 3 single point)
def Z_NL(p):
    return sum(crra_demand_vec(np.array([Lam(TAU*u)]), np.array([p]), GAMMA)[0]
               for u in (u1, u2, u3))
p_NL = brentq(Z_NL, 1e-6, 1-1e-6, xtol=1e-12)
results["posteriors_table"] = {
    "u1u2u3": [u1, u2, u3], "tau": TAU, "gamma": GAMMA, "G": G,
    "T_star": T_star,
    "prior": {"u1": prior_1, "u2": prior_2, "u3": prior_3},
    "CARA_FR": {"mu_all": mu_CARA, "p": mu_CARA},
    "REE_CRRA": {"u1": mu_REE_1, "u2": mu_REE_2, "u3": mu_REE_3, "p": p_REE},
    "no_learning_CRRA": {"u1": prior_1, "u2": prior_2, "u3": prior_3, "p": p_NL},
}
print(f"  T* = τ(u1+u2+u3) = {T_star:.4f}")
print(f"\n  Prior posteriors (no info from p):")
print(f"    μ_1 = Λ(τ·{u1:+.0f}) = {prior_1:.4f}")
print(f"    μ_2 = Λ(τ·{u2:+.0f}) = {prior_2:.4f}")
print(f"    μ_3 = Λ(τ·{u3:+.0f}) = {prior_3:.4f}")
print(f"\n  CARA / FR equilibrium:")
print(f"    μ_all = p = Λ(T*) = {mu_CARA:.4f}")
print(f"\n  REE CRRA (γ=0.5, τ=2):")
print(f"    p* = {p_REE_str}")
print(f"    μ_1 = {mu_REE_1:.4f}")
print(f"    μ_2 = {mu_REE_2:.4f}")
print(f"    μ_3 = {mu_REE_3:.4f}")
print(f"\n  No-learning CRRA (μ_k=Λ(τu_k)):")
print(f"    p* = {p_NL:.4f}")
print(f"    μ_k same as prior")

with open("results/full_ree/p0_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n=== P0 DONE ===")
print("Saved: results/full_ree/p0_results.json")

# Survival ratios (Task 3 finalize)
print("\n=== Survival ratios at τ=2 ===")
for nl in [r for r in results["no_learning"] if r["tau"] == 2.0]:
    g = nl["gamma"]
    nl_r2 = nl["no_learning_1-R^2"]
    # Look up REE 1-R² for this γ
    ree_r2 = None
    # Check existing strict files
    try:
        with open("results/full_ree/posterior_v3_strict_emin13.json") as f:
            ext = json.load(f)
        for r in ext["gamma"]:
            if abs(r["gamma"] - g) < 1e-6 and r["status"] == "strict_conv":
                ree_r2 = r["1-R^2"]
                break
    except FileNotFoundError:
        pass
    # Also check from paper-γ results
    for r in results["paper_gammas"]:
        if abs(r["gamma"] - g) < 1e-6 and r["status"] == "strict_conv":
            ree_r2 = r["1-R^2"]; break
    if ree_r2 is not None:
        print(f"  γ={g:>5.2f}: REE 1-R²={ree_r2:.4e}, no-learn 1-R²={nl_r2:.4e}, "
              f"survival = {ree_r2 / nl_r2 if nl_r2 > 0 else float('nan'):.4f}")
    else:
        print(f"  γ={g:>5.2f}: no-learn 1-R²={nl_r2:.4e}, REE not strictly converged")
