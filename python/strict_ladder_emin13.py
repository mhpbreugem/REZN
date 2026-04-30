"""Strict ladder: each cell must hit median Φ-residual < 1e-13 with both
monotonicity directions clean. Uses iterative slow-Picard-PAVA rounds
with progressively smaller damping; perturbs μ slightly between rounds
if a round stalls.

Phase 1: G ladder
Phase 2: γ ladder
Phase 3: τ ladder

Each cell warm-starts from the previous strict-converged cell.
"""
import time, json, warnings
import numpy as np

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from scipy.optimize import newton_krylov, NoConvergence
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
TOL_MAX = 1e-14   # strict: every cell at machine precision


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
    """One round of slow Picard-PAVA at constant damping with Cesaro averaging."""
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
    """F = Φ(μ) - μ; raw NK target on active cells."""
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu)
    F[~active] = 0.0
    return F.ravel()


def nk_polish(mu_warm, u_grid, p_grid, p_lo, p_hi, tau, gamma,
              f_tol=1e-14, maxiter=300):
    """NK polish on Φ-residual from a near-FP warm. Falls back to warm if
    NK drifts (monotonicity violations or 1-R² shifts dramatically)."""
    shape = mu_warm.shape
    try:
        sol = newton_krylov(
            lambda x: F_phi_residual(x, shape, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma),
            mu_warm.ravel(),
            f_tol=f_tol, maxiter=maxiter,
            method="lgmres", verbose=False,
        )
        mu_nk = np.clip(sol.reshape(shape), EPS, 1 - EPS)
        return mu_nk, "ok"
    except NoConvergence as e:
        mu_nk = (np.clip(e.args[0].reshape(shape), EPS, 1 - EPS)
                  if e.args else mu_warm)
        return mu_nk, "noconv_kept"
    except (ValueError, RuntimeError) as exc:
        return mu_warm, f"err:{type(exc).__name__}"


def strict_solve(G, tau, gamma, mu_warm, u_warm, p_warm, label=""):
    """Solve at (G, tau, gamma) strictly: max < 1e-14, monotone.

    Pipeline:
    1. Slow-Picard-PAVA rounds (warm-up the basin, drive bulk residual down)
    2. NK polish on Φ (machine-precision FP)
    3. Verify monotone & max < 1e-14
    If NK drifts (monotonicity broken), fall back to last clean Picard state.
    """
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G)
    if mu_warm is None:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(tau * u)
    else:
        mu = interp_mu_to_grid(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)

    rounds = [
        # (n_iter, n_avg, alpha)
        (5000, 2500, 0.05),
        (5000, 2500, 0.01),
        (10000, 5000, 0.003),
    ]

    t0 = time.time()
    last_med = float("inf")
    mu_picard_best = mu
    for r_idx, (n, na, a) in enumerate(rounds):
        mu = picard_pava_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                                 n, na, a)
        d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        elapsed = time.time() - t0
        print(f"  [{label}] picard r{r_idx+1} (n={n}, α={a}): "
              f"max={d['max']:.3e}, med={d['med']:.3e}, "
              f"u={d['u_viol']}, p={d['p_viol']}, t={elapsed:.0f}s",
              flush=True)
        mu_picard_best = mu
        # If max already at machine precision, skip NK
        if d["max"] < TOL_MAX and d["u_viol"] == 0 and d["p_viol"] == 0:
            return mu, d, u_grid, p_grid, p_lo, p_hi, "strict_conv", elapsed
        # If round didn't reduce med significantly, perturb
        if d["med"] > last_med * 0.5 and r_idx > 0:
            print(f"  [{label}] stalled; perturbing 1e-5", flush=True)
            mu = mu + np.random.RandomState(42 + r_idx).normal(0, 1e-5, mu.shape)
            mu = np.clip(mu, EPS, 1 - EPS)
            mu = pava_2d(mu)
        last_med = d["med"]

    # NK polish to push max < 1e-14
    print(f"  [{label}] NK polish on Φ-residual (target {TOL_MAX:.0e})",
          flush=True)
    mu_nk, nk_status = nk_polish(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                                   f_tol=TOL_MAX, maxiter=300)
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    elapsed = time.time() - t0
    print(f"  [{label}] post-NK: max={d_nk['max']:.3e}, med={d_nk['med']:.3e}, "
          f"u={d_nk['u_viol']}, p={d_nk['p_viol']}, NK={nk_status}, "
          f"t={elapsed:.0f}s", flush=True)

    if (d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0
        and d_nk["p_viol"] == 0):
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_conv", elapsed
    if d_nk["u_viol"] > 0 or d_nk["p_viol"] > 0:
        print(f"  [{label}] NK drifted ({d_nk['u_viol']}u, {d_nk['p_viol']}p "
              f"violations); falling back to picard state", flush=True)
        return (mu_picard_best, measure(mu_picard_best, u_grid, p_grid,
                                          p_lo, p_hi, tau, gamma),
                u_grid, p_grid, p_lo, p_hi, "no_strict_conv_fallback",
                elapsed)
    return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "no_strict_conv", elapsed


warnings.filterwarnings("ignore", category=RuntimeWarning)
all_results = {"G": [], "gamma": [], "tau": []}

# ===== Phase 1: G ladder at γ=0.5, τ=2 =====
print("\n=== Phase 1: G ladder at γ=0.5, τ=2 ===\n", flush=True)
G_LADDER = [10, 12, 14, 16, 18, 20]
mu_warm = u_warm = p_warm = None
# Try to seed from existing PAVA-Cesaro G=14 to skip warm-up
try:
    ck0 = np.load("results/full_ree/posterior_v3_pava_G14_gamma0.5.npz")
    mu_warm = ck0["mu"]; u_warm = ck0["u_grid"]; p_warm = ck0["p_grid"]
    print("Seeded from PAVA-Cesaro G=14, γ=0.5", flush=True)
except FileNotFoundError:
    pass

for G in G_LADDER:
    print(f"\n--- G = {G} ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        G, 2.0, 0.5, mu_warm, u_warm, p_warm, label=f"G={G}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, 0.5)
    rec = {
        "phase": "G", "G": G, "tau": 2.0, "gamma": 0.5,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status,
    }
    all_results["G"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_strict_emin13.json", "w") as f:
        json.dump(all_results, f, indent=2)

# Pick best converged G
G_good = [r for r in all_results["G"] if r["status"] == "strict_conv"]
G_BEST = max(G_good, key=lambda r: r["G"])["G"] if G_good else 14
print(f"\n=== G_BEST = {G_BEST} (strict-converged) ===\n", flush=True)
ck_best = np.load(f"results/full_ree/posterior_v3_strict_G{G_BEST}.npz")

# ===== Phase 2: γ ladder =====
print(f"=== Phase 2: γ ladder at G={G_BEST}, τ=2 ===\n", flush=True)
GAMMAS = [0.5, 0.3, 0.1, 1.0, 2.0]
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for gamma in GAMMAS:
    print(f"\n--- γ = {gamma} ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        G_BEST, 2.0, gamma, mu_warm, u_warm, p_warm, label=f"γ={gamma}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, gamma)
    rec = {
        "phase": "gamma", "G": G_BEST, "tau": 2.0, "gamma": gamma,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status,
    }
    all_results["gamma"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G_BEST}_gamma{gamma:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_strict_emin13.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ===== Phase 3: τ ladder =====
print(f"\n=== Phase 3: τ ladder at G={G_BEST}, γ=0.5 ===\n", flush=True)
TAUS = [2.0, 1.0, 0.5, 4.0, 8.0]
mu_warm = ck_best["mu"]; u_warm = ck_best["u_grid"]; p_warm = ck_best["p_grid"]
for tau in TAUS:
    print(f"\n--- τ = {tau} ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        G_BEST, tau, 0.5, mu_warm, u_warm, p_warm, label=f"τ={tau}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, tau, 0.5)
    rec = {
        "phase": "tau", "G": G_BEST, "tau": tau, "gamma": 0.5,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status,
    }
    all_results["tau"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G_BEST}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_strict_emin13.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ===== Summary =====
print(f"\n=== STRICT 1e-13 LADDER COMPLETE ===\n", flush=True)
for ph in ["G", "gamma", "tau"]:
    print(f"-- {ph} ladder --")
    print(f"  {'p':>6} {'max':>10} {'med':>10} {'u/p':>5} {'1-R²':>11} "
          f"{'slope':>7} {'status':>14}")
    for r in all_results[ph]:
        param = (r["G"] if ph == "G"
                 else (r["gamma"] if ph == "gamma" else r["tau"]))
        print(f"  {param:>6.2f} {r['max']:>10.2e} {r['med']:>10.2e} "
              f"{r['u_viol']:>2}/{r['p_viol']:<2} "
              f"{r['1-R^2']:>11.4e} {r['slope']:>7.4f} "
              f"{r['status']:>14}")
    print()
