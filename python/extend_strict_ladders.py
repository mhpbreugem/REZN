"""Extend strict ladders:
  γ: add 5, 10, 20, 30 (toward CARA)
  τ: more resolution + range (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16)

Same strict pipeline as strict_ladder_emin13.py: slow Picard + NK polish,
require max < 1e-14 and zero monotonicity violations.

Warm-start by continuation along each ladder.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
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
        mu_nk = np.clip(sol.reshape(shape), EPS, 1 - EPS)
        return mu_nk, "ok"
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
        print(f"  [{label}] picard r{r_idx+1} (n={n}, α={a}): "
              f"max={d['max']:.3e}, med={d['med']:.3e}, "
              f"u={d['u_viol']}, p={d['p_viol']}, t={elapsed:.0f}s",
              flush=True)
        mu_picard_best = mu
        if d["max"] < TOL_MAX and d["u_viol"] == 0 and d["p_viol"] == 0:
            return mu, d, u_grid, p_grid, p_lo, p_hi, "strict_conv", elapsed
        if d["med"] > last_med * 0.5 and r_idx > 0:
            print(f"  [{label}] stalled; perturbing 1e-5", flush=True)
            mu = mu + np.random.RandomState(42 + r_idx).normal(0, 1e-5, mu.shape)
            mu = np.clip(mu, EPS, 1 - EPS)
            mu = pava_2d(mu)
        last_med = d["med"]

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
        print(f"  [{label}] NK drifted; falling back to picard state",
              flush=True)
        return (mu_picard_best, measure(mu_picard_best, u_grid, p_grid,
                                          p_lo, p_hi, tau, gamma),
                u_grid, p_grid, p_lo, p_hi, "no_strict_conv_fallback",
                elapsed)
    return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "no_strict_conv", elapsed


warnings.filterwarnings("ignore", category=RuntimeWarning)
all_results = {"gamma_extend": [], "tau_extend": []}

# ===== γ extension: 5, 10, 20, 30 (warm from γ=2 strict) =====
print("\n=== γ extension at G=14, τ=2 ===\n", flush=True)
ck = np.load(f"results/full_ree/posterior_v3_strict_G{G}_gamma2.npz")
mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
for gamma in [5.0, 10.0, 20.0, 30.0]:
    print(f"\n--- γ = {gamma} ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        2.0, gamma, mu_warm, u_warm, p_warm, label=f"γ={gamma}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 2.0, gamma)
    rec = {
        "phase": "gamma", "G": G, "tau": 2.0, "gamma": gamma,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status,
    }
    all_results["gamma_extend"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}_gamma{gamma:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg
    with open("results/full_ree/posterior_v3_strict_extended.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ===== τ extension =====
# Down-ladder: 0.5 -> 0.25 (warm from τ=0.5 strict)
# Mid resolution: 1.5, 3, 6 (warm from τ=2 then sequentially)
# Up-ladder: 8 -> 12 -> 16 (warm from τ=8)
print("\n=== τ extension at G=14, γ=0.5 ===\n", flush=True)

# Branch 1: extend down to 0.25
ck = np.load(f"results/full_ree/posterior_v3_strict_G{G}_tau0.5.npz")
mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
print("\n--- τ = 0.25 (warm from τ=0.5) ---", flush=True)
mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
    0.25, 0.5, mu_w, u_w, p_w, label=f"τ=0.25")
r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, 0.25, 0.5)
rec = {"phase": "tau", "G": G, "tau": 0.25, "gamma": 0.5,
        "max": d["max"], "med": d["med"],
        "u_viol": d["u_viol"], "p_viol": d["p_viol"],
        "1-R^2": float(r2), "slope": float(slope),
        "elapsed_s": elapsed, "status": status}
all_results["tau_extend"].append(rec)
print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}", flush=True)
if status == "strict_conv":
    np.savez(f"results/full_ree/posterior_v3_strict_G{G}_tau0.25.npz",
             mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# Branch 2: mid-resolution τ=1.5 (warm from τ=2)
# τ=3 (warm from τ=2) and τ=6 (warm from τ=4)
# Run sequentially
ck = np.load(f"results/full_ree/posterior_v3_strict_G{G}_tau2.npz")
mu_w2 = ck["mu"]; u_w2 = ck["u_grid"]; p_w2 = ck["p_grid"]
for tau in [1.5, 3.0]:
    print(f"\n--- τ = {tau} (warm from τ=2) ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        tau, 0.5, mu_w2, u_w2, p_w2, label=f"τ={tau}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, tau, 0.5)
    rec = {"phase": "tau", "G": G, "tau": tau, "gamma": 0.5,
            "max": d["max"], "med": d["med"],
            "u_viol": d["u_viol"], "p_viol": d["p_viol"],
            "1-R^2": float(r2), "slope": float(slope),
            "elapsed_s": elapsed, "status": status}
    all_results["tau_extend"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# τ=6 warm from τ=4
ck = np.load(f"results/full_ree/posterior_v3_strict_G{G}_tau4.npz")
mu_w4 = ck["mu"]; u_w4 = ck["u_grid"]; p_w4 = ck["p_grid"]
for tau in [6.0]:
    print(f"\n--- τ = {tau} (warm from τ=4) ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        tau, 0.5, mu_w4, u_w4, p_w4, label=f"τ={tau}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, tau, 0.5)
    rec = {"phase": "tau", "G": G, "tau": tau, "gamma": 0.5,
            "max": d["max"], "med": d["med"],
            "u_viol": d["u_viol"], "p_viol": d["p_viol"],
            "1-R^2": float(r2), "slope": float(slope),
            "elapsed_s": elapsed, "status": status}
    all_results["tau_extend"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)

# Branch 3: extend up from τ=8 to 12, 16
ck = np.load(f"results/full_ree/posterior_v3_strict_G{G}_tau8.npz")
mu_w8 = ck["mu"]; u_w8 = ck["u_grid"]; p_w8 = ck["p_grid"]
for tau in [12.0, 16.0]:
    print(f"\n--- τ = {tau} (warm from previous) ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve(
        tau, 0.5, mu_w8, u_w8, p_w8, label=f"τ={tau}")
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, tau, 0.5)
    rec = {"phase": "tau", "G": G, "tau": tau, "gamma": 0.5,
            "max": d["max"], "med": d["med"],
            "u_viol": d["u_viol"], "p_viol": d["p_viol"],
            "1-R^2": float(r2), "slope": float(slope),
            "elapsed_s": elapsed, "status": status}
    all_results["tau_extend"].append(rec)
    print(f"  ===> 1-R²={r2:.4e}, slope={slope:.4f}, status={status}",
          flush=True)
    if status == "strict_conv":
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}_tau{tau:g}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_w8 = mu; u_w8 = ug; p_w8 = pg

    with open("results/full_ree/posterior_v3_strict_extended.json", "w") as f:
        json.dump(all_results, f, indent=2)

print("\n=== EXTENDED LADDERS COMPLETE ===\n")
for ph in ["gamma_extend", "tau_extend"]:
    print(f"-- {ph} --")
    print(f"  {'param':>6} {'max':>10} {'1-R²':>11} {'slope':>7} {'status':>14}")
    for r in all_results[ph]:
        param = r["gamma"] if ph == "gamma_extend" else r["tau"]
        print(f"  {param:>6.2f} {r['max']:>10.2e} {r['1-R^2']:>11.4e} "
              f"{r['slope']:>7.4f} {r['status']:>14}")
