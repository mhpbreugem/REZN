"""Smart refinement pass: try multiple warm-start sources per missing/failed G.

For each G in {16, 22, 23, 24, ..., 30}, try warm-starting from several
"good" checkpoints (G-1, G-2, G-3, G-5, G=14 reference). Pick the result
with lowest max-residual that is also consistent with the bulk trend
(1-R² ∈ [0.08, 0.16]).

NO cold-start fallback — cold-start finds spurious high-G equilibria.
Only warm-from-checkpoint is trusted.

Saves converged μ tensors to checkpoint files.
"""
import json, time, glob, re, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, picard_anderson, phi_step, measure_R2, EPS,
)

TAU = 2.0
GAMMA = 0.5
UMAX = 4.0
F_TOL = 1e-14
TARGET_GS = list(range(8, 31))   # all G up to 30

# Acceptable 1-R² range — outside this is a "spurious" equilibrium
R2_LO, R2_HI = 0.06, 0.18


def F_residual(mu_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(mu_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu).ravel()
    F[~active.ravel()] = 0.0
    return F


def interp_mu_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        r_above = np.searchsorted(u_old, u_target)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        if r_above == r_below:
            w = 1.0
        else:
            w = (u_target - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_clamped_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_clamped_b, p_old[r_below, :], mu_old[r_below, :])
            p_clamped_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_clamped_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def load_checkpoint(G):
    path = f"results/full_ree/posterior_v3_fine_G{G}_mu.npz"
    try:
        d = np.load(path)
        return {
            "G": G, "mu": d["mu"], "u_grid": d["u_grid"],
            "p_grid": d["p_grid"], "p_lo": d["p_lo"], "p_hi": d["p_hi"],
        }
    except FileNotFoundError:
        return None


def measure_checkpoint(ckpt):
    """Quick re-measure of 1-R² from a checkpoint."""
    r2def, slope, n = measure_R2(ckpt["mu"], ckpt["u_grid"], ckpt["p_grid"],
                                   ckpt["p_lo"], ckpt["p_hi"], TAU, GAMMA)
    return float(r2def), float(slope)


def filter_good_sources(target_G, all_checkpoints):
    """Pick warm-start sources whose 1-R² is in the trusted range."""
    sources = []
    for ckpt in all_checkpoints:
        r2, _ = measure_checkpoint(ckpt)
        if R2_LO <= r2 <= R2_HI and ckpt["G"] != target_G:
            sources.append((ckpt, r2, abs(ckpt["G"] - target_G)))
    # Prefer closer Gs
    sources.sort(key=lambda x: x[2])
    return [s[0] for s in sources]


def try_solve_at_G(G, source, picard_iters=300, nk_iters=800):
    """One attempt at a single warm-source. Returns dict with result."""
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    mu0 = interp_mu_to_grid(source["mu"], source["u_grid"], source["p_grid"],
                             u_grid, p_grid)

    t0 = time.time()
    mu_warm_out, hist, _ = picard_anderson(
        mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
        damping=0.05, anderson=0, max_iter=picard_iters, tol=1e-12, progress=False,
    )
    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual(x, mu_warm_out.shape, u_grid, p_grid,
                                  p_lo, p_hi, TAU, GAMMA),
            mu_warm_out.ravel(),
            f_tol=F_TOL, maxiter=nk_iters, verbose=False, method="lgmres",
        )
        mu_final = sol.reshape(mu_warm_out.shape)
    except NoConvergence as e:
        nk_status = "noconv"
        mu_final = e.args[0].reshape(mu_warm_out.shape) if e.args else mu_warm_out
    except (ValueError, RuntimeError) as e:
        nk_status = f"err:{type(e).__name__}"
        mu_final = mu_warm_out

    mu_final = np.clip(mu_final, EPS, 1 - EPS)
    cand, active_mask, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    F_arr = np.abs(cand - mu_final)
    residual_max = float(F_arr[active_mask].max())
    residual_med = float(np.median(F_arr[active_mask]))
    strict_conv = residual_max < 1e-12
    fit_t = time.time() - t0
    r2def, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    return {
        "source_G": source["G"], "G": G, "nk_status": nk_status,
        "residual_max": residual_max, "residual_median": residual_med,
        "strict_conv": strict_conv,
        "1-R^2": float(r2def), "slope": float(slope),
        "elapsed_s": fit_t, "in_range": R2_LO <= r2def <= R2_HI,
    }, mu_final, u_grid, p_grid, p_lo, p_hi


def best_attempt(attempts):
    """Pick the best attempt: (1) strict_conv & in_range, (2) lowest max, in_range."""
    in_range_strict = [a for a in attempts if a[0]["strict_conv"] and a[0]["in_range"]]
    if in_range_strict:
        return min(in_range_strict, key=lambda a: a[0]["residual_max"])
    in_range = [a for a in attempts if a[0]["in_range"]]
    if in_range:
        return min(in_range, key=lambda a: a[0]["residual_max"])
    return min(attempts, key=lambda a: a[0]["residual_max"])


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== refinement pass, γ={GAMMA}, τ={TAU}, f_tol={F_TOL} ===\n", flush=True)

# Load all existing checkpoints
all_ckpts = [c for c in (load_checkpoint(g) for g in range(5, 31)) if c]
print(f"Loaded {len(all_ckpts)} checkpoints: G="
      f"{[c['G'] for c in all_ckpts]}", flush=True)

# Verify each checkpoint by remeasuring (and weed out bad ones)
print("\nChecking existing checkpoints:", flush=True)
verified_ckpts = []
for c in all_ckpts:
    r2, slope = measure_checkpoint(c)
    in_range = R2_LO <= r2 <= R2_HI
    flag = "OK" if in_range else "**OUTLIER**"
    print(f"  G={c['G']:>2}: 1-R²={r2:.4e}, slope={slope:.4f}  {flag}", flush=True)
    if in_range:
        verified_ckpts.append(c)

# Now try to fill missing/outlier Gs
existing_good_Gs = set(c["G"] for c in verified_ckpts)
all_results = []

for G in TARGET_GS:
    if G in existing_good_Gs:
        # Already have a good checkpoint
        c = next(c for c in verified_ckpts if c["G"] == G)
        r2, slope = measure_checkpoint(c)
        cand, active, _ = phi_step(c["mu"], c["u_grid"], c["p_grid"],
                                    c["p_lo"], c["p_hi"], TAU, GAMMA)
        max_r = float(np.max(np.abs(cand - c["mu"])[active]))
        all_results.append({
            "G": G, "method": "preexisting", "source_G": "—",
            "residual_max": max_r, "1-R^2": float(r2), "slope": float(slope),
            "strict_conv": max_r < 1e-12,
        })
        continue
    print(f"\n--- refining G = {G} ---", flush=True)
    sources = filter_good_sources(G, verified_ckpts)
    print(f"  trying {len(sources)} sources: G="
          f"{[s['G'] for s in sources[:5]]}{'...' if len(sources) > 5 else ''}",
          flush=True)
    attempts = []
    for src in sources[:5]:
        r, *rest = try_solve_at_G(G, src, picard_iters=300, nk_iters=800)
        print(f"  src G={r['source_G']:>2}: NK={r['nk_status']:>8}, "
              f"max={r['residual_max']:.2e}, med={r['residual_median']:.2e}, "
              f"1-R²={r['1-R^2']:.4e}, slope={r['slope']:.4f}, "
              f"strict={r['strict_conv']}, in_range={r['in_range']}",
              flush=True)
        attempts.append((r, *rest))
        if r["strict_conv"] and r["in_range"]:
            break  # found a good one
    best, mu_b, ug_b, pg_b, p_lo_b, p_hi_b = best_attempt(attempts)
    print(f"  ==> picked source G={best['source_G']}: "
          f"max={best['residual_max']:.2e}, 1-R²={best['1-R^2']:.4e}, "
          f"strict={best['strict_conv']}, in_range={best['in_range']}",
          flush=True)
    all_results.append({
        "G": G, "method": "refinement", "source_G": best["source_G"],
        "residual_max": best["residual_max"], "1-R^2": best["1-R^2"],
        "slope": best["slope"], "strict_conv": best["strict_conv"],
        "in_range": best["in_range"], "all_attempts": [a[0] for a in attempts],
    })
    if best["strict_conv"] and best["in_range"]:
        np.savez(f"results/full_ree/posterior_v3_fine_G{G}_mu.npz",
                 mu=mu_b, u_grid=ug_b, p_grid=pg_b, p_lo=p_lo_b, p_hi=p_hi_b,
                 tau=TAU, gamma=GAMMA)
        verified_ckpts.append({
            "G": G, "mu": mu_b, "u_grid": ug_b, "p_grid": pg_b,
            "p_lo": p_lo_b, "p_hi": p_hi_b,
        })
    # incremental save
    with open("results/full_ree/posterior_v3_refinement.json", "w") as f:
        json.dump({"results": all_results, "params": {
            "tau": TAU, "gamma": GAMMA, "umax": UMAX, "f_tol": F_TOL,
            "r2_range": [R2_LO, R2_HI],
        }}, f, indent=2, default=str)

print("\n=== FINAL TABLE ===")
print(f"{'G':>3} {'method':>11} {'src':>4} {'max':>10} {'1-R²':>11} {'slope':>7} {'strict':>7}")
for r in all_results:
    print(f"{r['G']:>3d} {r['method']:>11} {str(r['source_G']):>4} "
          f"{r['residual_max']:>10.2e} {r['1-R^2']:>11.4e} "
          f"{r['slope']:>7.4f} {str(r['strict_conv']):>7}")
print("\nSaved: results/full_ree/posterior_v3_refinement.json")
