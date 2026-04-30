"""Small-step strict G ladder: G=14,15,16,17,18,19,20.

Each cell: Picard-PAVA + dual NK strategy:
  1. Standard NK on Φ. If monotone, accept.
  2. If NK drifts, try gap-reparam-NK with PAVA-p inside the residual.
  3. If still fails, fallback to Picard.

Warm-start chain: each cell warms from the previous strict-converged.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import (
    pava_p_only, pava_u_only,
    encode, decode, pack, unpack,
)

UMAX = 4.0
TAU = 2.0
GAMMA = 0.5
TOL_MAX = 1e-14
GS = [14, 15, 16, 17, 18, 19, 20]


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
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


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, n, na, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi_std(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def F_gap_pava_p(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """Gap-reparam-u + PAVA-p inside residual."""
    base, c = unpack(x, Gu, Gp)
    mu = decode(base, c)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    cand_proj = pava_p_only(cand)
    base_new, c_new = encode(cand_proj)
    return pack(base_new - base, c_new - c)


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def strict_solve_dual(G, mu_warm, u_warm, p_warm):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)

    t0 = time.time()
    # Picard rounds
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.05)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.01)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 10000, 5000, 0.003)
    mu_picard = mu
    d_p = measure(mu_picard, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  G={G} after picard: max={d_p['max']:.3e}, med={d_p['med']:.3e}, "
          f"u/p={d_p['u_viol']}/{d_p['p_viol']}, t={time.time()-t0:.0f}s",
          flush=True)
    if d_p["max"] < TOL_MAX and d_p["u_viol"] == 0 and d_p["p_viol"] == 0:
        return mu_picard, d_p, u_grid, p_grid, p_lo, p_hi, "strict_picard", time.time()-t0

    # Standard NK
    print(f"  G={G} standard NK...", flush=True)
    nk_status_std = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_phi_std(x, mu.shape, u_grid, p_grid, p_lo, p_hi,
                                  TAU, GAMMA),
            mu.ravel(), f_tol=TOL_MAX, maxiter=300,
            method="lgmres", verbose=False)
        mu_nk_std = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
    except NoConvergence as e:
        nk_status_std = "noconv_kept"
        mu_nk_std = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS) if e.args else mu_picard
    except (ValueError, RuntimeError) as exc:
        nk_status_std = f"err:{type(exc).__name__}"
        mu_nk_std = mu_picard
    d_std = measure(mu_nk_std, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  G={G} std-NK: max={d_std['max']:.3e}, med={d_std['med']:.3e}, "
          f"u/p={d_std['u_viol']}/{d_std['p_viol']}, NK={nk_status_std}",
          flush=True)
    if (d_std["max"] < TOL_MAX and d_std["u_viol"] == 0
        and d_std["p_viol"] == 0):
        return mu_nk_std, d_std, u_grid, p_grid, p_lo, p_hi, "strict_std_NK", time.time()-t0

    # Gap-reparam NK with PAVA-p inside
    print(f"  G={G} gap-reparam-NK with PAVA-p inside...", flush=True)
    base0, c0 = encode(mu_picard)
    x0 = pack(base0, c0)
    Gu, Gp = mu_picard.shape
    nk_status_gap = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_gap_pava_p(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi,
                                     TAU, GAMMA),
            x0, f_tol=TOL_MAX, maxiter=300,
            method="lgmres", verbose=False)
        base_f, c_f = unpack(sol, Gu, Gp)
        mu_nk_gap = decode(base_f, c_f)
    except NoConvergence as e:
        nk_status_gap = "noconv_kept"
        if e.args:
            base_f, c_f = unpack(e.args[0], Gu, Gp)
            mu_nk_gap = decode(base_f, c_f)
        else:
            mu_nk_gap = mu_picard
    except (ValueError, RuntimeError) as exc:
        nk_status_gap = f"err:{type(exc).__name__}"
        mu_nk_gap = mu_picard
    mu_nk_gap = np.clip(mu_nk_gap, EPS, 1 - EPS)
    d_gap = measure(mu_nk_gap, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    print(f"  G={G} gap-NK: max={d_gap['max']:.3e}, med={d_gap['med']:.3e}, "
          f"u/p={d_gap['u_viol']}/{d_gap['p_viol']}, NK={nk_status_gap}",
          flush=True)
    if (d_gap["max"] < TOL_MAX and d_gap["u_viol"] == 0
        and d_gap["p_viol"] == 0):
        return mu_nk_gap, d_gap, u_grid, p_grid, p_lo, p_hi, "strict_gap_NK", time.time()-t0

    # Pick the best monotone result
    candidates = []
    if d_p["u_viol"] == 0 and d_p["p_viol"] == 0:
        candidates.append((mu_picard, d_p, "picard"))
    if d_std["u_viol"] == 0 and d_std["p_viol"] == 0:
        candidates.append((mu_nk_std, d_std, "std_NK"))
    if d_gap["u_viol"] == 0 and d_gap["p_viol"] == 0:
        candidates.append((mu_nk_gap, d_gap, "gap_NK"))
    if candidates:
        best = min(candidates, key=lambda x: x[1]["max"])
        return best[0], best[1], u_grid, p_grid, p_lo, p_hi, f"non_strict_{best[2]}", time.time()-t0
    return mu_picard, d_p, u_grid, p_grid, p_lo, p_hi, "fallback_picard_violations", time.time()-t0


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Small-step G ladder, dual NK strategy ===\n", flush=True)
print(f"τ={TAU}, γ={GAMMA}, UMAX={UMAX}\n", flush=True)
results = []
ck0 = np.load("results/full_ree/posterior_v3_strict_G14.npz")
mu_warm = ck0["mu"]; u_warm = ck0["u_grid"]; p_warm = ck0["p_grid"]
print("Seed: G=14 strict\n", flush=True)
results.append({"G": 14, "status": "preexisting_strict",
                "max": float("nan"), "1-R^2": 0.1083})

for G in GS[1:]:
    print(f"\n--- G = {G} ---", flush=True)
    mu, d, ug, pg, plo, phi_, status, elapsed = strict_solve_dual(
        G, mu_warm, u_warm, p_warm)
    r2, slope, _ = measure_R2(mu, ug, pg, plo, phi_, TAU, GAMMA)
    print(f"  ===> status={status}, max={d['max']:.3e}, "
          f"1-R²={r2:.4e}, slope={slope:.4f}, t={elapsed:.0f}s", flush=True)
    rec = {"G": G, "tau": TAU, "gamma": GAMMA,
           "max": d["max"], "med": d["med"],
           "u_viol": d["u_viol"], "p_viol": d["p_viol"],
           "1-R^2": float(r2), "slope": float(slope),
           "status": status, "elapsed_s": elapsed}
    results.append(rec)
    # Save if strict
    if status.startswith("strict"):
        np.savez(f"results/full_ree/posterior_v3_strict_G{G}.npz",
                 mu=mu, u_grid=ug, p_grid=pg, p_lo=plo, p_hi=phi_)
        mu_warm = mu; u_warm = ug; p_warm = pg
    else:
        # If still monotone, use as warm anyway (looser warm)
        if d["u_viol"] == 0 and d["p_viol"] == 0:
            mu_warm = mu; u_warm = ug; p_warm = pg
            print(f"  (using non-strict monotone as warm)", flush=True)
    with open("results/full_ree/small_step_G_ladder.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\n=== SMALL-STEP LADDER SUMMARY ===")
print(f"{'G':>3} {'status':>30} {'max':>10} {'1-R²':>11} {'slope':>7}")
for r in results:
    print(f"{r['G']:>3d} {r.get('status', '?'):>30} {r['max']:>10.2e} "
          f"{r['1-R^2']:>11.4e} {r.get('slope', 0):>7.4f}")
