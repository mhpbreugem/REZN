"""Search (tau, gamma) configurations maximizing 1-R² subject to strict convergence.

Convergence criterion:  ‖Φ(P) - P‖∞  <  ABSTOL  (= 1e-10 by default).
Only strictly converged solutions are recorded; the rest are logged but
excluded from the ranked output.

Sampling strategy:
  1. Deterministic coarse grid over {γ} × {τ} with some symmetry culling.
  2. Random log-uniform samples for broader exploration.
  3. Targeted "endogenous noise trader" samples: low γ on one agent, high γ
     on the others, aligned/misaligned with τ.

Each config is tried with a damping ladder α ∈ {1.0, 0.3, 0.1, 0.03}
and max-iter ladder {3k, 10k, 30k}. We accept the first α that meets
the convergence criterion.

All results stream to /home/user/REZN/python/search_het_results.csv.

Usage:
    python3 -u search_het.py --budget-hours 3 --G 9 --seed 42
"""
import argparse
import csv
import itertools
import os
import random
import sys
import time

import numpy as np

import rezn_het as rh

# ---------------------------------------------------------
# Arguments / global config
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--G", type=int, default=9)
parser.add_argument("--umax", type=float, default=2.0)
parser.add_argument("--abstol", type=float, default=1e-10, help="‖Φ-I‖∞ tolerance")
parser.add_argument("--f-tol", type=float, default=1e-6, help="‖F‖∞ (market-clear) tolerance")
parser.add_argument("--budget-hours", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str,
                    default="/home/user/REZN/python/search_het_results.csv")
args = parser.parse_args()

G = args.G
UMAX = args.umax
ABSTOL = args.abstol
F_TOL = args.f_tol
BUDGET_S = args.budget_hours * 3600.0
random.seed(args.seed); np.random.seed(args.seed)

U_REPORT = (1.0, -1.0, 1.0)


# ---------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------

def coarse_grid():
    """Deterministic coarse grid with light symmetry culling."""
    gammas_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 50.0]
    taus_grid = [0.3, 1.0, 3.0, 10.0]
    # Homogeneous baselines: all gammas equal, all taus equal
    for g in gammas_grid:
        for t in taus_grid:
            yield (t, t, t), (g, g, g)
    # Heterogeneous gammas with equal taus
    for g1, g2, g3 in itertools.combinations_with_replacement(gammas_grid, 3):
        if (g1, g2, g3) == (g1,)*3: continue  # dedup with homo loop
        for t in taus_grid:
            yield (t, t, t), (g1, g2, g3)
    # Equal gammas with het taus
    for t1, t2, t3 in itertools.combinations_with_replacement(taus_grid, 3):
        if (t1, t2, t3) == (t1,)*3: continue
        for g in gammas_grid:
            yield (t1, t2, t3), (g, g, g)


def random_loguniform(n):
    """Random (tau_vec, gamma_vec) with log-uniform components — wide range."""
    for _ in range(n):
        tau = tuple(10 ** np.random.uniform(np.log10(0.1), np.log10(10.0), 3))
        gam = tuple(10 ** np.random.uniform(np.log10(0.1), np.log10(50.0), 3))
        yield tau, gam


def extreme_misaligned(n):
    """Low-γ / low-τ on one agent, high on the others (aligned or not)."""
    for _ in range(n):
        agent = np.random.randint(3)
        tau_noisy = 10 ** np.random.uniform(np.log10(0.1), np.log10(1.0))
        tau_info = 10 ** np.random.uniform(np.log10(2.0), np.log10(10.0))
        gam_aggressive = 10 ** np.random.uniform(np.log10(0.1), np.log10(0.5))
        gam_safe = 10 ** np.random.uniform(np.log10(3.0), np.log10(50.0))
        taus = [tau_info, tau_info, tau_info]
        gams = [gam_safe, gam_safe, gam_safe]
        taus[agent] = tau_noisy
        gams[agent] = gam_aggressive  # aggressive trader on noisy signal
        yield tuple(taus), tuple(gams)


def extreme_spread(n):
    """Very different τ's and γ's across agents (wide spread)."""
    for _ in range(n):
        taus = tuple(10 ** np.random.uniform(np.log10(0.1), np.log10(10.0), 3))
        gams = tuple(10 ** np.random.uniform(np.log10(0.1), np.log10(50.0), 3))
        # accept only configs with at least 10x range in each vector
        if max(taus)/min(taus) < 10 and max(gams)/min(gams) < 10:
            continue
        yield taus, gams


# ---------------------------------------------------------
# 1-R² with heterogeneous CARA reference
# ---------------------------------------------------------

def one_minus_R2_het(Pg, u, taus, gammas):
    """Regress logit(p) on the CARA FR predictor
       T_CARA = Σ (τ_k/γ_k) u_k  /  Σ (1/γ_k).
    1 - R² is the fraction of variance NOT explained by this linear combo."""
    G = u.shape[0]
    taus = np.asarray(taus, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1.0 - Pg)).reshape(-1)
    T = np.empty(G ** 3)
    k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T[k] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
                k += 1
    y_c = y - y.mean(); T_c = T - T.mean()
    Syy = float((y_c*y_c).sum()); STT = float((T_c*T_c).sum())
    SyT = float((y_c*T_c).sum())
    if Syy == 0.0 or STT == 0.0:
        return 0.0
    R2 = (SyT*SyT) / (Syy*STT)
    return max(0.0, 1.0 - R2)


# ---------------------------------------------------------
# Per-config solver (damping ladder, early exit on convergence)
# ---------------------------------------------------------

# Compact budget: the waterfall is only 3 attempts now.
# At G=9 each Picard iter ~5 ms, so worst-case per config is
# (1000 + 2500 + 4000)*5ms ≈ 38 s. ~300-600 configs in 3 h.
ALPHA_LADDER  = [1.0, 0.3, 0.1]
MAXITER_TABLE = {1.0: 1000, 0.3: 2500, 0.1: 4000}


# Cache of successfully-converged P tensors for warm-starting.
# List of dicts: {"log_tg": ndarray(6), "P_star": ndarray(G,G,G)}
_CACHE = []


def _log_tg(taus, gammas):
    return np.log(np.concatenate([np.asarray(taus, dtype=float),
                                  np.asarray(gammas, dtype=float)]))


def _cara_fr_tensor(G, u, taus, gammas):
    """CARA full-revelation analytical prediction:
       logit(p) = Σ (τ_k / γ_k) u_k / Σ (1/γ_k)."""
    taus = np.asarray(taus, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    denom = (1.0 / gammas).sum()
    w = (taus / gammas) / denom
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                z = w[0]*u[i] + w[1]*u[j] + w[2]*u[l]
                P[i,j,l] = 1.0 / (1.0 + np.exp(-z))
    return np.clip(P, 1e-9, 1.0 - 1e-9)


def _nearest_cached(taus, gammas, max_dist=2.3):
    """Return P_star of the cached entry closest in log-parameter space,
    or None if the cache is empty or the nearest is further than max_dist."""
    if not _CACHE:
        return None
    q = _log_tg(taus, gammas)
    dists = [float(np.linalg.norm(e["log_tg"] - q)) for e in _CACHE]
    i = int(np.argmin(dists))
    return _CACHE[i]["P_star"] if dists[i] <= max_dist else None


def _idw_interp(taus, gammas, k=5, power=2.0, max_dist=4.0):
    """Inverse-distance-weighted interpolation of P (in logit space) from
    up to k nearest cached points within max_dist in log-param L2 norm.
    Also covers mild extrapolation when the query is slightly outside the
    cached cloud (weights remain finite)."""
    if len(_CACHE) < 2:
        return None
    q = _log_tg(taus, gammas)
    dists = np.array([float(np.linalg.norm(e["log_tg"] - q)) for e in _CACHE])
    keep = np.where(dists <= max_dist)[0]
    if keep.size < 2:
        return None
    order = keep[np.argsort(dists[keep])][:k]
    d = dists[order]
    w = 1.0 / (d ** power + 1e-12)
    w /= w.sum()
    logit_avg = np.zeros_like(_CACHE[0]["P_star"])
    for idx, wi in zip(order, w):
        P = _CACHE[idx]["P_star"]
        logit_avg += wi * np.log(P / (1.0 - P))
    P_init = 1.0 / (1.0 + np.exp(-logit_avg))
    return np.clip(P_init, 1e-9, 1.0 - 1e-9)


def solve_with_ladder(taus, gammas):
    best = {"PhiI": float("inf"), "Finf": float("inf"),
            "iters": 0, "time": 0.0, "alpha": None,
            "P_star": None, "converged": False, "warm": False}

    # Initialisation ladder — NEVER start from CARA-FR:
    #   CARA-FR is a formal (knife-edge) fixed point of Φ even under CRRA
    #   demand, so Picard initialised there gets stuck in the FR basin
    #   rather than descending to the PR fixed point we want. Instead:
    #     1. COLD (no-learning): agents use only own signal → posteriors
    #        perturbed from FR → Picard descends to PR.
    #     2. warm-start from nearest cached CONVERGED P (which must itself
    #        have been found from cold start or from another warm PR seed,
    #        so the PR character propagates).
    init_attempts = [("cold", None)]
    P_warm = _nearest_cached(taus, gammas)
    if P_warm is not None:
        init_attempts.append(("warm", P_warm))
    P_idw = _idw_interp(taus, gammas)
    if P_idw is not None:
        init_attempts.append(("idw", P_idw))

    for init_tag, P_init in init_attempts:
        for alpha in ALPHA_LADDER:
            mi = MAXITER_TABLE[alpha]
            t0 = time.time()
            try:
                res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                                      maxiters=mi, abstol=ABSTOL,
                                      alpha=alpha, P_init=P_init)
            except Exception as e:
                sys.stderr.write(f"[error tau={taus} gamma={gammas} alpha={alpha} init={init_tag}]: {e}\n")
                continue
            dt = time.time() - t0
            PhiI = res["history"][-1] if res["history"] else float("inf")
            Finf = float(np.abs(res["residual"]).max())
            cand = {"PhiI": PhiI, "Finf": Finf,
                    "iters": len(res["history"]), "time": dt,
                    "alpha": alpha, "P_star": res["P_star"],
                    "converged": (PhiI < ABSTOL) and (Finf < F_TOL),
                    "init": init_tag,
                    "warm": (init_tag == "warm")}
            if cand["converged"] and not best["converged"]:
                best = cand
            elif cand["converged"] and best["converged"] and cand["time"] < best["time"]:
                best = cand
            elif not best["converged"] and cand["PhiI"] < best["PhiI"]:
                best = cand
            if cand["converged"]:
                # cache and return
                _CACHE.append({"log_tg": _log_tg(taus, gammas),
                               "P_star": cand["P_star"].copy()})
                return best
    return best


# ---------------------------------------------------------
# Main loop with CSV streaming
# ---------------------------------------------------------

def _evaluate(taus, gammas, u, i_r, j_r, l_r, best):
    """Helper: pack Picard output into a CSV row dict."""
    P = best["P_star"]
    if P is None:
        return None
    mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r,j_r,l_r]), P, u,
                          np.asarray(taus, dtype=float))
    one_het = one_minus_R2_het(P, u, taus, gammas)
    one_eq  = rh.one_minus_R2(P, u, taus)
    return {
        "taus": taus, "gammas": gammas,
        "alpha": best["alpha"], "iters": best["iters"],
        "time": best["time"], "PhiI": best["PhiI"], "Finf": best["Finf"],
        "oneR2_het": one_het, "oneR2_eq": one_eq,
        "p_star": float(P[i_r,j_r,l_r]), "mu": tuple(mu),
        "converged": best["converged"], "init": best.get("init",""),
    }


def _flush_csv(out, rows):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
                    "alpha","iters","time_s","PhiI","Finf",
                    "oneR2_het","oneR2_eq","p_star",
                    "mu_1","mu_2","mu_3","pr_gap","converged","init"])
        for r in rows:
            if r is None: continue
            t = r["taus"]; g = r["gammas"]; mu = r["mu"]
            w.writerow(list(t) + list(g) +
                       [f"{r['alpha']}", r["iters"], f"{r['time']:.2f}",
                        f"{r['PhiI']:.3e}", f"{r['Finf']:.3e}",
                        f"{r['oneR2_het']:.6e}", f"{r['oneR2_eq']:.6e}",
                        f"{r['p_star']:.10f}",
                        f"{mu[0]:.8f}", f"{mu[1]:.8f}", f"{mu[2]:.8f}",
                        f"{mu[0]-mu[1]:.6f}", int(r["converged"]), r["init"]])


def main():
    u = rh.build_grid(G, UMAX)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(U_REPORT[0]), idx(U_REPORT[1]), idx(U_REPORT[2])

    cands = list(coarse_grid()) \
          + list(random_loguniform(200)) \
          + list(extreme_misaligned(100)) \
          + list(extreme_spread(200))
    seen = set(); unique = []
    for t, g in cands:
        key = (tuple(round(x, 6) for x in t), tuple(round(x, 6) for x in g))
        if key in seen: continue
        seen.add(key); unique.append((t, g))
    print(f"total candidate configs: {len(unique)}  (budget {args.budget_hours:.1f} h)")
    sys.stdout.flush()

    _ = rh.solve_picard(5, 2.0, 0.5, maxiters=3, abstol=1e-3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- PASS 1: main sweep (configs in order) ----
    t_start = time.time()
    rows = []
    best_1mR2 = 0.0; best_tag = None; n_conv = 0
    for n_done, (taus, gammas) in enumerate(unique, start=1):
        if time.time() - t_start > BUDGET_S * 0.65:
            print(f"\npass-1 time cap reached after {n_done-1} configs")
            break
        best = solve_with_ladder(taus, gammas)
        row = _evaluate(taus, gammas, u, i_r, j_r, l_r, best)
        rows.append(row)
        if row and row["converged"]:
            n_conv += 1
            if row["oneR2_het"] > best_1mR2:
                best_1mR2 = row["oneR2_het"]; best_tag = (taus, gammas, row["alpha"])
        if n_done % 10 == 0 or n_done == 1:
            dt = time.time() - t_start
            eta = (dt / n_done) * (len(unique) - n_done) / 3600.0
            print(f"[pass-1 {n_done:4d}/{len(unique)}] converged={n_conv}  "
                  f"best 1-R²_het={best_1mR2:.3e}  at={best_tag}  "
                  f"elapsed={dt/60:.1f}min  ETA={eta:.1f}h")
            sys.stdout.flush()
            _flush_csv(args.out, rows)

    # Fill rows up to len(unique) with None so index alignment works.
    while len(rows) < len(unique):
        rows.append(None)

    # ---- REPAIR PASSES: retry unconverged configs with the now-richer cache ----
    for pass_idx in range(1, 6):
        if time.time() - t_start > BUDGET_S:
            print(f"\nbudget exhausted at start of pass-{pass_idx+1}")
            break
        fails = [i for i, r in enumerate(rows) if (r is None) or (not r["converged"])]
        if not fails:
            print(f"all {len(rows)} configs converged — no repair needed")
            break
        print(f"\n=== repair pass {pass_idx} — retry {len(fails)} non-converged configs (cache size {len(_CACHE)}) ===")
        sys.stdout.flush()
        n_repaired = 0
        for k, i in enumerate(fails, start=1):
            if time.time() - t_start > BUDGET_S:
                print(f"  budget exhausted mid-pass-{pass_idx} after {k-1} retries")
                break
            taus, gammas = unique[i]
            best = solve_with_ladder(taus, gammas)
            row = _evaluate(taus, gammas, u, i_r, j_r, l_r, best)
            if row and row["converged"]:
                rows[i] = row
                n_repaired += 1
                if row["oneR2_het"] > best_1mR2:
                    best_1mR2 = row["oneR2_het"]; best_tag = (taus, gammas, row["alpha"])
            elif row:
                # keep the better of old/new
                if rows[i] is None or (row["PhiI"] < (rows[i]["PhiI"] if rows[i] else float("inf"))):
                    rows[i] = row
            if k % 10 == 0:
                print(f"  [pass-{pass_idx} {k}/{len(fails)}] repaired {n_repaired}  "
                      f"best 1-R²_het={best_1mR2:.3e}  cache={len(_CACHE)}")
                sys.stdout.flush()
                _flush_csv(args.out, [r for r in rows if r is not None])
        _flush_csv(args.out, [r for r in rows if r is not None])
        if n_repaired == 0:
            print(f"  pass-{pass_idx}: 0 repaired — stopping repair loop")
            break

    _flush_csv(args.out, [r for r in rows if r is not None])
    n_final_conv = sum(1 for r in rows if r and r["converged"])
    print(f"\n=== search complete: {sum(1 for r in rows if r is not None)} tried, {n_final_conv} strictly converged ===")
    print(f"best 1-R²_het = {best_1mR2:.3e}  at {best_tag}")
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()
