"""Repair loop: retry ONLY the unconverged configs from search_het_results.csv
using warm-start / IDW interpolation from the ever-growing set of converged
configs. Iterates until a pass makes zero new conversions, then reports and
writes the updated CSV.

Strategy per unsolved config:
  1. Re-compute nearest cached point, warm-start with it. If successful, done.
  2. Compute IDW interpolation from the k nearest cached points (log-param
     space, inverse-distance-squared weighting in logit(P)). Try each
     damping level α ∈ {1.0, 0.3, 0.1}.
  3. If still unsolved, mark and move on.

After every full pass, any new converges are added to the cache and the
repair loop runs again. Continues until convergence-count doesn't grow.

No cold starts in this script — purely interpolation-based.
"""
from __future__ import annotations
import csv
import os
import sys
import time
import numpy as np
import rezn_het as rh

CSV_IN  = "/home/user/REZN/python/search_het_results.csv"
CSV_OUT = "/home/user/REZN/python/search_het_results.csv"   # in-place
G       = 9
UMAX    = 2.0
ABSTOL  = 1e-8
F_TOL   = 1.0
MAXITER_BY_ALPHA = {1.0: 1500, 0.3: 4000, 0.1: 8000}
ALPHA_LADDER = [1.0, 0.3, 0.1]
MAX_CACHE_DIST = 4.0        # log-param L2
IDW_K = 10                  # nearest neighbours for IDW


# -------- Cache -----------------------------------------------------------
# List of {"log_tg": ndarray(6), "P_star": ndarray, "one_R2_het": float,
# "taus", "gammas"}.
CACHE: list = []


def log_tg(taus, gammas):
    return np.log(np.concatenate([np.asarray(taus, float),
                                   np.asarray(gammas, float)]))


def nearest_cached(taus, gammas):
    if not CACHE: return None
    q = log_tg(taus, gammas)
    d = [float(np.linalg.norm(e["log_tg"] - q)) for e in CACHE]
    i = int(np.argmin(d))
    return CACHE[i]["P_star"] if d[i] <= MAX_CACHE_DIST else None


def idw_interp(taus, gammas, k=IDW_K, power=2.0):
    if len(CACHE) < 2: return None
    q = log_tg(taus, gammas)
    d = np.array([float(np.linalg.norm(e["log_tg"] - q)) for e in CACHE])
    keep = np.where(d <= MAX_CACHE_DIST)[0]
    if keep.size < 2: return None
    order = keep[np.argsort(d[keep])][:k]
    dw = d[order]
    w = 1.0 / (dw**power + 1e-12); w /= w.sum()
    logit_sum = np.zeros_like(CACHE[0]["P_star"])
    for idx, wi in zip(order, w):
        P = CACHE[idx]["P_star"]
        logit_sum += wi * np.log(P / (1 - P))
    P_init = 1.0 / (1.0 + np.exp(-logit_sum))
    return np.clip(P_init, 1e-9, 1 - 1e-9)


# -------- Solver wrapper --------------------------------------------------

def try_solve(taus, gammas):
    """Try warm→IDW across all alphas. Return best result dict."""
    best = None
    attempts = []
    P_warm = nearest_cached(taus, gammas)
    if P_warm is not None:
        attempts.append(("warm", P_warm))
    P_idw = idw_interp(taus, gammas)
    if P_idw is not None:
        attempts.append(("idw", P_idw))

    for init_tag, P_init in attempts:
        for alpha in ALPHA_LADDER:
            mi = MAXITER_BY_ALPHA[alpha]
            t0 = time.time()
            try:
                res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                                       maxiters=mi, abstol=ABSTOL, alpha=alpha,
                                       P_init=P_init)
            except Exception as e:
                continue
            dt = time.time() - t0
            PhiI = res["history"][-1] if res["history"] else float("inf")
            Finf = float(np.abs(res["residual"]).max())
            converged = (PhiI < ABSTOL) and (Finf < F_TOL)
            cand = dict(alpha=alpha, iters=len(res["history"]), time=dt,
                        PhiI=PhiI, Finf=Finf, P_star=res["P_star"],
                        converged=converged, init=init_tag)
            if best is None or (cand["converged"] and not best["converged"]) \
               or (cand["converged"] and best["converged"] and cand["time"] < best["time"]) \
               or (not best["converged"] and cand["PhiI"] < best["PhiI"]):
                best = cand
            if cand["converged"]:
                return best
    return best


# -------- 1-R² helper -----------------------------------------------------

def one_minus_R2_het(Pg, u, taus, gammas):
    G = u.shape[0]
    taus = np.asarray(taus, float); gammas = np.asarray(gammas, float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1 - Pg)).reshape(-1)
    T = np.empty(G**3); k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T[k] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]; k += 1
    y_c = y - y.mean(); T_c = T - T.mean()
    Syy = float((y_c*y_c).sum()); STT = float((T_c*T_c).sum()); SyT = float((y_c*T_c).sum())
    return 1 - (SyT*SyT)/(Syy*STT) if (Syy > 0 and STT > 0) else 0.0


# -------- Main ------------------------------------------------------------

def main():
    # Warmup
    _ = rh.solve_picard(5, 2.0, 0.5, maxiters=3)

    # Load CSV
    with open(CSV_IN, "r") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    u = rh.build_grid(G, UMAX)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(1.0), idx(-1.0), idx(1.0)

    # Populate cache from current converged rows (re-solve cold quickly)
    print(f"[init] loading {sum(1 for r in rows if r['converged']=='1')} converged configs into cache")
    sys.stdout.flush()
    for r in rows:
        if r["converged"] != "1": continue
        t = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
        g = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
        try:
            res = rh.solve_picard(G, t, g, umax=UMAX, maxiters=3000, abstol=ABSTOL, alpha=1.0)
            PhiI = res["history"][-1] if res["history"] else float("inf")
            Finf = float(np.abs(res["residual"]).max())
            if PhiI < ABSTOL and Finf < F_TOL:
                CACHE.append({"log_tg": log_tg(t, g), "P_star": res["P_star"].copy(),
                              "taus": t, "gammas": g})
        except Exception:
            continue
    print(f"[init] cache size = {len(CACHE)}")
    sys.stdout.flush()

    pass_idx = 0
    while True:
        pass_idx += 1
        unsolved = [r for r in rows if r["converged"] != "1"]
        if not unsolved:
            print(f"[pass {pass_idx}] all {len(rows)} configs converged — done")
            break
        print(f"\n[pass {pass_idx}] {len(unsolved)} unsolved configs, cache size = {len(CACHE)}")
        sys.stdout.flush()

        n_new = 0
        for k, r in enumerate(unsolved, start=1):
            t = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
            g = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
            best = try_solve(t, g)
            if best is None: continue
            P = best["P_star"]
            if best["converged"] and P is not None:
                mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r, j_r, l_r]), P, u,
                                       np.asarray(t))
                one_het = one_minus_R2_het(P, u, t, g)
                one_eq  = rh.one_minus_R2(P, u, t)
                r.update({
                    "alpha": f"{best['alpha']}",
                    "iters": best["iters"],
                    "time_s": f"{best['time']:.2f}",
                    "PhiI": f"{best['PhiI']:.3e}",
                    "Finf": f"{best['Finf']:.3e}",
                    "oneR2_het": f"{one_het:.6e}",
                    "oneR2_eq":  f"{one_eq:.6e}",
                    "p_star": f"{float(P[i_r, j_r, l_r]):.10f}",
                    "mu_1": f"{mu[0]:.8f}",
                    "mu_2": f"{mu[1]:.8f}",
                    "mu_3": f"{mu[2]:.8f}",
                    "pr_gap": f"{mu[0]-mu[1]:.6f}",
                    "converged": "1",
                    "init": best["init"],
                })
                CACHE.append({"log_tg": log_tg(t, g), "P_star": P.copy(),
                              "taus": t, "gammas": g})
                n_new += 1
                if n_new % 5 == 0:
                    print(f"  [pass {pass_idx}] +{n_new} new  latest τ={t} γ={g} "
                          f"1-R²={one_het:.3e}")
                    sys.stdout.flush()

            # flush CSV periodically
            if k % 15 == 0:
                with open(CSV_OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for rr in rows: w.writerow(rr)

        # final flush per pass
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rr in rows: w.writerow(rr)

        conv_total = sum(1 for r in rows if r["converged"] == "1")
        print(f"[pass {pass_idx}] +{n_new} repaired this pass. "
              f"total converged = {conv_total}/{len(rows)}")
        sys.stdout.flush()
        if n_new == 0:
            print(f"[done] no new conversions in pass {pass_idx}; stopping")
            break

    # Summary
    conv = [r for r in rows if r["converged"] == "1"]
    conv.sort(key=lambda r: -float(r["oneR2_het"]))
    print(f"\n=== Final: {len(conv)}/{len(rows)} converged ===")
    print(f"{'1-R²_het':>10}  {'τ':<22} {'γ':<22} {'init':<6} {'PhiI':>10}")
    for r in conv[:20]:
        print(f"{float(r['oneR2_het']):>10.3e}  "
              f"({r['tau_1']},{r['tau_2']},{r['tau_3']})".ljust(35)
              + f"  ({r['gamma_1']},{r['gamma_2']},{r['gamma_3']})".ljust(25)
              + f"{r['init']:<6} {r['PhiI']:>10}")


if __name__ == "__main__":
    main()
