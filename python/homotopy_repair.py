"""Homotopy/continuation repair for still-unsolved configs.

For each (τ_target, γ_target) that didn't converge via warm/IDW, we
construct a bridge path in log-parameter space from the nearest already-
converged config (τ₀, γ₀) to the target and solve intermediate points
one at a time, chaining warm-starts:

    s = 0 (τ₀, γ₀)  →  s = s_1, s_2, …, s_N = 1  (τ_target, γ_target)

At step k we warm-start Picard from the solution of step k-1. Intermediate
points don't have to be in the original candidate grid — they're pure
bridge nodes. If any intermediate fails to converge we adaptively bisect
the step and retry.

All bridge-solved intermediate REEs are also cached (they're valid REE
fixed points too) to strengthen future warm/IDW starts elsewhere in the
search.

Reads and writes to the same CSV as search_het.py. Only retries rows
with converged="0".
"""
from __future__ import annotations
import csv
import sys
import time
import numpy as np
import rezn_het as rh

CSV_IN  = "/home/user/REZN/python/search_het_results.csv"
CSV_OUT = "/home/user/REZN/python/search_het_results.csv"
G       = 9
UMAX    = 2.0
ABSTOL  = 1e-8
F_TOL   = 1.0
MAXITER = {1.0: 2000, 0.3: 5000, 0.1: 10000}
ALPHAS  = [1.0, 0.3, 0.1]
MIN_STEPS = 20
MAX_STEPS = 80
PERTURB_SIGMA = 0.02      # small Gaussian perturbation of the starting P in
                          # logit space (per cell) to break symmetry before
                          # the first bridge step.
BRIDGE_SUCCESS_DIST = 1.5   # don't bridge if nearest cached is already < this


# -------- Cache -----------------------------------------------------------
CACHE: list = []   # {"log_tg": ndarray(6), "P_star": ndarray, "taus", "gammas"}

def log_tg(t, g):
    return np.log(np.concatenate([np.asarray(t, float), np.asarray(g, float)]))

def nearest_cached(t, g):
    if not CACHE: return None, None, float("inf")
    q = log_tg(t, g)
    d = [float(np.linalg.norm(e["log_tg"] - q)) for e in CACHE]
    i = int(np.argmin(d))
    return CACHE[i]["taus"], CACHE[i]["gammas"], d[i]


def add_to_cache(t, g, P):
    CACHE.append({"log_tg": log_tg(t, g), "P_star": P.copy(),
                  "taus": t, "gammas": g})


# -------- Picard wrapper -------------------------------------------------

def picard(taus, gammas, P_init=None):
    """Try all alphas; return best (converged or lowest ‖Φ-I‖)."""
    best = None
    for alpha in ALPHAS:
        try:
            res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                                   maxiters=MAXITER[alpha], abstol=ABSTOL,
                                   alpha=alpha, P_init=P_init)
        except Exception:
            continue
        PhiI = res["history"][-1] if res["history"] else float("inf")
        Finf = float(np.abs(res["residual"]).max())
        converged = (PhiI < ABSTOL) and (Finf < F_TOL)
        cand = dict(alpha=alpha, iters=len(res["history"]),
                    PhiI=PhiI, Finf=Finf, P_star=res["P_star"],
                    converged=converged)
        if best is None \
           or (cand["converged"] and not best["converged"]) \
           or (not best["converged"] and cand["PhiI"] < best["PhiI"]):
            best = cand
        if cand["converged"]: return best
    return best


# -------- Bridge path ----------------------------------------------------

def bridge_path(t0, g0, t1, g1, n_steps):
    """Return list of (taus, gammas) along log-linear path from (t0,g0)
    to (t1,g1) inclusive. n_steps >= 2 (includes both endpoints)."""
    lt0 = np.log(np.asarray(t0, float)); lg0 = np.log(np.asarray(g0, float))
    lt1 = np.log(np.asarray(t1, float)); lg1 = np.log(np.asarray(g1, float))
    pts = []
    for k in range(n_steps):
        s = k / (n_steps - 1)
        t = tuple(np.exp((1-s)*lt0 + s*lt1))
        g = tuple(np.exp((1-s)*lg0 + s*lg1))
        pts.append((t, g))
    return pts


def solve_with_bridge(t_tgt, g_tgt):
    """Start from nearest cached, bridge in log-space, adaptive step size."""
    if not CACHE: return None, None
    t0, g0, d0 = nearest_cached(t_tgt, g_tgt)
    if d0 <= BRIDGE_SUCCESS_DIST:
        # nearest is close enough — already tried warm/IDW; bridge unlikely
        # to help. Still try with small N as fallback.
        pass
    # find initial P_prev: the cached P for (t0, g0)
    P_base = None
    for e in CACHE:
        if e["taus"] == t0 and e["gammas"] == g0:
            P_base = e["P_star"]; break
    if P_base is None: return None, None

    # adaptive: start with MIN_STEPS, double if any intermediate fails.
    # Re-perturb on each retry.
    n_steps = MIN_STEPS
    retry_idx = 0
    while n_steps <= MAX_STEPS:
        retry_idx += 1
        # perturb starting P in logit space with a per-retry seed
        rng = np.random.default_rng(
            int(abs(hash((t_tgt, g_tgt, n_steps, retry_idx))) % (2**32)))
        logit_P0 = np.log(P_base / (1.0 - P_base))
        noise = rng.normal(0.0, PERTURB_SIGMA, P_base.shape)
        P_start = 1.0 / (1.0 + np.exp(-(logit_P0 + noise)))
        P_start = np.clip(P_start, 1e-9, 1.0 - 1e-9)

        print(f"    bridge: {n_steps} steps from τ={t0} γ={g0}  "
              f"(perturb σ={PERTURB_SIGMA} retry#{retry_idx})")
        sys.stdout.flush()
        pts = bridge_path(t0, g0, t_tgt, g_tgt, n_steps)
        P_prev = P_start
        all_ok = True
        last_res = None
        for k, (t, g) in enumerate(pts):
            if k == 0: continue                 # skip starting point
            res = picard(t, g, P_init=P_prev)
            last_res = res
            if res is None:
                all_ok = False; break
            if not res["converged"]:
                all_ok = False
                print(f"    bridge failed at step {k}/{n_steps-1}  "
                      f"τ={t} γ={g}  PhiI={res['PhiI']:.2e}")
                sys.stdout.flush()
                break
            P_prev = res["P_star"]
            if k < n_steps - 1:
                add_to_cache(t, g, P_prev)
        if all_ok:
            return last_res, "bridge"
        n_steps *= 2
    return last_res, "bridge_failed"


# -------- Main ------------------------------------------------------------

def main():
    _ = rh.solve_picard(5, 2.0, 0.5, maxiters=3)

    with open(CSV_IN, "r") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    # Rebuild cache from currently-converged rows
    for r in rows:
        if r["converged"] != "1": continue
        t = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
        g = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
        try:
            res = rh.solve_picard(G, t, g, umax=UMAX, maxiters=3000,
                                   abstol=ABSTOL, alpha=1.0)
            PhiI = res["history"][-1] if res["history"] else float("inf")
            Finf = float(np.abs(res["residual"]).max())
            if PhiI < ABSTOL and Finf < F_TOL:
                add_to_cache(t, g, res["P_star"])
        except Exception: pass
    print(f"[init] cache = {len(CACHE)} configs")
    sys.stdout.flush()

    u = rh.build_grid(G, UMAX)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(1.0), idx(-1.0), idx(1.0)

    unsolved = [r for r in rows if r["converged"] != "1"]
    print(f"[bridge] retrying {len(unsolved)} unsolved configs with homotopy")
    sys.stdout.flush()

    n_new = 0
    t_start = time.time()
    for k, r in enumerate(unsolved, start=1):
        t = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
        g = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
        print(f"[{k}/{len(unsolved)}] target τ={t} γ={g}")
        sys.stdout.flush()
        t0 = time.time()
        res, status = solve_with_bridge(t, g)
        dt = time.time() - t0
        if res and res["converged"]:
            P = res["P_star"]
            mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r,j_r,l_r]), P, u,
                                   np.asarray(t))
            taus_a = np.asarray(t, float); gammas_a = np.asarray(g, float)
            wn = (1.0 / gammas_a) / (1.0 / gammas_a).sum()
            coef = wn * taus_a
            y = np.log(P / (1 - P)).reshape(-1)
            T = np.empty(G**3); kk = 0
            for i in range(G):
                for j in range(G):
                    for l in range(G):
                        T[kk] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
                        kk += 1
            yc = y - y.mean(); Tc = T - T.mean()
            one_het = 1 - (yc*Tc).sum()**2 / ((yc*yc).sum()*(Tc*Tc).sum())
            one_eq  = rh.one_minus_R2(P, u, t)
            r.update({
                "alpha": f"{res['alpha']}",
                "iters": res["iters"],
                "time_s": f"{dt:.2f}",
                "PhiI": f"{res['PhiI']:.3e}",
                "Finf": f"{res['Finf']:.3e}",
                "oneR2_het": f"{one_het:.6e}",
                "oneR2_eq":  f"{one_eq:.6e}",
                "p_star":  f"{float(P[i_r,j_r,l_r]):.10f}",
                "mu_1": f"{mu[0]:.8f}", "mu_2": f"{mu[1]:.8f}", "mu_3": f"{mu[2]:.8f}",
                "pr_gap": f"{mu[0]-mu[1]:.6f}",
                "converged": "1", "init": status,
            })
            add_to_cache(t, g, P)
            n_new += 1
            print(f"    → bridged OK  1-R²_het={one_het:.3e}  new converged = {n_new}")
            sys.stdout.flush()
        else:
            print(f"    → still unsolved  t={dt:.1f}s")
            sys.stdout.flush()

        # flush CSV
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rr in rows: w.writerow(rr)

    conv_total = sum(1 for r in rows if r["converged"] == "1")
    print(f"\n[done] bridged {n_new} configs in {(time.time()-t_start)/60:.1f}min. "
          f"total converged = {conv_total}/{len(rows)}")


if __name__ == "__main__":
    main()
