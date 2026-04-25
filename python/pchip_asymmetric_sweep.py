"""Quick τ sweep for asymmetric γ=(5, 3, 1) at G=11 PCHIP.

Complements the homogeneous γ=(3,3,3) forward sweep running in parallel.
Uses SEPARATE status file, CSV, and CACHE pickle so it doesn't conflict.

Start at τ=2.0 (near-CARA regime, Picard should converge from cold start),
warm-chain τ upward in 0.1 steps to τ=5.0.
"""
from __future__ import annotations
import csv
import os
import pickle
import sys
import time

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

import rezn_het as rh
import rezn_pchip as rp


G           = 11
UMAX        = 2.5
GAMMAS_ASYM = (5.0, 3.0, 1.0)
ABSTOL      = 1e-4
F_TOL       = 3e-3

TAU_LO      = 0.5
TAU_HI      = 5.0
TAU_STEP    = 0.5              # 10 points

CSV_OUT     = "/home/user/REZN/python/pchip_asymmetric_results.csv"
STATUS_PATH = "/home/user/REZN/python/sweep_asym_status.txt"
CACHE_PKL   = "/home/user/REZN/python/pchip_asym_cache.pkl"


FIELDS = ["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
          "alpha","iters","time_s","PhiI","Finf","oneR2_het",
          "p_star","mu_1","mu_2","mu_3","pr_gap","converged","init"]


def _one_minus_R2_het(Pg, u, taus, gammas):
    taus = np.asarray(taus, float); gammas = np.asarray(gammas, float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1 - Pg)).reshape(-1)
    G_ = u.shape[0]
    T = np.empty(G_*G_*G_)
    it = 0
    for i in range(G_):
        for j in range(G_):
            for l in range(G_):
                T[it] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
                it += 1
    yc = y - y.mean(); Tc = T - T.mean()
    num = (yc*Tc).sum()**2
    den = (yc*yc).sum() * (Tc*Tc).sum()
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(1.0 - num/den)


def _solve_nk(taus, gammas, P_init):
    u = np.linspace(-UMAX, UMAX, G)
    taus_v = np.asarray(taus, float)
    gammas_v = np.asarray(gammas, float)
    Ws = rh._as_vec3(1.0)

    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus_v, gammas_v, Ws)
        return x - Pn.reshape(-1)

    x0 = np.clip(P_init, 1e-9, 1-1e-9).reshape(-1)
    try:
        sol = newton_krylov(F, x0, f_tol=max(ABSTOL, 1e-8),
                            rdiff=1e-7, method="lgmres",
                            maxiter=20, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
    if not np.all(np.isfinite(sol)):
        sol = x0
    P_star = np.clip(sol.reshape(G, G, G), 1e-9, 1 - 1e-9)
    Pn = rp._phi_map_pchip(P_star, u, taus_v, gammas_v, Ws)
    return P_star, float(np.abs(P_star - Pn).max())


def _solve_one(taus, gammas, P_warm):
    """P1.0 (short) → NK → Anderson → P0.3 (short)."""
    prefix = (f"τ=({taus[0]:.2f},{taus[1]:.2f},{taus[2]:.2f}) "
              f"γ=({gammas[0]:.2f},{gammas[1]:.2f},{gammas[2]:.2f})")
    attempts = [
        ("P1.0", dict(kind="picard", alpha=1.0, maxiters=400)),
        ("NK",   dict(kind="nk")),
        ("A6",   dict(kind="anderson", m=6, maxiters=300)),
        ("A10",  dict(kind="anderson", m=10, maxiters=300)),
        ("P0.3", dict(kind="picard", alpha=0.3, maxiters=1000)),
    ]
    best = None
    for tag, opts in attempts:
        t0 = time.time()
        try:
            if opts["kind"] == "picard":
                res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                             maxiters=opts["maxiters"],
                                             abstol=ABSTOL, alpha=opts["alpha"],
                                             P_init=P_warm,
                                             status_path=STATUS_PATH,
                                             status_every=25,
                                             status_prefix=f"{prefix} | {tag}")
                P_star = res["P_star"]
                Finf = float(np.abs(res["residual"]).max())
                iters = len(res["history"])
            elif opts["kind"] == "anderson":
                res = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                               maxiters=opts["maxiters"],
                                               abstol=ABSTOL,
                                               m_window=opts["m"], damping=1.0,
                                               P_init=P_warm,
                                               status_path=STATUS_PATH,
                                               status_every=25,
                                               status_prefix=f"{prefix} | {tag}")
                P_star = res["P_star"]
                Finf = float(np.abs(res["residual"]).max())
                iters = len(res["history"])
            else:
                with open(STATUS_PATH, "w") as _sf:
                    _sf.write(f"{prefix} | {tag} (newton-krylov)\n")
                P_star, Finf = _solve_nk(taus, gammas, P_warm)
                iters = 1
        except Exception as e:
            print(f"    {tag} error: {e}")
            continue
        if not (np.isfinite(Finf) and np.all(np.isfinite(P_star))):
            Finf = float("inf")
        dt = time.time() - t0
        converged = Finf < F_TOL
        cand = dict(alpha=tag, iters=iters, time=dt, Finf=Finf,
                    P_star=P_star, converged=converged)
        if best is None or (cand["converged"] and not best["converged"]) \
           or (not best["converged"] and cand["Finf"] < best["Finf"]):
            best = cand
        if converged:
            break
    return best


def main():
    print(f"Asymmetric γ={GAMMAS_ASYM} τ sweep τ∈[{TAU_LO}, {TAU_HI}] step={TAU_STEP}")
    sys.stdout.flush()

    # Warmup
    print("numba warmup…"); sys.stdout.flush()
    _ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

    u = np.linspace(-UMAX, UMAX, G)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    ir, jr, lr = idx(1.0), idx(-1.0), idx(1.0)

    # Load previous progress if any
    rows = []
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT) as f:
            for r in csv.DictReader(f):
                try:
                    for k in ("tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3"):
                        r[k] = float(r[k])
                    if int(r["converged"]) == 1 and float(r["Finf"]) <= F_TOL:
                        rows.append(r)
                except Exception:
                    continue
        print(f"[preload] {len(rows)} previously-converged rows")
        sys.stdout.flush()

    cache = []
    if os.path.exists(CACHE_PKL):
        try:
            with open(CACHE_PKL, "rb") as f:
                cache = pickle.load(f)
            print(f"[cache] {len(cache)} CACHE entries from pickle")
            sys.stdout.flush()
        except Exception:
            pass

    def flush_csv():
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows: w.writerow(r)

    def save_cache():
        try:
            with open(CACHE_PKL, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass

    # Walk τ upward from TAU_LO
    taus_list = np.arange(TAU_LO, TAU_HI + 1e-9, TAU_STEP)
    P_cur = None
    if cache:
        # Find the lowest-τ entry to seed from
        cache_sorted = sorted(cache, key=lambda e: e["taus"][0])
        P_cur = cache_sorted[0]["P_star"]

    for tau in taus_list:
        t = (float(tau), float(tau), float(tau))  # homogeneous τ; heterogeneous γ
        g = GAMMAS_ASYM
        # Skip if already done
        already = any(abs(r["tau_1"] - t[0]) < 1e-9 for r in rows)
        if already:
            # Use its P if cached
            match = [e for e in cache if abs(e["taus"][0] - t[0]) < 1e-9]
            if match:
                P_cur = match[0]["P_star"]
            continue

        best = _solve_one(t, g, P_cur)
        if best is None:
            print(f"  τ={tau:.2f}  SOLVER CRASH")
            continue

        P_star = best["P_star"]
        Finf = best["Finf"]
        one_het = _one_minus_R2_het(P_star, u, t, g)
        try:
            mu = rh.posteriors_at(ir, jr, lr, float(P_star[ir,jr,lr]),
                                   P_star, u, np.asarray(t))
        except Exception:
            mu = (float("nan"),)*3

        rows.append({
            "tau_1": t[0], "tau_2": t[1], "tau_3": t[2],
            "gamma_1": g[0], "gamma_2": g[1], "gamma_3": g[2],
            "alpha": best["alpha"],
            "iters": best["iters"],
            "time_s": f"{best['time']:.2f}",
            "PhiI": f"{Finf:.3e}",
            "Finf": f"{Finf:.3e}",
            "oneR2_het": f"{one_het:.6e}",
            "p_star": f"{float(P_star[ir,jr,lr]):.10f}",
            "mu_1": f"{mu[0]:.8f}", "mu_2": f"{mu[1]:.8f}", "mu_3": f"{mu[2]:.8f}",
            "pr_gap": f"{mu[0]-mu[1]:.6f}",
            "converged": int(best["converged"]),
            "init": "warm" if P_cur is not None else "cold",
        })
        flush_csv()

        if best["converged"]:
            P_cur = P_star
            cache.append({"log_tg": np.log(np.concatenate([np.asarray(t, float),
                                                          np.asarray(g, float)])),
                          "P_star": P_star.copy(), "taus": t, "gammas": g})
            save_cache()

        print(f"  τ={tau:.2f}  α={best['alpha']:<5}  iters={best['iters']:<5}  "
              f"Finf={Finf:.2e}  1-R²={one_het:.3e}  conv={int(best['converged'])}  "
              f"time={best['time']:.1f}s")
        sys.stdout.flush()

    print(f"\n[done] {len(rows)} rows → {CSV_OUT}")


if __name__ == "__main__":
    main()
