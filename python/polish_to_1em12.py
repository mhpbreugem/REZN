"""Polish the cached G=11 UMAX=2.0 P-tensors to Finf < 1e-12.

Loads pchip_G11u20_cache.pkl, runs analytic Newton with tight lgmres
settings on each entry, writes pchip_G11u20_polished.csv with the new
Finf values. Skips entries that already meet the target.
"""
from __future__ import annotations
import csv
import os
import pickle
import sys
import time
import numpy as np

import rezn_het as rh
import rezn_pchip as rp
import pchip_jacobian as pj


G = 11
UMAX = 2.0
TARGET = 1e-12
LGMRES_TOL = 1e-10
LGMRES_MAXITER = 200
NEWTON_MAXITERS = 15

CACHE_PKL = "/home/user/REZN/python/pchip_G11u20_cache.pkl"
CSV_OUT   = "/home/user/REZN/python/pchip_G11u20_polished.csv"
STATUS    = "/home/user/REZN/python/polish_status.txt"


def main():
    if not os.path.exists(CACHE_PKL):
        print(f"missing {CACHE_PKL}", flush=True); return

    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {len(cache)} cache entries", flush=True)

    fields = ["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
              "newton_iters","time_s","Finf_in","Finf_out","oneR2_het",
              "p_star","mu_1","mu_2","mu_3","pr_gap","converged"]
    rows = []

    u = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])
    idx_r = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx_r(1.0), idx_r(-1.0), idx_r(1.0)

    for k, e in enumerate(cache):
        taus = np.asarray(e["taus"], float)
        gammas = np.asarray(e["gammas"], float)
        P0 = np.clip(e["P_star"], 1e-9, 1 - 1e-9)
        Phi0 = rp._phi_map_pchip(P0, u, taus, gammas, Ws)
        Finf_in = float(np.abs(P0 - Phi0).max())

        prefix = f"τ={tuple(taus)} γ={tuple(gammas)}"
        with open(STATUS, "w") as sf:
            sf.write(f"[{k+1}/{len(cache)}] {prefix} Finf_in={Finf_in:.3e}\n")
        print(f"[{k+1}/{len(cache)}] {prefix} Finf_in={Finf_in:.3e}",
              flush=True)

        if Finf_in < TARGET:
            P = P0
            Finf_out = Finf_in
            iters = 0
            dt = 0.0
            print(f"  already meets target", flush=True)
        else:
            t0 = time.time()
            res = pj.solve_newton_analytic(
                G, taus, gammas, umax=UMAX,
                P_init=P0, maxiters=NEWTON_MAXITERS,
                abstol=TARGET,
                lgmres_tol=LGMRES_TOL, lgmres_maxiter=LGMRES_MAXITER,
                status_path=STATUS, status_every=1,
                status_prefix=prefix)
            dt = time.time() - t0
            P = res["P_star"]
            Finf_out = res["best_Finf"]
            iters = len(res["history"])
            print(f"  Newton: {iters} iters, {dt:.1f}s, "
                  f"Finf_out={Finf_out:.3e}", flush=True)

        try:
            mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r,j_r,l_r]),
                                    P, u, taus)
            T_grid = np.empty(G**3); ki = 0
            for i in range(G):
                for j in range(G):
                    for l in range(G):
                        T_grid[ki] = (taus[0]*u[i] + taus[1]*u[j]
                                       + taus[2]*u[l]); ki += 1
            y = np.log(P / (1 - P)).reshape(-1)
            yc = y - y.mean(); Tc = T_grid - T_grid.mean()
            Syy = float((yc*yc).sum()); STT = float((Tc*Tc).sum())
            SyT = float((yc*Tc).sum())
            R2 = (SyT*SyT)/(Syy*STT) if Syy*STT > 0 else 1.0
            one_het = 1.0 - R2
        except Exception as ex:
            mu = (float("nan"),)*3; one_het = float("nan")

        rows.append({
            "tau_1": float(taus[0]), "tau_2": float(taus[1]),
            "tau_3": float(taus[2]),
            "gamma_1": float(gammas[0]), "gamma_2": float(gammas[1]),
            "gamma_3": float(gammas[2]),
            "newton_iters": iters,
            "time_s": f"{dt:.2f}",
            "Finf_in": f"{Finf_in:.3e}",
            "Finf_out": f"{Finf_out:.3e}",
            "oneR2_het": f"{one_het:.6e}",
            "p_star": f"{float(P[i_r,j_r,l_r]):.10f}",
            "mu_1": f"{mu[0]:.10f}", "mu_2": f"{mu[1]:.10f}",
            "mu_3": f"{mu[2]:.10f}",
            "pr_gap": f"{mu[0]-mu[1]:.10f}",
            "converged": int(Finf_out < TARGET),
        })

        # Write incrementally so progress is preserved on interrupt
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rr in rows: w.writerow(rr)

    print("\n=== summary ===", flush=True)
    print(f"target Finf < {TARGET:.0e}", flush=True)
    n_pass = sum(1 for r in rows if int(r["converged"]) == 1)
    print(f"passed: {n_pass}/{len(rows)}", flush=True)
    for r in rows:
        print(f"  τ={r['tau_1']:.2f} γ={r['gamma_1']:.1f}  "
              f"Finf {r['Finf_in']} → {r['Finf_out']}  "
              f"({r['newton_iters']} iters, {r['time_s']}s)",
              flush=True)


if __name__ == "__main__":
    main()
