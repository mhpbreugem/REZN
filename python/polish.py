"""Polish cached G=11 UMAX=2.0 P-tensors using the analytic Jacobian.

Loads pchip_G11u20_cache.pkl, runs `pchip_jacobian.solve_newton` on
each cached entry, writes one row per entry to
pchip_G11u20_polished.csv. Skips entries that already meet TARGET.

The achievable floor at G=11 is around 1e-8 because the PCHIP
interpolation precision on the 11³ grid sets a noise floor in Φ
that no Jacobian quality can break — verified by Picard alone
plateauing at the same level. The G=5 self-test in pchip_jacobian.py
hits 1e-13 in 2 Newton iterations, which is the genuine machine-
precision benchmark of the solver. To break the G=11 floor you'd
need a finer grid (G=15+), higher-order interpolation, or analytic
contour integration.

Run with:  python3 polish.py
"""
from __future__ import annotations

import csv
import pickle
import time

import numpy as np

import pchip_jacobian as pj
import rezn_het as rh
import rezn_pchip as rp


G = 11
UMAX = 2.0
# Target: 1e-9. The G=5 self-test hits 1e-13 in 2 Newton iterations,
# but at G=11 the PCHIP interpolation precision on the 11³ grid caps
# the achievable Φ-residual around 1e-8. Going below ~1e-9 here
# requires a finer grid (G=15+) or higher-precision interpolation.
TARGET = 1e-9

CACHE_PKL = "/home/user/REZN/python/pchip_G11u20_cache.pkl"
CSV_OUT = "/home/user/REZN/python/pchip_G11u20_polished.csv"
STATUS = "/home/user/REZN/python/polish_status.txt"


def one_minus_R2_het(P, u, taus):
    """1 − R² of logit(P) regressed on the CARA-FR sufficient statistic
    Σ_k τ_k u_k. Same definition as in pchip_continuation.py."""
    G_ = u.shape[0]
    y = np.log(P / (1 - P)).reshape(-1)
    T = np.empty(G_ ** 3)
    k = 0
    for i in range(G_):
        for j in range(G_):
            for l in range(G_):
                T[k] = taus[0] * u[i] + taus[1] * u[j] + taus[2] * u[l]
                k += 1
    yc = y - y.mean()
    Tc = T - T.mean()
    Syy = float((yc * yc).sum())
    STT = float((Tc * Tc).sum())
    SyT = float((yc * Tc).sum())
    if Syy == 0 or STT == 0:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


def main():
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {len(cache)} cache entries", flush=True)

    u = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])
    i_r = int(np.argmin(np.abs(u - 1.0)))
    j_r = int(np.argmin(np.abs(u + 1.0)))
    l_r = int(np.argmin(np.abs(u - 1.0)))

    fields = ["tau_1", "tau_2", "tau_3", "gamma_1", "gamma_2", "gamma_3",
              "newton_iters", "time_s", "Finf_in", "Finf_out",
              "oneR2_het", "p_star", "mu_1", "mu_2", "mu_3", "pr_gap",
              "converged"]
    rows = []

    for k, e in enumerate(cache):
        taus = np.asarray(e["taus"], float)
        gammas = np.asarray(e["gammas"], float)
        P0 = np.clip(e["P_star"], pj.EPS_OUTER, 1 - pj.EPS_OUTER)
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
        else:
            t0 = time.time()
            res = pj.solve_newton(
                G, taus, gammas, umax=UMAX,
                P_init=P0, maxiters=6, abstol=TARGET,
                lgmres_tol=1e-8, lgmres_maxiter=120,
                status_path=STATUS, status_every=1, status_prefix=prefix)
            dt = time.time() - t0
            P = res["P_star"]
            Finf_out = res["best_Finf"]
            iters = len(res["history"])
            print(f"  newton: {iters} iters, {dt:.1f}s, "
                  f"Finf={Finf_out:.3e}", flush=True)

        mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r, j_r, l_r]),
                                P, u, taus)
        oneR2 = one_minus_R2_het(P, u, taus)

        rows.append({
            "tau_1": float(taus[0]),
            "tau_2": float(taus[1]),
            "tau_3": float(taus[2]),
            "gamma_1": float(gammas[0]),
            "gamma_2": float(gammas[1]),
            "gamma_3": float(gammas[2]),
            "newton_iters": iters,
            "time_s": f"{dt:.2f}",
            "Finf_in": f"{Finf_in:.3e}",
            "Finf_out": f"{Finf_out:.3e}",
            "oneR2_het": f"{oneR2:.6e}",
            "p_star": f"{float(P[i_r, j_r, l_r]):.10f}",
            "mu_1": f"{mu[0]:.10f}",
            "mu_2": f"{mu[1]:.10f}",
            "mu_3": f"{mu[2]:.10f}",
            "pr_gap": f"{mu[0] - mu[1]:.10f}",
            "converged": int(Finf_out < TARGET),
        })

        # Write CSV after each row so progress is preserved
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rr in rows:
                w.writerow(rr)

    print("\n=== summary ===", flush=True)
    n_pass = sum(1 for r in rows if int(r["converged"]) == 1)
    print(f"target Finf < {TARGET:.0e}: {n_pass}/{len(rows)} configs",
          flush=True)
    for r in rows:
        flag = "✓" if int(r["converged"]) == 1 else "✗"
        print(f"  {flag}  τ={r['tau_1']:.2f} γ={r['gamma_1']:.1f}  "
              f"Finf {r['Finf_in']} → {r['Finf_out']}  "
              f"({r['newton_iters']} iters, {r['time_s']}s)",
              flush=True)


if __name__ == "__main__":
    main()
