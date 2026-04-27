"""Strict PR search batch 2: heterogeneous + extreme parameters.

Same strict tolerance as batch 1: only declare PR if Finf < 1e-12.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CONFIGS = [
    # G,  τ_vec,                 γ_vec,                    label
    # extreme heterogeneity per HANDOFF
    (11, (1.0, 3.0, 10.0),      (0.3, 3.0, 30.0),        "extreme γ=(0.3,3,30) τ=(1,3,10)"),
    (11, (0.5, 3.0, 10.0),      (0.3, 3.0, 10.0),        "spec extreme"),
    (11, (10.0, 3.0, 1.0),      (1.0, 3.0, 10.0),        "aligned (10,3,1)"),
    (11, (1.0, 3.0, 10.0),      (10.0, 3.0, 1.0),        "anti-aligned"),
    # γ=1 log utility
    (11, (2.0, 2.0, 2.0),       (1.0, 1.0, 1.0),         "γ=1 τ=2"),
    (11, (5.0, 5.0, 5.0),       (1.0, 1.0, 1.0),         "γ=1 τ=5"),
    # very low γ
    (11, (2.0, 2.0, 2.0),       (0.1, 0.1, 0.1),         "γ=0.1 τ=2"),
    (11, (1.0, 1.0, 1.0),       (0.1, 0.1, 0.1),         "γ=0.1 τ=1"),
    # different G's
    ( 9, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=9 γ=0.5 τ=2"),
    (13, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=13 γ=0.5 τ=2"),
    # mid τ region
    (11, (4.0, 4.0, 4.0),       (3.0, 3.0, 3.0),         "γ=3 τ=4"),
    (11, (2.5, 2.5, 2.5),       (3.0, 3.0, 3.0),         "γ=3 τ=2.5"),
]
UMAX = 2.0
TARGET = 1e-12

print(f"{'cfg':35s}  G   {'p':>10s}  {'1-R²':>11s}  {'PR-gap':>9s}  "
      f"{'Finf':>11s}  {'time':>6s}  status", flush=True)

for G, taus_t, gammas_t, label in CONFIGS:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)
    t0 = time.time()
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=200, abstol=TARGET, alpha=0.3)
    P_warm = res_p["P_star"]
    try:
        res_n = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_warm, maxiters=12, abstol=TARGET,
            lgmres_tol=1e-12, lgmres_maxiter=80)
        P = res_n["P_star"]; Finf = res_n["best_Finf"]
    except Exception:
        P = P_warm; Finf = float(np.abs(res_p["residual"]).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[ir, jr, lr])
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr_gap = mu[0] - mu[1]
    if Finf > TARGET:
        status = "no fp at tol"
    elif one_r2 > 0.01:
        status = "PR!"
    else:
        status = "FR (machine)"
    print(f"{label:35s}  {G:2d}  {p:10.4f}  {one_r2:11.3e}  "
          f"{pr_gap:+9.4f}  {Finf:11.3e}  {time.time()-t0:6.1f}s  {status}",
          flush=True)
