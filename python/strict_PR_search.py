"""Strict-tolerance PR search.

Each config: cold Picard burn-in then analytic-Jacobian Newton with
abstol=1e-12. Only declare PR if Finf < 1e-12 AND 1-R² > 0.01.
Anything looser is a transient or numerical noise — does not count
as a fixed point.

Configs span homogeneous CRRA across (γ, τ), heterogeneous γ, and
several G values. If any config delivers a *machine-precision* PR
fixed point, we have a real PR branch and can use that P as a seed
for the figures.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CONFIGS = [
    # G,  τ_vec,                 γ_vec,                    label
    (11, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "homog γ=0.5, τ=2"),
    (11, (3.4, 3.4, 3.4),       (3.0, 3.0, 3.0),         "homog γ=3, τ=3.4"),
    (11, (5.0, 5.0, 5.0),       (1.0, 1.0, 1.0),         "homog γ=1, τ=5"),
    (11, (2.0, 2.0, 2.0),       (5.0, 3.0, 1.0),         "het γ=(5,3,1), τ=2"),
    (11, (2.0, 2.0, 2.0),       (1.0, 3.0, 5.0),         "het γ=(1,3,5), τ=2"),
    ( 7, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=7 γ=0.5 τ=2"),
    ( 5, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=5 γ=0.5 τ=2"),
    (15, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=15 γ=0.5 τ=2"),
]
UMAX = 2.0
TARGET = 1e-12

print(f"{'cfg':30s}  G   {'p':>10s}  {'1-R²':>11s}  {'PR-gap':>9s}  "
      f"{'Finf':>11s}  {'time':>6s}  status", flush=True)

for G, taus_t, gammas_t, label in CONFIGS:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)
    t0 = time.time()
    # Picard burn-in to a good seed
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=200, abstol=TARGET, alpha=0.3)
    P_warm = res_p["P_star"]
    # Analytic Newton to machine precision
    try:
        res_n = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_warm, maxiters=12, abstol=TARGET,
            lgmres_tol=1e-12, lgmres_maxiter=80)
        P = res_n["P_star"]
        Finf = res_n["best_Finf"]
    except Exception as ex:
        P = P_warm; Finf = float(np.abs(res_p["residual"]).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[ir, jr, lr])
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr_gap = mu[0] - mu[1]
    if Finf > TARGET:
        status = "no fixed point at this tol"
    elif one_r2 > 0.01:
        status = "PR!"
    else:
        status = "FR (machine prec)"
    print(f"{label:30s}  {G:2d}  {p:10.4f}  {one_r2:11.3e}  "
          f"{pr_gap:+9.4f}  {Finf:11.3e}  {time.time()-t0:6.1f}s  {status}",
          flush=True)
