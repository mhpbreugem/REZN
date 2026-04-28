"""Batch 7: longest-Newton retry from the configs that came closest to
convergence with non-trivial 1-R².

The most promising candidates from batches 1+2 had Finf in [1e-6,
1e-3] with 1-R² in [0.03, 0.15]. If those iterates are transients on
the PR manifold, more Newton iterations push them down to machine
precision while keeping 1-R² > 0.01. If they're slow-FR, they
collapse to 1-R² = 0 at machine precision.

For each candidate we run from no-learning seed with
  maxiters = 40 Newton iters
  lgmres_maxiter = 100, lgmres_tol = 1e-13
  abstol = 1e-12

Saves the tensor as PR_seed_g{γ}_t{τ}_b7.pkl iff the converged
result has 1-R² > 0.01 AND Finf < 1e-12.
"""
import time
import pickle
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CANDIDATES = [
    # (G, τ_vec, γ_vec, label)  — promising (high 1-R², low Finf)
    (11, (3.0, 3.0, 3.0),       (3.0, 3.0, 3.0),       "γ=3 τ=3"),
    (11, (4.0, 4.0, 4.0),       (3.0, 3.0, 3.0),       "γ=3 τ=4"),
    (11, (5.0, 5.0, 5.0),       (3.0, 3.0, 3.0),       "γ=3 τ=5"),
    (11, (5.0, 5.0, 5.0),       (1.0, 1.0, 1.0),       "γ=1 τ=5"),
    (11, (5.0, 5.0, 5.0),       (0.5, 0.5, 0.5),       "γ=0.5 τ=5"),
    (11, (3.4, 3.4, 3.4),       (3.0, 3.0, 3.0),       "γ=3 τ=3.4"),
    (11, (10.0, 3.0, 1.0),      (1.0, 3.0, 10.0),      "aligned (10,3,1) τ=(1,3,10)"),
    (11, (1.0, 3.0, 10.0),      (10.0, 3.0, 1.0),      "anti-aligned"),
    (11, (1.0, 3.0, 10.0),      (1.0, 3.0, 10.0),      "opposed (1,3,10)"),
    (11, (0.5, 3.0, 10.0),      (0.3, 3.0, 10.0),      "spec extreme"),
    (11, (1.0, 3.0, 10.0),      (0.3, 3.0, 30.0),      "extreme γ=(0.3,3,30) τ=(1,3,10)"),
]
UMAX = 2.0
TARGET = 1e-12
NEWTON_ITERS = 40                # 5x batch 6
LGMRES_MAXITER = 100
LGMRES_TOL = 1e-13


print(f"{'cfg':40s}  G   {'p':>10s}  {'1-R²':>11s}  {'PR-gap':>9s}  "
      f"{'Finf':>11s}  {'time':>6s}  status", flush=True)

for G, taus_t, gammas_t, label in CANDIDATES:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)
    Ws = np.array([1.0, 1.0, 1.0])

    # No-learning seed
    P0 = rh._nolearning_price(u, taus, gammas, Ws)

    # Picard burn-in to a near-fixed-point starting state
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=300, abstol=TARGET, alpha=0.3,
                                    P_init=P0)
    P_warm = res_p["P_star"]
    Finf_picard = float(np.abs(res_p["residual"]).max())

    t0 = time.time()
    try:
        res = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_warm, maxiters=NEWTON_ITERS, abstol=TARGET,
            lgmres_tol=LGMRES_TOL, lgmres_maxiter=LGMRES_MAXITER)
        P = res["P_star"]; Finf = res["best_Finf"]
    except Exception as ex:
        P = P_warm; Finf = Finf_picard
    one_r2 = rh.one_minus_R2(P, u, taus)
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[ir, jr, lr])
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr_gap = mu[0] - mu[1]
    if Finf < TARGET and one_r2 > 0.01:
        status = "PR! @ machine prec"
        fn = f"/home/user/REZN/python/PR_seed_g{gammas_t[0]}_t{taus_t[0]}_b7.pkl"
        with open(fn, "wb") as f:
            pickle.dump({"P": P, "taus": taus, "gammas": gammas,
                          "G": G, "umax": UMAX, "Finf": Finf,
                          "1-R²": one_r2, "label": label}, f)
    elif Finf < TARGET:
        status = "FR (machine)"
    else:
        status = "no fp at tol"
    print(f"{label:40s}  {G:2d}  {p:10.4f}  {one_r2:11.3e}  "
          f"{pr_gap:+9.4f}  {Finf:11.3e}  {time.time()-t0:6.1f}s  {status}",
          flush=True)
