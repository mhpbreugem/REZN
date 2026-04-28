"""Batch 8: heterogeneous-τ PR search at γ ∈ {2, 3}.

Hypothesis: the symmetric FR basin in the production PCHIP+contour
kernel is a strong attractor under all-equal (γ, τ). Breaking the
τ-symmetry (each agent sees a different precision) destroys the
permutation-invariance the FR basin relies on and may expose a
PR fixed point that's stable but invisible from symmetric seeds.

We pin γ=2 first (matches the historical 9e-10 result documented in
comparison_table.md) and γ=3 second (matches HANDOFF's headline PR
config), and scan a ladder of τ-asymmetries from mild (1,2,3) to
large (1,3,7). A few fully-heterogeneous (γ, τ) configs at the end.

Each solve: no-learning seed → Picard burn-in → strict Newton.
Strict acceptance: Finf < 1e-12 AND 1-R² > 0.01 → save tensor.
"""
import time
import pickle
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CANDIDATES = [
    # (G, τ_vec, γ_vec, label)
    # γ=2 ladder (homogeneous γ, asymmetric τ)
    (11, (1.0, 2.0, 3.0),       (2.0, 2.0, 2.0),       "γ=2 τ=(1,2,3)"),
    (11, (0.5, 2.0, 3.5),       (2.0, 2.0, 2.0),       "γ=2 τ=(0.5,2,3.5)"),
    (11, (1.0, 3.0, 5.0),       (2.0, 2.0, 2.0),       "γ=2 τ=(1,3,5)"),
    (11, (2.0, 3.0, 4.0),       (2.0, 2.0, 2.0),       "γ=2 τ=(2,3,4)"),
    (11, (1.0, 4.0, 7.0),       (2.0, 2.0, 2.0),       "γ=2 τ=(1,4,7)"),
    (11, (0.5, 3.0, 6.0),       (2.0, 2.0, 2.0),       "γ=2 τ=(0.5,3,6)"),
    # γ=3 ladder (matches HANDOFF flavor)
    (11, (1.0, 3.0, 5.0),       (3.0, 3.0, 3.0),       "γ=3 τ=(1,3,5)"),
    (11, (2.0, 3.4, 5.0),       (3.0, 3.0, 3.0),       "γ=3 τ=(2,3.4,5)"),
    (11, (1.0, 3.4, 7.0),       (3.0, 3.0, 3.0),       "γ=3 τ=(1,3.4,7)"),
    (11, (0.5, 3.4, 8.0),       (3.0, 3.0, 3.0),       "γ=3 τ=(0.5,3.4,8)"),
    (11, (2.0, 3.0, 5.0),       (3.0, 3.0, 3.0),       "γ=3 τ=(2,3,5)"),
    # Fully heterogeneous (γ AND τ asymmetric)
    (11, (2.0, 3.0, 4.0),       (2.0, 3.0, 4.0),       "γ=(2,3,4) τ=(2,3,4)"),
    (11, (1.0, 2.0, 3.0),       (1.0, 2.0, 3.0),       "γ=(1,2,3) τ=(1,2,3)"),
    (11, (0.5, 2.0, 5.0),       (0.5, 2.0, 5.0),       "γ=(0.5,2,5) τ=(0.5,2,5)"),
    (11, (1.0, 3.0, 5.0),       (3.0, 2.0, 1.0),       "γ-anti τ-asc"),
]
UMAX = 2.0
TARGET = 1e-12
NEWTON_ITERS = 40
LGMRES_MAXITER = 100
LGMRES_TOL = 1e-13


print(f"{'cfg':40s}  G   {'p':>10s}  {'1-R²':>11s}  {'PR-gap':>9s}  "
      f"{'Finf':>11s}  {'time':>6s}  status", flush=True)

for G, taus_t, gammas_t, label in CANDIDATES:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)
    Ws = np.array([1.0, 1.0, 1.0])

    P0 = rh._nolearning_price(u, taus, gammas, Ws)

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
    except Exception:
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
        tag = (f"g{gammas_t[0]}-{gammas_t[1]}-{gammas_t[2]}_"
               f"t{taus_t[0]}-{taus_t[1]}-{taus_t[2]}")
        fn = f"/home/user/REZN/python/PR_seed_{tag}_b8.pkl"
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
