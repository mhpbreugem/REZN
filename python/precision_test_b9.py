"""Batch 9 — hybrid float128 Newton precision test.

Question: when float64 Newton stalls at Finf ~ 1e-3 to 1e-6 on a
PR-suspected config, is the floor caused by float64 round-off in the
outer iteration (cancellation in F = P - Φ(P), accumulation in
P += dP), or is it intrinsic to the discretised Φ map?

We can't trivially evaluate Φ in float128 because rezn_pchip uses
numba (float64-only). So we use a hybrid scheme:

    P              kept in float128
    Φ(P)           cast P → float64, evaluate, cast back → float128
    F = P − Φ(P)   computed in float128  (catches cancellation)
    J · dP = −F    LGMRES in float64       (numba constraint)
    P += α · dP    in float128             (catches accumulation)

If hybrid Newton drops Finf systematically below the float64-only
result, the outer-loop precision is the bottleneck and a full
float128 build is justified. If it floors at the same place, the
bottleneck is inside Φ or the fixed point doesn't exist there.

Reports for each config:
    Finf_64    — float64 Newton from Picard warm start
    Finf_hybr  — hybrid float128 continuation from Finf_64 iterate
    ratio      — Finf_hybr / Finf_64

Configs include the FR control (γ=3 τ=3, expected to converge in
both regimes) and the four most-stalled PR-suspects from batch 7.
"""
import time
import pickle
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CONFIGS = [
    # (G, τ_vec, γ_vec, label)
    (11, (3.0, 3.0, 3.0),  (3.0, 3.0, 3.0), "γ=3 τ=3 (FR ctrl)"),
    (11, (3.4, 3.4, 3.4),  (3.0, 3.0, 3.0), "γ=3 τ=3.4 (HANDOFF)"),
    (11, (4.0, 4.0, 4.0),  (3.0, 3.0, 3.0), "γ=3 τ=4 (PR-susp)"),
    (11, (5.0, 5.0, 5.0),  (3.0, 3.0, 3.0), "γ=3 τ=5 (PR-susp)"),
    (11, (5.0, 5.0, 5.0),  (1.0, 1.0, 1.0), "γ=1 τ=5 (PR-susp)"),
]
UMAX = 2.0
EPS_OUTER = 1e-9
PICARD_ITERS = 300
F64_NEWTON_ITERS = 12
HYBR_ITERS = 25
LGMRES_TOL = 1e-13
LGMRES_MAXITER = 100


def hybrid_newton(G, taus, gammas, P_init, umax, maxiters):
    """Hybrid f128/f64 Newton continuation.

    State P kept in float128; Φ called in float64 (numba); residual
    F = P - Φ(P) and update P += α·dP done in float128.
    """
    u = np.linspace(-umax, umax, G).astype(np.float64)
    Ws = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    eps128 = np.float128(EPS_OUTER)
    one_m_eps128 = np.float128(1.0 - EPS_OUTER)

    P128 = np.clip(np.asarray(P_init, dtype=np.float128).copy(),
                   eps128, one_m_eps128)

    history = []
    best_Finf = float("inf")
    P_best = P128.copy()
    N = G ** 3

    for it in range(maxiters):
        P64 = P128.astype(np.float64)
        Phi64 = rp._phi_map_pchip(P64, u, taus, gammas, Ws)
        F128 = P128 - Phi64.astype(np.float128)
        Finf = float(np.abs(F128).max())
        history.append(Finf)
        if Finf < best_Finf:
            best_Finf = Finf
            P_best = P128.copy()
        if Finf < 1e-15:
            break

        precomp = pj.precompute(P64, u, taus, gammas, Ws)

        def matvec(v, _pre=precomp, _shape=P64.shape):
            return pj.J_dot_v(v.reshape(_shape), _pre).reshape(-1)

        op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        F64 = F128.astype(np.float64)
        dP_flat, _ = lgmres(op, -F64.reshape(-1),
                             rtol=LGMRES_TOL, atol=0.0,
                             maxiter=LGMRES_MAXITER)
        dP128 = dP_flat.reshape(P128.shape).astype(np.float128)

        alpha = np.float128(1.0)
        accepted = False
        for _ in range(8):
            P_try = np.clip(P128 + alpha * dP128, eps128, one_m_eps128)
            Phi_try = rp._phi_map_pchip(P_try.astype(np.float64),
                                         u, taus, gammas, Ws)
            F_try = P_try - Phi_try.astype(np.float128)
            Finf_try = float(np.abs(F_try).max())
            if Finf_try < Finf:
                P128 = P_try
                break
            alpha *= np.float128(0.5)
        else:
            P128 = np.clip(Phi64.astype(np.float128), eps128, one_m_eps128)

    return P_best, best_Finf, history


print(f"{'cfg':28s}  {'Finf_64':>11s}  {'Finf_hybr':>11s}  "
      f"{'ratio':>9s}  {'1-R²':>10s}  {'time':>6s}  verdict",
      flush=True)

for G, taus_t, gammas_t, label in CONFIGS:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)
    Ws = np.array([1.0, 1.0, 1.0])
    t0 = time.time()

    P0 = rh._nolearning_price(u, taus, gammas, Ws)

    # 1. Picard burn-in (float64)
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=PICARD_ITERS, abstol=1e-15,
                                    alpha=0.3, P_init=P0)
    P_warm = res_p["P_star"]

    # 2. float64 Newton baseline
    res64 = pj.solve_newton(G, taus, gammas, umax=UMAX,
                              P_init=P_warm, maxiters=F64_NEWTON_ITERS,
                              abstol=1e-15,
                              lgmres_tol=LGMRES_TOL,
                              lgmres_maxiter=LGMRES_MAXITER)
    P64_best = res64["P_star"]
    Finf_64 = res64["best_Finf"]

    # 3. Hybrid float128 continuation from the float64 best iterate
    try:
        P_hybr, Finf_hybr, hist_hybr = hybrid_newton(
            G, taus, gammas, P64_best, UMAX, maxiters=HYBR_ITERS)
    except Exception as ex:
        Finf_hybr = float("nan")
        P_hybr = P64_best.astype(np.float128)

    ratio = (Finf_hybr / Finf_64) if Finf_64 > 0 else float("nan")
    one_r2 = rh.one_minus_R2(P_hybr.astype(np.float64), u, taus)

    if Finf_hybr < 1e-12 and one_r2 > 0.01:
        verdict = "PR! @ f128"
        # save tensor
        tag = (f"g{gammas_t[0]}_t{taus_t[0]}_b9hybrid")
        fn = f"/home/user/REZN/python/PR_seed_{tag}.pkl"
        with open(fn, "wb") as f:
            pickle.dump({"P": P_hybr.astype(np.float64),
                          "P_f128": P_hybr,
                          "taus": taus, "gammas": gammas,
                          "G": G, "umax": UMAX,
                          "Finf_f64": Finf_64, "Finf_hybr": Finf_hybr,
                          "1-R²": one_r2, "label": label}, f)
    elif Finf_hybr < Finf_64 * 0.1:
        verdict = "precision helps (>10x)"
    elif Finf_hybr < Finf_64 * 0.5:
        verdict = "precision helps (some)"
    elif np.isnan(Finf_hybr):
        verdict = "FAILED"
    else:
        verdict = "no — same floor"

    print(f"{label:28s}  {Finf_64:11.3e}  {Finf_hybr:11.3e}  "
          f"{ratio:9.3e}  {one_r2:10.3e}  "
          f"{time.time()-t0:6.1f}s  {verdict}",
          flush=True)
