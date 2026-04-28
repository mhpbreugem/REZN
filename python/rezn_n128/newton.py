"""Newton-class polishers for F(P) = P − Φ(P) = 0 in float128.

build_fd_jacobian — central differences in float128, with a 10s heartbeat.

lm_solve(J, F, λ) — Levenberg-Marquardt: solve (JᵀJ + λI) dP = −JᵀF.
Normal equations are built in f128, the LU is f64 (numpy/scipy don't ship a
f128 LAPACK), and the residual is iteratively refined in f128 — gaining
~3 extra digits in dP relative to a pure-f64 solve.

tsvd_solve(J, F, rcond) — truncated-SVD pseudoinverse: dP = V·diag(1/σ if
σ>σ_max·rcond else 0)·Uᵀ·(−F). SVD is f64, refinement uses J in f128.
"""
from __future__ import annotations
import time
import numpy as np

from .primitives import DTYPE, EPS_OUTER


# ---------------------------------------------------------------- Jacobian
def build_fd_jacobian(P, F_eval, *, h=DTYPE("1e-7"), log=print, log_interval_s=10.0):
    """Central-difference Jacobian J[:, k] = (F(P + h·e_k) − F(P − h·e_k)) / (2h).

    F_eval(P_array) returns the residual array (same shape as P).
    All arithmetic in float128. Per-perturbation P is reclipped so the FD
    stays inside (EPS_OUTER, 1−EPS_OUTER).

    Default h = 1e-7 is near the f128 sweet spot: ε^(1/3) ≈ (1e-19)^(1/3)
    ≈ 5e-7. Smaller h (e.g., 1e-12 used previously for f64-derived code)
    causes catastrophic cancellation: the numerator F(P+h)-F(P-h) ≈ 2Jh
    has only eps/h ≈ 1e-7 relative precision. With h=1e-7 the column has
    ~1e-13 precision, ~6 digits better.
    """
    eps = DTYPE(EPS_OUTER)
    one_m_eps = DTYPE(1.0) - eps
    h = DTYPE(h)
    P_flat = P.reshape(-1).astype(DTYPE)
    N = P_flat.size
    J = np.empty((N, N), dtype=DTYPE)
    t0 = time.time()
    t_last = 0.0
    for k in range(N):
        Pp = P_flat.copy(); Pp[k] = Pp[k] + h
        Pm = P_flat.copy(); Pm[k] = Pm[k] - h
        Pp = np.clip(Pp.reshape(P.shape), eps, one_m_eps)
        Pm = np.clip(Pm.reshape(P.shape), eps, one_m_eps)
        Fp = F_eval(Pp).reshape(-1)
        Fm = F_eval(Pm).reshape(-1)
        J[:, k] = (Fp - Fm) / (DTYPE(2.0) * h)
        elapsed = time.time() - t0
        if elapsed - t_last >= log_interval_s:
            log(f"    [FD-J] col {k+1}/{N}  elapsed={elapsed:.1f}s")
            t_last = elapsed
    log(f"    [FD-J] done in {time.time()-t0:.1f}s")
    return J


# ---------------------------------------------------------------- LM
def lm_solve(J, F, lam, *, refine_max=12, refine_tol=1e-25, log=print):
    """Solve (JᵀJ + λI) dP = −JᵀF with iterative refinement.

    Returns dP as float128, or None on factorisation failure.
    """
    N = J.shape[0]
    Ft = F.reshape(-1).astype(DTYPE)
    lam = DTYPE(lam)

    log(f"      [LM] building JᵀJ (N={N})...")
    t0 = time.time()
    A = J.T @ J
    diag = np.arange(N)
    A[diag, diag] = A[diag, diag] + lam
    rhs = -(J.T @ Ft)
    log(f"      [LM] normal eqns built in {time.time()-t0:.1f}s")

    A64 = A.astype(np.float64)
    rhs64 = rhs.astype(np.float64)
    log(f"      [LM] solving f64 LU...")
    t0 = time.time()
    try:
        dP = np.linalg.solve(A64, rhs64).astype(DTYPE)
    except Exception as ex:
        log(f"      [LM] f64 LU FAILED: {ex}")
        return None
    r0 = float(np.abs(rhs - A @ dP).max())
    log(f"      [LM] f64 LU done in {time.time()-t0:.1f}s, ‖r‖∞={r0:.3e}")

    for r_it in range(refine_max):
        r128 = rhs - (A @ dP)
        r_norm = float(np.abs(r128).max())
        if r_norm < refine_tol:
            log(f"      [LM-refine {r_it}] CONVERGED ‖r‖∞={r_norm:.3e}")
            break
        delta = np.linalg.solve(A64, r128.astype(np.float64))
        dP_new = dP + delta.astype(DTYPE)
        new_r = float(np.abs(rhs - A @ dP_new).max())
        rate = new_r / r_norm if r_norm > 0 else float("nan")
        log(f"      [LM-refine {r_it}] ‖r‖∞ {r_norm:.3e} → {new_r:.3e}  rate={rate:.3e}")
        if new_r >= r_norm:
            log(f"      [LM-refine {r_it}] STAGNATED — stop")
            break
        dP = dP_new
    return dP


# ---------------------------------------------------------------- TSVD
def tsvd_solve(J, b, rcond, *, refine_max=6, refine_tol=1e-22, log=print):
    """dP = J⁺·b with truncation σ_i ≤ σ_max·rcond → drop component.

    Returns (dP_f128, rank_kept, ‖r_final‖∞).
    """
    N = J.shape[0]
    J64 = J.astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    b128 = b.reshape(-1).astype(DTYPE)

    log(f"      [TSVD] computing SVD...")
    t0 = time.time()
    U, sigma, Vt = np.linalg.svd(J64, full_matrices=False)
    log(f"      [TSVD] SVD done in {time.time()-t0:.1f}s; "
        f"σ_max={sigma[0]:.3e}, σ_min={sigma[-1]:.3e}")

    threshold = sigma[0] * rcond
    keep = sigma > threshold
    rank = int(keep.sum())
    inv_sigma = np.where(keep, 1.0 / sigma, 0.0)
    log(f"      [TSVD] rcond={rcond:.0e}, threshold={threshold:.3e}, "
        f"rank kept={rank}/{N}")

    dP = ((Vt.T * inv_sigma) @ (U.T @ b64)).astype(DTYPE)

    for r_it in range(refine_max):
        r128 = b128 - (J @ dP)
        r_norm = float(np.abs(r128).max())
        if r_norm < refine_tol:
            log(f"      [TSVD-refine {r_it}] CONVERGED ‖r‖∞={r_norm:.3e}")
            break
        delta = ((Vt.T * inv_sigma) @ (U.T @ r128.astype(np.float64))).astype(DTYPE)
        dP_new = dP + delta
        new_r = float(np.abs(b128 - J @ dP_new).max())
        rate = new_r / r_norm if r_norm > 0 else float("nan")
        log(f"      [TSVD-refine {r_it}] ‖r‖∞ {r_norm:.3e} → {new_r:.3e}  rate={rate:.3e}")
        if new_r >= r_norm:
            log(f"      [TSVD-refine {r_it}] STAGNATED — stop")
            break
        dP = dP_new

    final_r = float(np.abs(b128 - J @ dP).max())
    return dP, rank, final_r
