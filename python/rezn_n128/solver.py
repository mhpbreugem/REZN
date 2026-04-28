"""End-to-end solver: nolearning seed → adaptive Picard → LM polish →
TSVD fallback. Strict float128 throughout — Φ is computed in pure-numpy f128,
LM/TSVD use f64 LAPACK but iteratively refine residuals in f128.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

from .primitives import DTYPE, EPS_OUTER, build_grid, cast_problem
from .phi import phi_map, nolearning_seed
from .picard import picard_adaptive
from .newton import build_fd_jacobian, lm_solve, tsvd_solve
from .diagnostics import analyse, pretty
from .io import load, save as io_save


def one_minus_R2(P, u, taus):
    """1 − R² where the CARA-FR reference is logit(p) = (1/K) · Σ τ_k u_k."""
    K = P.ndim
    G = u.shape[0]
    P64 = P.astype(np.float64)
    u64 = u.astype(np.float64)
    taus64 = np.asarray(taus, dtype=np.float64).reshape(-1)
    y = np.log(P64 / (1.0 - P64)).reshape(-1)
    T = np.zeros(G ** K, dtype=np.float64)
    for idx in np.ndindex(*P.shape):
        flat = np.ravel_multi_index(idx, P.shape)
        s = 0.0
        for k in range(K):
            s += taus64[k] * u64[idx[k]]
        T[flat] = s
    y_c = y - y.mean()
    T_c = T - T.mean()
    Syy = float((y_c * y_c).sum())
    STT = float((T_c * T_c).sum())
    SyT = float((y_c * T_c).sum())
    if Syy == 0.0 or STT == 0.0:
        return 0.0
    R2 = (SyT * SyT) / (Syy * STT)
    return 1.0 - R2


class Tee:
    """Mirror writes to stdout + an open file handle."""
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, msg):
        line = str(msg)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.fh.write(line + "\n")
        self.fh.flush()


def _resolve_seed(P_init, u, taus, gammas, Ws, log):
    if P_init is None:
        log("  [seed] using no-learning seed")
        return nolearning_seed(u, taus, gammas, Ws)
    if isinstance(P_init, str):
        log(f"  [seed] loading from {P_init}")
        rec = load(P_init)
        if "P_f128" in rec and rec["P_f128"] is not None:
            return np.asarray(rec["P_f128"], dtype=DTYPE).copy()
        return np.asarray(rec["P"], dtype=DTYPE).copy()
    return np.asarray(P_init, dtype=DTYPE).copy()


def solve(
    *,
    gammas, taus, Ws,
    G, umax,
    P_init=None,
    picard_iters=20000, picard_alpha0=0.20,
    picard_alpha_min=0.02, picard_alpha_max=0.30,
    lm_iters=15, lm_lambda0=DTYPE("1e-3"),
    tsvd_iters=8, tsvd_rcond_init=1e-6,
    target_finf=1e-12,
    fd_h=DTYPE("1e-12"),
    log_path=None,
    log_interval_s=120.0,
    save_to=None, label="",
    checkpoint_path=None, checkpoint_every=50,
):
    """Run the full pipeline. Returns a result dict.

    Mandatory inputs:
      gammas, taus, Ws : length-K arrays
      G, umax          : grid

    Optional inputs:
      P_init   : ndarray, file path, or None (no-learning seed)
      log_path : file to mirror stdout into (None → stdout only)
      save_to  : pickle output path (None → don't save)
    """
    taus_f, gammas_f, Ws_f = cast_problem(taus, gammas, Ws)
    K = taus_f.shape[0]
    u = build_grid(G, umax)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_fh = open(log_path, "w")
        log = Tee(log_fh)
    else:
        log = print
        log_fh = None

    eps128 = DTYPE(EPS_OUTER)
    one_m_eps = DTYPE(1.0) - eps128
    target_f = float(target_finf)

    log(f"=== rezn_n128.solve  K={K}  G={G}  umax={umax} ===")
    log(f"  γ = {np.asarray(gammas_f, dtype=np.float64).tolist()}")
    log(f"  τ = {np.asarray(taus_f, dtype=np.float64).tolist()}")
    log(f"  W = {np.asarray(Ws_f, dtype=np.float64).tolist()}")
    log(f"  target Finf = {target_f:.1e}")

    P = _resolve_seed(P_init, u, taus_f, gammas_f, Ws_f, log)
    P = np.clip(P, eps128, one_m_eps)
    timings = {}

    def Phi(Pin):
        return phi_map(Pin, u, taus_f, gammas_f, Ws_f)

    def F_eval(Pin):
        return Pin - Phi(Pin)

    F = F_eval(P)
    Finf0 = float(np.abs(F).max())
    log(f"  initial Finf = {Finf0:.3e}")

    history = []

    # ----- Phase 1: Picard
    if picard_iters > 0:
        log("\n=== Phase 1: adaptive Picard ===")
        t0 = time.time()

        def extra_metrics(Pcur):
            return {"1-R²": one_minus_R2(Pcur, u, taus_f)}

        pic = picard_adaptive(
            P, Phi,
            alpha0=picard_alpha0,
            alpha_min=picard_alpha_min,
            alpha_max=picard_alpha_max,
            maxiter=picard_iters,
            abstol=DTYPE(target_f),
            log=log,
            log_interval_s=log_interval_s,
            extra_fn=extra_metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )
        timings["picard_s"] = time.time() - t0
        history.extend(("picard", h) for h in pic["history"])
        P = pic["P_best"]
        F = F_eval(P)
        Finf = float(np.abs(F).max())
        log(f"  picard end: best Finf={pic['Finf_best']:.3e}  "
            f"verified={Finf:.3e}")
        if Finf <= target_f:
            return _finalise(P, F, u, taus_f, gammas_f, Ws_f, G, umax,
                             Finf, history, timings, save_to, label, log, log_fh)

    # ----- Phase 2: LM polish
    if lm_iters > 0:
        log("\n=== Phase 2: Levenberg-Marquardt polish ===")
        t_phase = time.time()
        lam = DTYPE(lm_lambda0)
        for it in range(lm_iters):
            t_iter = time.time()
            log(f"\n--- LM iter {it} (λ={float(lam):.3e}) ---")
            J = build_fd_jacobian(P, F_eval, h=fd_h, log=log,
                                  log_interval_s=log_interval_s)

            accepted = False
            Finf_try = Finf
            alpha = DTYPE(1.0)
            for lam_attempt in range(4):
                log(f"  [LM] λ-attempt {lam_attempt}: λ={float(lam):.3e}")
                dP_flat = lm_solve(J, F, lam, log=log)
                if dP_flat is None:
                    lam = lam * DTYPE(5.0); continue
                dP = dP_flat.reshape(P.shape)
                alpha = DTYPE(1.0)
                for back in range(10):
                    P_try = np.clip(P + alpha * dP, eps128, one_m_eps)
                    F_try = F_eval(P_try)
                    Finf_try = float(np.abs(F_try).max())
                    if Finf_try < Finf:
                        P = P_try; F = F_try
                        accepted = True
                        break
                    alpha = alpha * DTYPE(0.5)
                if accepted:
                    lam = max(DTYPE("1e-12"), lam / DTYPE(3.0))
                    log(f"  [LM] accepted; λ → {float(lam):.3e}")
                    break
                else:
                    lam = lam * DTYPE(5.0)
                    log(f"  [LM] rejected; λ → {float(lam):.3e}")

            ratio = Finf_try / Finf if Finf > 0 else float("nan")
            r2 = one_minus_R2(P, u, taus_f)
            log(f"  LM iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
                f"ratio={ratio:.4f}  1-R²={r2:.3e}  α={float(alpha):.6f}  "
                f"accepted={accepted}  t={time.time()-t_iter:.1f}s")
            history.append(("lm", Finf_try))
            Finf = Finf_try
            if Finf <= target_f:
                log(f"  LM CONVERGED at iter {it}")
                break
            if not accepted:
                log("  LM step rejected at all λ; moving on.")
                break
        timings["lm_s"] = time.time() - t_phase

    # ----- Phase 3: TSVD fallback
    if tsvd_iters > 0 and Finf > target_f:
        log("\n=== Phase 3: TSVD pseudoinverse polish ===")
        t_phase = time.time()
        rcond = float(tsvd_rcond_init)
        rcond_min, rcond_max = 1e-12, 1e-2
        for it in range(tsvd_iters):
            t_iter = time.time()
            log(f"\n--- TSVD iter {it} (rcond={rcond:.0e}) ---")
            J = build_fd_jacobian(P, F_eval, h=fd_h, log=log,
                                  log_interval_s=log_interval_s)

            accepted = False
            Finf_try = Finf
            alpha = DTYPE(1.0)
            last_rank = -1
            for rcond_attempt in range(5):
                log(f"  [TSVD] attempt {rcond_attempt}: rcond={rcond:.0e}")
                dP_flat, rank, _ = tsvd_solve(J, -F, rcond, log=log)
                last_rank = rank
                dP = dP_flat.reshape(P.shape)
                alpha = DTYPE(1.0)
                for back in range(10):
                    P_try = np.clip(P + alpha * dP, eps128, one_m_eps)
                    F_try = F_eval(P_try)
                    Finf_try = float(np.abs(F_try).max())
                    if Finf_try < Finf:
                        P = P_try; F = F_try
                        accepted = True
                        break
                    alpha = alpha * DTYPE(0.5)
                if accepted:
                    rcond = max(rcond_min, rcond / 3.0)
                    log(f"  [TSVD] accepted; rcond → {rcond:.0e}")
                    break
                else:
                    rcond = min(rcond_max, rcond * 10.0)
                    log(f"  [TSVD] rejected; rcond → {rcond:.0e}")

            ratio = Finf_try / Finf if Finf > 0 else float("nan")
            r2 = one_minus_R2(P, u, taus_f)
            log(f"  TSVD iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
                f"ratio={ratio:.4f}  1-R²={r2:.3e}  α={float(alpha):.6f}  "
                f"rank={last_rank}  accepted={accepted}  t={time.time()-t_iter:.1f}s")
            history.append(("tsvd", Finf_try))
            Finf = Finf_try
            if Finf <= target_f:
                log(f"  TSVD CONVERGED at iter {it}")
                break
            if not accepted:
                log("  TSVD step rejected at all rcond; stopping.")
                break
        timings["tsvd_s"] = time.time() - t_phase

    return _finalise(P, F, u, taus_f, gammas_f, Ws_f, G, umax, Finf,
                     history, timings, save_to, label, log, log_fh)


def _finalise(P, F, u, taus, gammas, Ws, G, umax, Finf,
              history, timings, save_to, label, log, log_fh):
    log("\n=== Finalise ===")
    one_r2 = one_minus_R2(P, u, taus)
    log(f"  Finf      = {Finf:.3e}")
    log(f"  1-R²      = {one_r2:.6e}")
    diag = analyse(P, F)
    pretty(diag, log=log)

    result = dict(
        P=P.astype(np.float64),
        P_f128=P,
        u=u.astype(np.float64),
        taus=np.asarray(taus, dtype=np.float64),
        gammas=np.asarray(gammas, dtype=np.float64),
        Ws=np.asarray(Ws, dtype=np.float64),
        G=int(G),
        umax=float(umax),
        Finf=float(Finf),
        one_minus_R2=float(one_r2),
        history=history,
        diagnostics=diag,
        timings=timings,
    )

    if save_to:
        io_save(save_to, P, taus=taus, gammas=gammas, Ws=Ws, G=G, umax=umax,
                Finf=Finf, one_minus_R2=one_r2, label=label,
                history=history, diagnostics=diag,
                extra={"timings": timings})
        log(f"  saved → {save_to}")
    if log_fh is not None:
        log_fh.close()
    return result
