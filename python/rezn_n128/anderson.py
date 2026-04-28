"""Anderson acceleration for the fixed-point problem P = Φ(P).

Anderson (Type-II / "fixed-point" form) replaces the damped Picard step

    P_{k+1} = P_k − β · F(P_k)

with a step that reuses the last m residuals:

    γ = argmin_γ  ‖F(P_k) − ΔF · γ‖²
    P_{k+1} = P_k − β·F(P_k) − (ΔP − β·ΔF) · γ

where ΔF[:, j] = F(P_k) − F(P_{k-m+j}) and ΔP[:, j] = P_k − P_{k-m+j}.

For β=1 this is equivalent to a quasi-Newton step taking the affine
combination of past Φ(P_j) that minimises ‖F‖. With β<1 it is damped.

Reference: Walker & Ni (2011), "Anderson Acceleration for Fixed-Point
Iterations," SIAM J. Num. Anal. 49, 1715–1735.

Why this matters here: plain damped Picard contracts at a rate bounded
by the spectral radius of DΦ at the fp. Near a near-singular fp (which
this REE problem hits) that rate is ≈1, so Picard stalls. Anderson can
break through by extrapolating from a low-dimensional subspace of past
iterates. Empirically 5–50× faster than damped Picard with no Jacobian.
"""
from __future__ import annotations
import os
import pickle
import time
import numpy as np

from .primitives import DTYPE, EPS_OUTER


def _atomic_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def anderson_solve(
    P_init,
    phi_fn,
    *,
    m=8, beta=DTYPE(1.0),
    maxiter=2000, abstol=DTYPE("1e-13"),
    reg=DTYPE("1e-12"),
    safeguard_rho=DTYPE("0.999"),
    log=print, log_interval_s=10.0,
    extra_fn=None,
    checkpoint_path=None, checkpoint_every=50,
):
    """Anderson-accelerated fixed point.

      m              : history depth (number of past residuals retained)
      beta           : mixing parameter; β=1 → pure Anderson, β<1 → damped
      reg            : Tikhonov regularization on the LS solve
      safeguard_rho  : if ‖F_new‖ > ρ ‖F_old‖ after an Anderson step, fall
                       back to a plain Picard step (typed-II safeguarding)

    Returns dict(P_best, Finf_best, P_last, Finf_last, history,
    n_safeguard, elapsed).
    """
    eps = DTYPE(EPS_OUTER)
    one_m_eps = DTYPE(1.0) - eps
    abstol_f = float(abstol)
    beta = DTYPE(beta)
    reg = DTYPE(reg)

    P = np.clip(np.asarray(P_init, dtype=DTYPE).copy(), eps, one_m_eps)
    Phi_P = phi_fn(P)
    F = P - Phi_P
    Finf = float(np.abs(F).max())
    P_best = P.copy()
    Finf_best = float("inf")
    if Finf < 0.5:
        Finf_best = Finf
    history = [Finf]

    n_safeguard = 0
    P_hist = [P.copy()]
    F_hist = [F.copy()]

    t_start = time.time()
    t_last_print = -1.0

    for it in range(maxiter):
        if Finf < abstol_f:
            elapsed = time.time() - t_start
            log(f"  [anderson] CONVERGED iter {it}: Finf={Finf:.3e}  "
                f"elapsed={elapsed:.1f}s")
            break

        m_k = min(m, len(P_hist) - 1)
        if m_k < 1:
            # First iter — plain Picard
            P_new = np.clip(P - beta * F, eps, one_m_eps)
        else:
            # Build dF_mat, dP_mat
            N = P.size
            dF = np.empty((N, m_k), dtype=DTYPE)
            dP = np.empty((N, m_k), dtype=DTYPE)
            for j in range(m_k):
                # column j: difference between current F and (m_k - j)th most recent past F
                dF[:, j] = (F - F_hist[-1 - (m_k - j)]).reshape(-1)
                dP[:, j] = (P - P_hist[-1 - (m_k - j)]).reshape(-1)
            # Solve regularized LS: gamma = argmin ||F - dF @ gamma||² + reg ||gamma||²
            # Normal eqns: (dFᵀ dF + reg I) gamma = dFᵀ F
            # In float128 throughout
            G_mat = dF.T @ dF
            G_mat[np.arange(m_k), np.arange(m_k)] = (
                G_mat[np.arange(m_k), np.arange(m_k)] + reg
            )
            rhs = dF.T @ F.reshape(-1)
            try:
                # f64 LU for speed; f128 residual is dominated by the LS error anyway
                gamma = np.linalg.solve(G_mat.astype(np.float64),
                                          rhs.astype(np.float64)).astype(DTYPE)
            except np.linalg.LinAlgError:
                # Singular: fall back to Picard
                gamma = np.zeros(m_k, dtype=DTYPE)
            update = (dP - beta * dF) @ gamma
            P_new_flat = (P.reshape(-1) - beta * F.reshape(-1) - update)
            P_new = np.clip(P_new_flat.reshape(P.shape), eps, one_m_eps)

        # Evaluate at new iterate
        Phi_new = phi_fn(P_new)
        F_new = P_new - Phi_new
        Finf_new = float(np.abs(F_new).max())

        # Type-II safeguarding: if Anderson step blows up, fall back to plain Picard
        if Finf_new > float(safeguard_rho) * Finf and m_k > 0:
            P_new = np.clip(P - beta * F, eps, one_m_eps)
            Phi_new = phi_fn(P_new)
            F_new = P_new - Phi_new
            Finf_new = float(np.abs(F_new).max())
            n_safeguard += 1

        # Maintain history (drop oldest if at capacity)
        P_hist.append(P_new.copy())
        F_hist.append(F_new.copy())
        if len(P_hist) > m + 1:
            P_hist.pop(0)
            F_hist.pop(0)

        # Update best
        if Finf_new < 0.5 and Finf_new < Finf_best:
            Finf_best = Finf_new
            P_best = P_new.copy()

        if (checkpoint_path is not None
                and (it + 1) % checkpoint_every == 0
                and Finf_best < float("inf")):
            _atomic_pickle(checkpoint_path, dict(
                P_f128=P_best, P=P_best.astype(np.float64),
                iter=it + 1, Finf=Finf_best,
                schema="anderson_checkpoint/1",
            ))

        P, Phi_P, F, Finf = P_new, Phi_new, F_new, Finf_new
        history.append(Finf)

        # Heartbeat
        elapsed = time.time() - t_start
        if (elapsed - t_last_print) >= log_interval_s or it == maxiter - 1:
            extra_str = ""
            if extra_fn is not None:
                try:
                    for k, v in extra_fn(P).items():
                        extra_str += f"  {k}={float(v):.3e}"
                except Exception as ex:
                    extra_str = f"  [extra_fn err: {ex}]"
            log(f"  [anderson] iter {it+1}/{maxiter}  Finf={Finf:.3e}  "
                f"best={Finf_best:.3e}{extra_str}  m_k={m_k}  "
                f"safeguards={n_safeguard}  elapsed={elapsed:.1f}s")
            t_last_print = elapsed

    elapsed = time.time() - t_start
    log(f"  [anderson] done: Finf_last={Finf:.3e}  Finf_best={Finf_best:.3e}  "
        f"safeguards={n_safeguard}  elapsed={elapsed:.1f}s")
    return dict(
        P_best=P_best, Finf_best=Finf_best,
        P_last=P, Finf_last=Finf,
        history=history,
        n_safeguard=n_safeguard,
        elapsed=elapsed,
    )
