"""Adaptive Picard iteration for P = Φ(P).

α-shrink on log-residual slope ≥ 0; α-relax on a monotone-decrease streak;
σ-perturbation when best-in-window stalls; tracks the best non-saturated
iterate across the whole run and returns it (the last iterate may be a
spurious saturated one).

If `checkpoint_path` is given, atomically writes the best iterate every
`checkpoint_every` iters as a minimal pickle ({'P_f128', 'iter', 'Finf',
'alpha'}). Loadable by `solver._resolve_seed` (just keys 'P_f128'/'P').
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


def picard_adaptive(
    P_init,
    phi_fn,
    *,
    alpha0=0.20, alpha_min=0.02, alpha_max=0.30,
    maxiter=20000, abstol=DTYPE("1e-13"),
    slope_window=100, alpha_chg_cooldown=50,
    mono_relax=400,
    stall_window=500, stall_ratio=0.95,
    perturb_sigma=DTYPE("1e-6"), perturb_seed=42,
    log=print, log_interval_s=10.0,
    extra_fn=None,
    checkpoint_path=None, checkpoint_every=50,
):
    """Run adaptive Picard until residual < abstol or maxiter exhausted.

    `phi_fn(P)` returns Φ(P). `log(msg)` is called for heartbeats and
    α/perturb events.

    Returns dict(P_best, Finf_best, P_last, Finf_last, history,
    n_perturb, n_alpha_chg, elapsed).
    """
    rng = np.random.default_rng(perturb_seed)
    eps = DTYPE(EPS_OUTER)
    one_m_eps = DTYPE(1.0) - eps

    P = np.clip(np.asarray(P_init, dtype=DTYPE).copy(), eps, one_m_eps)
    P_best = P.copy()
    Finf_best = float("inf")
    alpha = DTYPE(alpha0)
    a_min = DTYPE(alpha_min)
    a_max = DTYPE(alpha_max)
    abstol_f = float(abstol)
    sigma = DTYPE(perturb_sigma)

    history = []
    decr_streak = 0
    last_perturb = -10**6
    last_alpha_chg = -10**6
    n_perturb = 0
    n_alpha_chg = 0
    t_start = time.time()
    t_last_print = -1.0

    for it in range(maxiter):
        Phi = phi_fn(P)
        F = P - Phi
        Finf = float(np.abs(F).max())
        history.append(Finf)

        if Finf < 0.5 and Finf < Finf_best:
            Finf_best = Finf
            P_best = P.copy()

        if (checkpoint_path is not None
                and (it + 1) % checkpoint_every == 0
                and Finf_best < float("inf")):
            _atomic_pickle(checkpoint_path, dict(
                P_f128=P_best, P=P_best.astype(np.float64),
                iter=it + 1, Finf=Finf_best, alpha=float(alpha),
                schema="picard_checkpoint/1",
            ))

        if Finf < abstol_f:
            elapsed = time.time() - t_start
            log(f"  [picard] CONVERGED iter {it+1}: Finf={Finf:.3e}  "
                f"elapsed={elapsed:.1f}s  α={float(alpha):.4f}")
            break

        # α adapt: shrink on positive log-slope
        alpha_changed = False
        if it >= slope_window and (it - last_alpha_chg) >= alpha_chg_cooldown:
            window = np.asarray(history[-slope_window:], dtype=np.float64)
            log_w = np.log(np.maximum(window, 1e-300))
            xs = np.arange(slope_window, dtype=np.float64)
            slope = float(np.polyfit(xs, log_w, 1)[0])
            if slope >= 0.0 and alpha > a_min:
                new_a = max(float(a_min), float(alpha) * 0.7)
                if new_a < float(alpha):
                    alpha = DTYPE(new_a)
                    last_alpha_chg = it
                    alpha_changed = True
                    n_alpha_chg += 1
                    decr_streak = 0
        if it > 0 and Finf < history[-2]:
            decr_streak += 1
            if (decr_streak >= mono_relax
                    and alpha < a_max
                    and (it - last_alpha_chg) >= alpha_chg_cooldown):
                new_a = min(float(a_max), float(alpha) * 1.05)
                if new_a > float(alpha):
                    alpha = DTYPE(new_a)
                    last_alpha_chg = it
                    alpha_changed = True
                    n_alpha_chg += 1
                    decr_streak = 0
        else:
            decr_streak = 0

        # stall → in-place perturb
        perturbed = False
        if (it >= 2 * stall_window
                and (it - last_perturb) >= stall_window):
            recent_min = min(history[-stall_window:])
            prior_min = min(history[-2 * stall_window:-stall_window])
            if prior_min > 0 and recent_min >= stall_ratio * prior_min:
                support = (P > eps * DTYPE(10)) & (P < one_m_eps - DTYPE(9) * eps)
                scale = DTYPE(float(P[support].mean())) if support.any() else DTYPE(0.5)
                noise = (sigma * scale
                         * np.asarray(rng.standard_normal(P.shape), dtype=DTYPE))
                P = np.clip(P + noise, eps, one_m_eps)
                last_perturb = it
                n_perturb += 1
                perturbed = True

        if not perturbed:
            P = np.clip(alpha * Phi + (DTYPE(1.0) - alpha) * P, eps, one_m_eps)

        elapsed = time.time() - t_start
        if (
            (elapsed - t_last_print) >= log_interval_s
            or alpha_changed or perturbed
            or it == maxiter - 1
        ):
            tag = ""
            if alpha_changed:
                tag += f" α→{float(alpha):.4f}"
            if perturbed:
                tag += f" PERTURB σ={float(sigma):.0e}"
            extra_str = ""
            if extra_fn is not None:
                try:
                    for k, v in extra_fn(P).items():
                        extra_str += f"  {k}={float(v):.3e}"
                except Exception as ex:
                    extra_str = f"  [extra_fn err: {ex}]"
            log(f"  [picard] iter {it+1}/{maxiter}  Finf={Finf:.3e}  "
                f"best={Finf_best:.3e}{extra_str}  α={float(alpha):.4f}  "
                f"decr={decr_streak}  perturbs={n_perturb}  "
                f"α-chg={n_alpha_chg}  elapsed={elapsed:.1f}s{tag}")
            t_last_print = elapsed

    elapsed = time.time() - t_start
    Phi_final = phi_fn(P)
    Finf_last = float(np.abs(P - Phi_final).max())
    log(f"  [picard] done: Finf_last={Finf_last:.3e}  Finf_best={Finf_best:.3e}  "
        f"perturbs={n_perturb}  α-chg={n_alpha_chg}  elapsed={elapsed:.1f}s")

    return dict(
        P_best=P_best, Finf_best=Finf_best,
        P_last=P, Finf_last=Finf_last,
        history=history,
        n_perturb=n_perturb, n_alpha_chg=n_alpha_chg,
        elapsed=elapsed,
    )
