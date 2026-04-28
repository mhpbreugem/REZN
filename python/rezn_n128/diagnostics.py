"""Solver-state diagnostics: clip distribution, top-|F| cells, J's σ-spectrum,
F decomposition over singular vectors.

All routines accept f128 inputs and report f64 numbers (the diagnostics aren't
sensitive to the last 3 digits).
"""
from __future__ import annotations
import numpy as np

from .primitives import EPS_OUTER


def clip_stats(P):
    """Return a dict of cell-counts at/near the clip boundary."""
    P_flat = P.reshape(-1).astype(np.float64)
    eps = float(EPS_OUTER)
    N = P_flat.size
    return dict(
        N=N,
        at_lo=int((P_flat <= 2.0 * eps).sum()),
        near_lo=int((P_flat <= 10.0 * eps).sum()),
        at_hi=int((P_flat >= 1.0 - 2.0 * eps).sum()),
        near_hi=int((P_flat >= 1.0 - 10.0 * eps).sum()),
        quantiles=dict(
            min=float(P_flat.min()),
            q01=float(np.quantile(P_flat, 0.01)),
            q50=float(np.quantile(P_flat, 0.50)),
            q99=float(np.quantile(P_flat, 0.99)),
            max=float(P_flat.max()),
        ),
    )


def top_residual_cells(P, F, n=10):
    """Return list of (multi_idx, |F|, P, clip_status) for the n largest |F|."""
    G = P.shape[0]
    P_flat = P.reshape(-1).astype(np.float64)
    F_abs = np.abs(F).reshape(-1).astype(np.float64)
    eps = float(EPS_OUTER)
    top = np.argsort(F_abs)[-n:][::-1]
    out = []
    for idx in top:
        pv = float(P_flat[idx])
        if pv < 10 * eps:
            status = "LO!"
        elif pv > 1 - 10 * eps:
            status = "HI!"
        else:
            status = "ok"
        multi = tuple(int(x) for x in np.unravel_index(idx, P.shape))
        out.append(dict(idx=multi, abs_F=float(F_abs[idx]), P=pv, status=status))
    return out


def jacobian_svd_summary(J):
    """Return σ spectrum stats, # near-zero σ, condition number."""
    J64 = J.astype(np.float64)
    sigma = np.linalg.svd(J64, compute_uv=False)
    s0 = float(sigma[0])
    sm = float(max(sigma[-1], 1e-300))
    return dict(
        sigma=sigma.astype(np.float64),
        sigma_max=s0,
        sigma_min=float(sigma[-1]),
        cond=s0 / sm,
        n_below_1em6=int((sigma < 1e-6 * s0).sum()),
        n_below_1em8=int((sigma < 1e-8 * s0).sum()),
        n_below_1em10=int((sigma < 1e-10 * s0).sum()),
    )


def f_decomposition(J, F):
    """Decompose F over singular directions of J. Returns ‖F‖ in vs out of
    range(J), the most-reducible (top |c|·σ) and hardest (top |c|/σ) modes.
    """
    J64 = J.astype(np.float64)
    F64 = F.reshape(-1).astype(np.float64)
    U, sigma, Vt = np.linalg.svd(J64, full_matrices=False)
    contrib = U.T @ F64
    F_proj = U @ contrib
    norm_in = float(np.linalg.norm(F_proj))
    norm_out = float(np.linalg.norm(F64 - F_proj))
    norm_total = float(np.linalg.norm(F64))
    abs_c = np.abs(contrib)
    top_red = np.argsort(abs_c * sigma)[-5:][::-1]
    top_hard = np.argsort(abs_c / np.maximum(sigma, 1e-30))[-5:][::-1]
    return dict(
        norm_in_range=norm_in,
        norm_orth=norm_out,
        norm_total=norm_total,
        frac_outside=norm_out / max(norm_total, 1e-300),
        top_reducible=[
            dict(i=int(i), sigma=float(sigma[i]),
                 abs_c=float(abs_c[i]),
                 c_times_sigma=float(abs_c[i] * sigma[i]))
            for i in top_red
        ],
        top_hardest=[
            dict(i=int(i), sigma=float(sigma[i]),
                 abs_c=float(abs_c[i]),
                 c_over_sigma=float(abs_c[i] / max(sigma[i], 1e-30)))
            for i in top_hard
        ],
    )


def analyse(P, F, J=None):
    """Bundle clip stats, top residual cells, and (if J given) SVD diagnostics."""
    out = dict(
        clip=clip_stats(P),
        top_F=top_residual_cells(P, F),
    )
    if J is not None:
        out["svd"] = jacobian_svd_summary(J)
        out["F_decomp"] = f_decomposition(J, F)
    return out


def pretty(diag, log=print):
    c = diag["clip"]
    log(f"  Q1. Clip distribution (N={c['N']}):")
    log(f"      AT lower (≤2·EPS):  {c['at_lo']}")
    log(f"      near lower (≤10·EPS): {c['near_lo']}")
    log(f"      AT upper:           {c['at_hi']}")
    log(f"      near upper:         {c['near_hi']}")
    q = c["quantiles"]
    log(f"      P quantiles min/1%/50%/99%/max = "
        f"{q['min']:.3e} / {q['q01']:.3e} / {q['q50']:.4f} / "
        f"{q['q99']:.4f} / {q['max']:.4f}")
    log(f"  Q2. Top-{len(diag['top_F'])} |F| cells:")
    for r in diag["top_F"]:
        log(f"      {r['idx']}  |F|={r['abs_F']:.3e}  P={r['P']:.4f}  {r['status']}")
    if "svd" in diag:
        s = diag["svd"]
        log(f"  Q3. SVD: σ_max={s['sigma_max']:.3e}  σ_min={s['sigma_min']:.3e}  "
            f"cond={s['cond']:.3e}")
        log(f"      # σ_i/σ_max < 1e-6/1e-8/1e-10 = "
            f"{s['n_below_1em6']} / {s['n_below_1em8']} / {s['n_below_1em10']}")
    if "F_decomp" in diag:
        d = diag["F_decomp"]
        log(f"      ‖F‖ in range(J)={d['norm_in_range']:.3e}  "
            f"orth={d['norm_orth']:.3e}  total={d['norm_total']:.3e}  "
            f"frac-outside={d['frac_outside']:.3e}")
        log(f"      most-reducible modes (top |c|·σ):")
        for r in d["top_reducible"]:
            log(f"        σ[{r['i']}]={r['sigma']:.3e}  |c|={r['abs_c']:.3e}  "
                f"|c|·σ={r['c_times_sigma']:.3e}")
        log(f"      hardest modes (top |c|/σ):")
        for r in d["top_hardest"]:
            log(f"        σ[{r['i']}]={r['sigma']:.3e}  |c|={r['abs_c']:.3e}  "
                f"|c|/σ={r['c_over_sigma']:.3e}")
