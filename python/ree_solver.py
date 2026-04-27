"""Common REE solver helper for the full-equilibrium figures.

Wraps the production PCHIP+contour kernel with Picard+Anderson+continuation
heuristics that give converged P_star reliably across the (τ, γ) range
needed for Figs 6, 7, 8, 9, 10.

Strategy:
  1. Picard α=0.3 from no-learning seed (or warm-start from a previous
     solve via P_init).  Modest abstol — we just need to get into the
     basin.
  2. Anderson m=8 polish from Picard's best iterate, target abstol
     down to 1e-6.
  3. Return whichever solver delivered the smaller Finf.

Returns a dict with P_star, Finf, the converged μ at one reference
realisation, 1-R², and the solver iters.
"""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import rezn_het as rh


def _logit(p):
    return np.log(p / (1.0 - p))


def solve_REE(taus, gammas, G=11, umax=2.0,
                P_init=None,
                picard_iters=600, picard_alpha=0.3,
                anderson_iters=200, anderson_m=8,
                abstol=1e-6,
                verbose=False):
    """Solve the full-equilibrium REE on the production PCHIP+contour
    kernel.  Returns the smaller-Finf of (Picard, Anderson polish).
    """
    taus_arr   = np.asarray(taus,   float)
    gammas_arr = np.asarray(gammas, float)

    res_p = rp.solve_picard_pchip(
        G, taus_arr, gammas_arr, umax=umax,
        maxiters=picard_iters, abstol=abstol, alpha=picard_alpha,
        P_init=P_init)
    Finf_p = float(np.abs(res_p["residual"]).max())

    res_a = rp.solve_anderson_pchip(
        G, taus_arr, gammas_arr, umax=umax,
        maxiters=anderson_iters, abstol=abstol, m_window=anderson_m,
        P_init=res_p["P_star"])
    Finf_a = float(np.abs(res_a["residual"]).max())

    if Finf_a <= Finf_p:
        P_star = res_a["P_star"]; Finf = Finf_a; src = "Anderson"
        iters = (len(res_p["history"]), len(res_a["history"]))
    else:
        P_star = res_p["P_star"]; Finf = Finf_p; src = "Picard"
        iters = (len(res_p["history"]), len(res_a["history"]))

    if verbose:
        print(f"  Picard {iters[0]} / Finf={Finf_p:.2e},  "
              f"Anderson {iters[1]} / Finf={Finf_a:.2e}  "
              f"-> {src}")

    return {
        "P_star": P_star,
        "Finf":   Finf,
        "iters":  iters,
        "src":    src,
        "taus":   taus_arr,
        "gammas": gammas_arr,
    }


def diagnostics(P, taus, umax=2.0):
    """Compute the metrics each figure needs from a converged P."""
    G = P.shape[0]
    u = np.linspace(-umax, umax, G)
    one_r2 = float(rh.one_minus_R2(P, u, taus))
    i = int(np.argmin(np.abs(u - 1.0)))
    j = int(np.argmin(np.abs(u + 1.0)))
    l = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[i, j, l])
    mu = rh.posteriors_at(i, j, l, p, P, u, np.asarray(taus, float))
    return {
        "one_minus_R2": one_r2,
        "p_at_realisation": p,
        "mu_at_realisation": tuple(float(m) for m in mu),
        "PR_gap": float(mu[0] - mu[1]),
    }
