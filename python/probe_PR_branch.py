"""Probe: does a PR-branch REE exist at γ=1?

Strategy: at γ=1, τ=3, run analytic-Jacobian Newton from several
off-FR initialisations and see whether any converge to an equilibrium
with 1−R² substantially above the near-FR floor.

Inits tried:
  • no-learning (the FR-attracted seed used by Picard) — control
  • logit-scaled no-learning (logit ×α for α ∈ {1.5, 2.0, 3.0}) —
    moves seed toward "more aggressive aggregation"
  • random perturbation around no-learning (3 seeds)
  • backward-style: read PR-branch P at γ=3 from a re-converged
    backward-snap tensor and homotopy γ=3→γ=1

Reports per-init:
  Newton iters, final Finf, final 1−R², runtime.
"""
from __future__ import annotations
import numpy as np
import time
import pchip_jacobian as pj
import rezn_het as rh
import rezn_pchip as rp


G        = 11
UMAX     = 2.0
TAU      = 3.0
GAMMA    = 1.0


def one_minus_R2(P, u, taus):
    G_ = u.shape[0]
    y = np.log(np.clip(P, 1e-12, 1 - 1e-12) /
                (1 - np.clip(P, 1e-12, 1 - 1e-12))).reshape(-1)
    T = np.empty(G_ ** 3)
    k = 0
    for i in range(G_):
        for j in range(G_):
            for l in range(G_):
                T[k] = taus[0] * u[i] + taus[1] * u[j] + taus[2] * u[l]
                k += 1
    yc = y - y.mean(); Tc = T - T.mean()
    Syy = float((yc * yc).sum()); STT = float((Tc * Tc).sum())
    SyT = float((yc * Tc).sum())
    return 1.0 - (SyT * SyT) / (Syy * STT) if (Syy > 0 and STT > 0) else 0.0


def run(name, P_init, u, taus, gammas):
    t0 = time.time()
    try:
        res = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_init, maxiters=20, abstol=1e-9,
            lgmres_tol=1e-7, lgmres_maxiter=120)
        P = res["P_star"]
        finf = res["best_Finf"]
        iters = len(res["history"])
    except Exception as ex:
        print(f"  [{name}] FAIL: {ex}")
        return
    R2 = one_minus_R2(P, u, taus)
    print(f"  [{name}] iters={iters:2d}  Finf={finf:.2e}  "
          f"1-R²={R2:.4e}  ({time.time()-t0:.1f}s)")


def main():
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([TAU, TAU, TAU])
    gammas = np.array([GAMMA, GAMMA, GAMMA])
    Ws = np.array([1.0, 1.0, 1.0])

    P_NL = rh._nolearning_price(u, taus, gammas, Ws)
    print(f"γ={GAMMA}, τ={TAU}, G={G}")
    print(f"no-learning 1-R² (baseline): "
           f"{one_minus_R2(P_NL, u, taus):.4e}")
    print()

    # 1. Plain no-learning seed
    run("nolearning", P_NL, u, taus, gammas)

    # 2. Logit-scaled (push toward steeper / more confident prices)
    eps = 1e-9
    L_NL = np.log(np.clip(P_NL, eps, 1-eps) / (1 - np.clip(P_NL, eps, 1-eps)))
    L_mean = L_NL.mean()
    for alpha in (0.5, 1.5, 2.0, 3.0, 0.3):
        L_seed = L_mean + alpha * (L_NL - L_mean)
        P_seed = 1.0 / (1.0 + np.exp(-L_seed))
        P_seed = np.clip(P_seed, eps, 1 - eps)
        run(f"logit-scaled α={alpha}", P_seed, u, taus, gammas)

    # 3. Random perturbations
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        eps_perturb = 0.1
        delta = eps_perturb * rng.standard_normal(P_NL.shape)
        P_seed = np.clip(P_NL + delta, eps, 1 - eps)
        run(f"rand σ=0.1 seed={seed}", P_seed, u, taus, gammas)

    # 4. Try a much higher γ baseline as warm start for γ=1 (homotopy)
    print()
    print("γ-homotopy 50 → 1:")
    P = P_NL.copy()
    for g in (50, 30, 20, 10, 5, 3, 2, 1):
        gammas_g = np.array([float(g)] * 3)
        try:
            res = pj.solve_newton(
                G, taus, gammas_g, umax=UMAX,
                P_init=P, maxiters=15, abstol=1e-9,
                lgmres_tol=1e-7, lgmres_maxiter=120)
            P = res["P_star"]
            R2 = one_minus_R2(P, u, gammas_g)
            print(f"  γ={g:>4}  Finf={res['best_Finf']:.2e}  "
                  f"1-R²={R2:.4e}")
        except Exception as ex:
            print(f"  γ={g:>4}  FAIL: {ex}")
            break


if __name__ == "__main__":
    main()
