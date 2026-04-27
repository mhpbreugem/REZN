"""Strict PR search batch 5: global search via simulated annealing on a
low-dimensional ansatz.

Observation: Newton/Picard from any seed I have tried lands in the FR
basin. The FR fixed point may be the unique fixed point of Φ at the
production grid resolution. To reject this conclusion definitively, we
need to scan a *global* neighborhood of P-tensor space, not local
basins.

Approach: parameterize P via a small set of polynomial coefficients in
the symmetric statistics of (u_1, u_2, u_3):

    logit(P[i,j,l]) = α·S₁(u)
                    + β·S₁(u)·S₂(u)
                    + γ·S₃(u)
                    + δ·(S₁(u))³

where  S₁ = τ·Σu_k,  S₂ = Σ(u_k − ū)²,  S₃ = Σu_k·u_ℓ·u_m  (third moment).

This is a 4-parameter family. The FR fixed point corresponds to
(α=1, β=γ=δ=0). Any other fixed point lives at a different (α, β, γ, δ).

Cost: ||P(α,β,γ,δ) - Φ(P(α,β,γ,δ))||_∞. Minimize globally with scipy
dual_annealing — explores far from the FR basin.

If the global minimum of cost is achieved at (1, 0, 0, 0) only,
FR is the unique fixed point. If global min is achieved elsewhere
with cost < 1e-6 and 1-R² > 0.01, that's a NEW PR fixed point.
"""
import time
import pickle
import numpy as np
from scipy.optimize import dual_annealing
import rezn_pchip as rp
import rezn_het as rh


CONFIGS = [
    # (G, τ, γ, label)
    (11, 3.4,  3.0, "γ=3 τ=3.4"),
    (11, 2.0,  0.5, "γ=0.5 τ=2"),
    (11, 5.0,  3.0, "γ=3 τ=5"),
    (11, 5.0,  1.0, "γ=1 τ=5"),
]
UMAX = 2.0


def build_P(coefs, G, tau, u):
    alpha, beta, gamma_p, delta = coefs
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                S1 = tau * (u[i] + u[j] + u[l])
                ubar = (u[i] + u[j] + u[l]) / 3.0
                S2 = (u[i]-ubar)**2 + (u[j]-ubar)**2 + (u[l]-ubar)**2
                S3 = u[i]*u[j]*u[l]
                lp = (alpha*S1 + beta*S1*S2 + gamma_p*S3*tau
                      + delta*S1**3 / 100.0)        # /100 keeps δ near O(1)
                P[i, j, l] = 1.0 / (1.0 + np.exp(-lp))
    return np.clip(P, 1e-9, 1 - 1e-9)


def cost(coefs, G, tau, gamma, u, taus, gammas, Ws):
    P = build_P(coefs, G, tau, u)
    Phi = rp._phi_map_pchip(P, u, taus, gammas, Ws)
    return float(np.abs(P - Phi).max())


def report(P, u, taus):
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[ir, jr, lr])
    Phi = rp._phi_map_pchip(P, u, taus,
                              np.array([CURRENT_GAMMA]*3),
                              np.array([1.0,1.0,1.0]))
    Finf = float(np.abs(P - Phi).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    return p, one_r2, Finf, mu[0]-mu[1]


for G, tau, gamma, label in CONFIGS:
    CURRENT_GAMMA = gamma
    print(f"\n=== {label}: dual_annealing on 4-param ansatz ===", flush=True)
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([tau, tau, tau])
    gammas = np.array([gamma, gamma, gamma])
    Ws = np.array([1.0, 1.0, 1.0])

    bounds = [(0.0, 2.5),    # α (FR-aggregator slope, 1.0 is FR)
              (-2.0, 2.0),   # β (asymmetry coefficient)
              (-2.0, 2.0),   # γ_p (third-moment coefficient)
              (-2.0, 2.0)]   # δ (cubic in S1)

    t0 = time.time()
    result = dual_annealing(
        cost, bounds=bounds,
        args=(G, tau, gamma, u, taus, gammas, Ws),
        maxiter=200, seed=42, x0=[1.0, 0.0, 0.0, 0.0])
    coefs = result.x
    print(f"  best (α, β, γ, δ) = ({coefs[0]:.4f}, {coefs[1]:.4f}, "
          f"{coefs[2]:.4f}, {coefs[3]:.4f})", flush=True)
    P = build_P(coefs, G, tau, u)
    p, one_r2, Finf, pr_gap = report(P, u, taus)
    print(f"  Finf={Finf:.3e}  p={p:.4f}  1-R²={one_r2:.4e}  "
          f"PR-gap={pr_gap:+.4f}  ({time.time()-t0:.1f}s)", flush=True)

    if Finf < 1e-3 and one_r2 > 0.01 and abs(coefs[1]) + abs(coefs[2]) + abs(coefs[3]) > 0.05:
        # Looks like a non-FR fixed point — refine with strict Newton
        print(f"  candidate non-FR fp! refining with strict Newton …",
              flush=True)
        import pchip_jacobian as pj
        try:
            res = pj.solve_newton(
                G, taus, gammas, umax=UMAX,
                P_init=P, maxiters=15, abstol=1e-12,
                lgmres_tol=1e-12, lgmres_maxiter=80)
            P_n = res["P_star"]
            p_n, r_n, finf_n, gap_n = report(P_n, u, taus)
            print(f"  after Newton: Finf={finf_n:.3e}  p={p_n:.4f}  "
                  f"1-R²={r_n:.4e}  PR-gap={gap_n:+.4f}", flush=True)
            if finf_n < 1e-12 and r_n > 0.01:
                fn = (f"/home/user/REZN/python/PR_seed_g{gamma}_t{tau}_"
                       f"globopt.pkl")
                with open(fn, "wb") as f:
                    pickle.dump({"P": P_n, "taus": taus, "gammas": gammas,
                                  "G": G, "umax": UMAX,
                                  "Finf": finf_n, "1-R²": r_n,
                                  "coefs": coefs}, f)
                print(f"    PR FOUND! saved to {fn}", flush=True)
        except Exception as ex:
            print(f"  refinement FAILED: {ex}", flush=True)
