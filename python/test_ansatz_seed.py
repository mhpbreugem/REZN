"""Closed-form ansatz seed for the PR branch.

Theory: at the documented PR fixed point (γ=3, τ=3.4, cell (1,-1,1)),
logit(p_PR) = 0.54 while T* = 3.4. The slope of logit(p) vs T* is
~0.16. The ansatz P_seed = sigmoid(α·T*) with α ∈ (0, 1) interpolates
between α=1 (CARA-FR) and α=0 (no information). Plug it into Newton
and see if any α-value lands in the PR basin.

If yes, we have a constructive seed and can sweep τ and γ.
"""
from __future__ import annotations
import time
import numpy as np
import pchip_jacobian as pj
import rezn_pchip as rp
import rezn_het as rh

G, UMAX, TAU, GAMMA = 11, 2.0, 3.4, 3.0

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])
Ws = np.array([1.0, 1.0, 1.0])

ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))


def build_seed(alpha):
    """P_seed[i,j,l] = sigmoid(α·T*[i,j,l]), T* = τ·Σu."""
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T_star = TAU * (u[i] + u[j] + u[l])
                P[i, j, l] = 1.0 / (1.0 + np.exp(-alpha * T_star))
    return np.clip(P, 1e-9, 1 - 1e-9)


def report(name, P, t):
    p = float(P[ir, jr, lr])
    finf = float(np.abs(P - rp._phi_map_pchip(P, u, taus, gammas, Ws)).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr = "PR!" if (one_r2 > 0.01 and finf < 1e-3) else "FR/?"
    print(f"  [{name:18s}]  p={p:.4f}  1-R²={one_r2:.4e}  "
          f"PR-gap={mu[0]-mu[1]:+.4f}  Finf={finf:.2e}  [{pr}]  "
          f"({t:.1f}s)", flush=True)


print(f"=== Ansatz seed at γ={GAMMA}, τ={TAU}, G={G} ===")
print(f"Target: PR fixed point with logit(p_PR) ≈ 0.16·T* (slope from")
print(f"the SESSION_SUMMARY at cell (1,-1,1))\n")
print("Step 1: report each ansatz seed (no solve):")
for alpha in (0.10, 0.16, 0.20, 0.25, 0.33, 0.50):
    P_seed = build_seed(alpha)
    report(f"seed α={alpha}", P_seed, 0.0)

print("\nStep 2: run Picard α=0.3 from each seed (1500 iters):")
for alpha in (0.10, 0.16, 0.20, 0.25, 0.33, 0.50):
    t0 = time.time()
    P_seed = build_seed(alpha)
    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                  maxiters=1500, abstol=1e-7, alpha=0.3,
                                  P_init=P_seed)
    report(f"Picard α={alpha}", res["P_star"], time.time() - t0)

print("\nStep 3: run analytic Newton from each seed (10 iters):")
for alpha in (0.10, 0.16, 0.20, 0.25, 0.33, 0.50):
    t0 = time.time()
    P_seed = build_seed(alpha)
    try:
        res = pj.solve_newton(G, taus, gammas, umax=UMAX,
                                P_init=P_seed, maxiters=10, abstol=1e-7,
                                lgmres_tol=1e-7, lgmres_maxiter=60)
        report(f"Newton α={alpha}", res["P_star"], time.time() - t0)
    except Exception as ex:
        print(f"  [Newton α={alpha}]  FAIL: {ex}", flush=True)
