"""Strict PR search batch 4: loose γ-homotopy chain → tight Newton polish.

Idea: HANDOFF's reported PR-branch values had Finf ≈ 3-8e-4 — that's
*loose* convergence, what Picard produces at iter ~50-100 from a good
warm-start. Try the chain that produced those CSV rows:

  γ=50 (cold) → Picard ≈ 1e-3 → save P_warm
  γ=30 (warm-start P_warm) → Picard ≈ 1e-3 → save
  …
  γ=3   warm-start at τ=3.4 → Picard ≈ 1e-3 → save

Then: take each γ-step's loose P_warm and run *strict* Newton (1e-12)
from it. If any of these strict-Newton converges to a fixed point
with 1-R² > 0.01, that's the PR branch.

If none, the result is conclusive: the production kernel has only
the FR fixed point at every (γ, τ) we test, regardless of warm-start
provenance.
"""
import time
import pickle
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


G = 11
UMAX = 2.0
TAU = 3.4
GAMMAS_CHAIN = [50, 30, 15, 10, 7, 5, 4, 3.5, 3.2, 3.0]   # the homotopy
LOOSE_TOL = 1e-3
STRICT_TOL = 1e-12

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
Ws = np.array([1.0, 1.0, 1.0])

ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))


def metrics(P):
    p = float(P[ir, jr, lr])
    Phi = rp._phi_map_pchip(P, u, taus, np.array([CURRENT_G]*3), Ws)
    Finf = float(np.abs(P - Phi).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    return p, one_r2, Finf, mu


# Step 1: build the loose-tolerance γ-homotopy chain
chain = {}
P_warm = None
print(f"=== Loose γ-homotopy at τ={TAU}, G={G}, abstol={LOOSE_TOL} ===\n")
for gamma_v in GAMMAS_CHAIN:
    CURRENT_G = float(gamma_v)
    gammas = np.array([gamma_v, gamma_v, gamma_v])
    t0 = time.time()
    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                  maxiters=300, abstol=LOOSE_TOL,
                                  alpha=0.3, P_init=P_warm)
    Finf = float(np.abs(res["residual"]).max())
    P_warm = res["P_star"]
    p, one_r2, _, mu = metrics(P_warm)
    chain[gamma_v] = P_warm.copy()
    print(f"  γ={gamma_v:5g}  iters={len(res['history']):4d}  "
          f"Finf={Finf:.2e}  p={p:.4f}  1-R²={one_r2:.3e}  "
          f"PR-gap={mu[0]-mu[1]:+.4f}  ({time.time()-t0:.1f}s)",
          flush=True)


# Step 2: tight-tolerance Newton polish from each chain link
print(f"\n=== Strict Newton polish (abstol={STRICT_TOL}) ===\n")
for gamma_v, P_seed in chain.items():
    CURRENT_G = float(gamma_v)
    gammas = np.array([gamma_v, gamma_v, gamma_v])
    t0 = time.time()
    try:
        res = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_seed, maxiters=20, abstol=STRICT_TOL,
            lgmres_tol=1e-12, lgmres_maxiter=200)
        P = res["P_star"]; Finf = res["best_Finf"]
    except Exception as ex:
        P = P_seed
        Finf = float(np.abs(P_seed -
                              rp._phi_map_pchip(P_seed, u, taus, gammas, Ws)).max())
    p, one_r2, _, mu = metrics(P)
    if Finf < STRICT_TOL and one_r2 > 0.01:
        print(f"  γ={gamma_v:5g}  Finf={Finf:.2e}  p={p:.4f}  "
              f"1-R²={one_r2:.4e}  PR-gap={mu[0]-mu[1]:+.4f}  PR! "
              f"({time.time()-t0:.1f}s)",
              flush=True)
        fn = f"/home/user/REZN/python/PR_seed_g{gamma_v}_t{TAU}.pkl"
        with open(fn, "wb") as f:
            pickle.dump({"P": P, "taus": taus, "gammas": gammas,
                          "G": G, "umax": UMAX,
                          "Finf": Finf, "1-R²": one_r2}, f)
        print(f"    saved to {fn}", flush=True)
    elif Finf < STRICT_TOL:
        print(f"  γ={gamma_v:5g}  Finf={Finf:.2e}  p={p:.4f}  "
              f"1-R²={one_r2:.4e}  FR (machine prec) "
              f"({time.time()-t0:.1f}s)",
              flush=True)
    else:
        print(f"  γ={gamma_v:5g}  Finf={Finf:.2e}  p={p:.4f}  "
              f"1-R²={one_r2:.4e}  no fp at strict tol "
              f"({time.time()-t0:.1f}s)",
              flush=True)
