"""Alternating Newton-Picard with perturb, targeting Finf < 1e-12.

Strategy: 200 iters Picard → NK → perturb σ=1e-11 → NK → 200 iters Picard → NK → ...
Keep best iterate seen. Up to 40 rounds.
"""
import pickle
import sys
import time
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

import rezn_het as rh
import rezn_pchip as rp

G = 11
UMAX = 2.0
TAU = 3.0
GAMMA = 3.0
STATUS = "/home/user/REZN/python/sweep_status.txt"

u = np.linspace(-UMAX, UMAX, G)
taus = rh._as_vec3(TAU)
gammas = rh._as_vec3(GAMMA)
Ws = rh._as_vec3(1.0)


def finf(P):
    return float(np.abs(P - rp._phi_map_pchip(P, u, taus, gammas, Ws)).max())


def F(x):
    P = x.reshape(G, G, G)
    return x - rp._phi_map_pchip(P, u, taus, gammas, Ws).reshape(-1)


def status(prefix):
    with open(STATUS, "w") as f:
        f.write(f"{prefix} best={best['finf']:.3e}\n")


print("numba warmup…", flush=True)
_ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)


# Load warm start from cache
P_init = None
try:
    with open("/home/user/REZN/python/pchip_G11logit_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    same_g = [e for e in cache if abs(e["gammas"][0] - GAMMA) < 1e-9]
    same_g.sort(key=lambda e: abs(e["taus"][0] - TAU))
    if same_g:
        P_init = same_g[0]["P_star"].copy()
        print(f"warm start from τ={same_g[0]['taus'][0]:.3f}: "
              f"Finf={finf(P_init):.3e}", flush=True)
except Exception as e:
    print(f"cache load failed: {e}", flush=True)

if P_init is None:
    P_init = rh._nolearning_price(u, taus, gammas, Ws)

best = {"P": P_init.copy(), "finf": finf(P_init)}
print(f"[init] Finf={best['finf']:.3e}", flush=True)
status("init")


def keep(tag, P, round_=None):
    P = np.clip(P, 1e-9, 1 - 1e-9)
    f = finf(P)
    marker = "*" if f < best["finf"] else " "
    rtag = f"rd{round_:02d}" if round_ is not None else "---"
    print(f"  [{rtag}] {tag:<12} Finf={f:.3e} {marker}", flush=True)
    if f < best["finf"]:
        best["P"] = P.copy()
        best["finf"] = f
    status(f"{rtag} {tag} Finf={f:.3e} best={best['finf']:.3e}")


rng = np.random.default_rng(0)

for rd in range(40):
    t0 = time.time()
    status(f"rd{rd:02d} Picard start")
    res = rp.solve_picard_pchip(G, (TAU,)*3, (GAMMA,)*3, umax=UMAX,
                                 maxiters=200, abstol=1e-16, alpha=1.0,
                                 P_init=best["P"].copy(),
                                 status_path=STATUS, status_every=25,
                                 status_prefix=f"rd{rd:02d}-P")
    keep("picard-200", res["P_star"], rd)
    if best["finf"] < 1e-12:
        break

    status(f"rd{rd:02d} NK start")
    x0 = np.clip(best["P"], 1e-9, 1-1e-9).reshape(-1)
    try:
        sol = newton_krylov(F, x0, f_tol=1e-16, rdiff=1e-8,
                            method="lgmres", maxiter=25, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
    keep("NK-25", sol.reshape(G, G, G), rd)
    if best["finf"] < 1e-12:
        break

    # Perturb in logit space and NK again
    L = np.log(best["P"] / (1 - best["P"]))
    sig = min(1e-11, best["finf"] * 0.1)
    L_p = L + rng.standard_normal(L.shape) * sig
    P_p = 1.0 / (1.0 + np.exp(-L_p))
    status(f"rd{rd:02d} NK-perturb σ={sig:.0e}")
    try:
        sol = newton_krylov(F, P_p.reshape(-1), f_tol=1e-16, rdiff=1e-8,
                            method="lgmres", maxiter=25, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
    keep(f"NK-perturb", sol.reshape(G, G, G), rd)
    if best["finf"] < 1e-12:
        break

    print(f"  rd{rd:02d} time={time.time()-t0:.1f}s  best={best['finf']:.3e}",
          flush=True)

print(f"\n=== FINAL Finf = {best['finf']:.3e} ===", flush=True)
