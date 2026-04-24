"""Target: drive Finf below 1e-12 at G=11, τ=γ=(3,3,3), UMAX=2.

Uses the best cached P_star (from the G11 logit-PCHIP sweep) as warm
start; if no cache, starts cold. Iterates aggressively through Picard,
Anderson, NK, and perturb-retry variants, tracking the minimum Finf
seen and reporting after each strategy.
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

u = np.linspace(-UMAX, UMAX, G)
taus = rh._as_vec3(TAU)
gammas = rh._as_vec3(GAMMA)
Ws = rh._as_vec3(1.0)


def finf(P):
    Pn = rp._phi_map_pchip(P, u, taus, gammas, Ws)
    return float(np.abs(P - Pn).max()), Pn


def report(tag, P, extra=""):
    f, _ = finf(P)
    print(f"  [{tag:<15}] Finf={f:.3e}  {extra}", flush=True)
    return f


print("numba warmup…", flush=True)
_ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)


# -------- Initial guess ----------
P_init = None
for path in ["/home/user/REZN/python/pchip_G11logit_cache.pkl",
             "/home/user/REZN/python/pchip_cache.pkl"]:
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        best_cand = None
        for e in cache:
            if (abs(e["taus"][0] - TAU) < 1e-9
                and abs(e["gammas"][0] - GAMMA) < 1e-9):
                fc, _ = finf(e["P_star"])
                if best_cand is None or fc < best_cand[0]:
                    best_cand = (fc, e["P_star"])
        if best_cand is not None:
            P_init = best_cand[1].copy()
            print(f"warm start from cache {path}: Finf={best_cand[0]:.3e}",
                  flush=True)
            break
    except Exception:
        pass

if P_init is None:
    print("no warm cache — cold start", flush=True)
    P_init = rh._nolearning_price(u, taus, gammas, Ws)

f0 = report("init", P_init)
best = dict(P=P_init.copy(), finf=f0)


def keep(tag, P):
    f = report(tag, P)
    if f < best["finf"]:
        best["P"] = P.copy()
        best["finf"] = f


# -------- Strategy 1: long Picard ---------
print("\n-- Picard α=1.0, 50000 iters --", flush=True)
t0 = time.time()
res = rp.solve_picard_pchip(G, (TAU,)*3, (GAMMA,)*3, umax=UMAX,
                             maxiters=50000, abstol=1e-16, alpha=1.0,
                             P_init=best["P"].copy())
print(f"  time={time.time()-t0:.1f}s  last PhiI={res['history'][-1]:.3e}  "
      f"min PhiI={min(res['history']):.3e}", flush=True)
keep("picard-long", res["P_star"])


# -------- Strategy 2: Anderson m=10, 5000 iters ---------
print("\n-- Anderson m=10, 5000 iters --", flush=True)
t0 = time.time()
res = rp.solve_anderson_pchip(G, (TAU,)*3, (GAMMA,)*3, umax=UMAX,
                               maxiters=5000, abstol=1e-16, m_window=10,
                               damping=1.0, P_init=best["P"].copy())
print(f"  time={time.time()-t0:.1f}s  last PhiI={res['history'][-1]:.3e}  "
      f"min PhiI={min(res['history']):.3e}", flush=True)
keep("anderson-m10", res["P_star"])


# -------- Strategy 3: Newton-Krylov ----------
print("\n-- Newton-Krylov maxiter=80 --", flush=True)

def F(x):
    P = x.reshape(G, G, G)
    return x - rp._phi_map_pchip(P, u, taus, gammas, Ws).reshape(-1)

x0 = np.clip(best["P"], 1e-9, 1-1e-9).reshape(-1)
t0 = time.time()
try:
    sol = newton_krylov(F, x0, f_tol=1e-16, rdiff=1e-8, method="lgmres",
                        maxiter=80, verbose=False)
except NoConvergence as e:
    sol = np.asarray(e.args[0])
print(f"  time={time.time()-t0:.1f}s", flush=True)
keep("NK", np.clip(sol.reshape(G, G, G), 1e-9, 1-1e-9))


# -------- Strategy 4: perturb + NK retries with different seeds ----
print("\n-- Perturb + NK retries --", flush=True)
for seed_rng in range(5):
    rng = np.random.default_rng(seed_rng)
    for sig in [1e-12, 1e-11, 1e-10, 1e-9]:
        L = np.log(best["P"] / (1.0 - best["P"]))
        L_p = L + rng.standard_normal(L.shape) * sig
        P_p = 1.0 / (1.0 + np.exp(-L_p))
        P_p = np.clip(P_p, 1e-9, 1-1e-9)
        try:
            sol = newton_krylov(F, P_p.reshape(-1), f_tol=1e-16, rdiff=1e-8,
                                method="lgmres", maxiter=40, verbose=False)
        except NoConvergence as e:
            sol = np.asarray(e.args[0])
        P_try = np.clip(sol.reshape(G, G, G), 1e-9, 1-1e-9)
        keep(f"NK-perturb-s{seed_rng}-σ{sig:.0e}", P_try)
        if best["finf"] < 1e-12:
            break
    if best["finf"] < 1e-12:
        break


# -------- Strategy 5: Alternating Picard/Anderson ----------
if best["finf"] > 1e-12:
    print("\n-- Alternating Picard-Anderson rounds --", flush=True)
    for rd in range(5):
        res = rp.solve_picard_pchip(G, (TAU,)*3, (GAMMA,)*3, umax=UMAX,
                                     maxiters=20000, abstol=1e-16, alpha=1.0,
                                     P_init=best["P"].copy())
        keep(f"round{rd}-picard", res["P_star"])
        if best["finf"] < 1e-12:
            break
        res = rp.solve_anderson_pchip(G, (TAU,)*3, (GAMMA,)*3, umax=UMAX,
                                       maxiters=3000, abstol=1e-16,
                                       m_window=15, damping=1.0,
                                       P_init=best["P"].copy())
        keep(f"round{rd}-anderson", res["P_star"])
        if best["finf"] < 1e-12:
            break


print(f"\n=== FINAL best Finf = {best['finf']:.3e} ===", flush=True)
