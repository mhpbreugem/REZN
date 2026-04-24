"""Push Finf < 1e-12 with central-difference NK + Aitken Δ² extrapolation.

Strategies:
  A. Aitken Δ² applied after Picard runs — squeezes extra digits past
     Picard's noise floor by extrapolating the linear-convergence tail.
  B. Custom Newton-Krylov with CENTRAL finite-difference Jacobian
     (truncation O(ε²) vs forward-diff O(ε); allows larger ε with less
     roundoff). Built on scipy.sparse.linalg.lgmres.

Plus: alternating cycles of (Picard → Aitken → central-NK).
"""
import pickle
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

import rezn_het as rh
import rezn_pchip as rp

G = 11
UMAX = 2.0
TAU = 3.0
GAMMA = 3.0
N = G ** 3
STATUS = "/home/user/REZN/python/sweep_status.txt"

u = np.linspace(-UMAX, UMAX, G)
taus = rh._as_vec3(TAU)
gammas = rh._as_vec3(GAMMA)
Ws = rh._as_vec3(1.0)


def F_flat(x):
    P = x.reshape(G, G, G)
    return x - rp._phi_map_pchip(P, u, taus, gammas, Ws).reshape(-1)


def finf(P):
    return float(np.abs(P - rp._phi_map_pchip(P, u, taus, gammas, Ws)).max())


def status(prefix):
    with open(STATUS, "w") as f:
        f.write(f"{prefix} best={best['finf']:.3e}\n")


# -------- Aitken Δ² acceleration on Picard iterates ------------------
# Run Picard, collect the LAST 3 iterates X_{n}, X_{n+1}, X_{n+2}.
# For each cell, apply: X̂ = X_n - (X_{n+1} - X_n)² / (X_{n+2} - 2X_{n+1} + X_n)
# Returns the accelerated estimate. Robust to NaN: fall back to last iterate.
def picard_with_aitken(P_init, n_iter=300):
    Pcur = np.clip(P_init, 1e-9, 1-1e-9).copy()
    Pprev = None
    Pprev2 = None
    best_local = (Pcur.copy(), finf(Pcur))
    for it in range(n_iter):
        Pnew = rp._phi_map_pchip(Pcur, u, taus, gammas, Ws)
        Pnew = np.clip(Pnew, 1e-9, 1-1e-9)
        f = float(np.abs(Pnew - Pcur).max())
        if f < best_local[1]:
            best_local = (Pnew.copy(), f)
        if Pprev2 is not None:
            d1 = Pcur - Pprev
            d2 = Pnew - 2 * Pcur + Pprev
            with np.errstate(divide='ignore', invalid='ignore'):
                P_aitken = np.where(np.abs(d2) > 1e-30,
                                     Pprev - d1 * d1 / d2,
                                     Pnew)
            P_aitken = np.clip(P_aitken, 1e-9, 1-1e-9)
            if np.all(np.isfinite(P_aitken)):
                f_a = finf(P_aitken)
                if f_a < best_local[1]:
                    best_local = (P_aitken.copy(), f_a)
        Pprev2 = Pprev
        Pprev = Pcur.copy()
        Pcur = Pnew.copy()
        if (it+1) % 25 == 0:
            with open(STATUS, "w") as fp:
                fp.write(f"Picard+Aitken iter={it+1}/{n_iter} "
                         f"step={f:.3e} best={best_local[1]:.3e}\n")
    return best_local


# -------- Central-difference Newton-Krylov ----------------------------
def central_nk(P_init, eps=1e-6, outer_iter=20, inner_iter=40, target=1e-13):
    x = np.clip(P_init, 1e-9, 1-1e-9).reshape(-1).copy()
    F0 = F_flat(x)
    f_now = float(np.abs(F0).max())
    best_local = (x.reshape(G, G, G).copy(), f_now)
    print(f"  central-NK init Finf={f_now:.3e}", flush=True)

    for k in range(outer_iter):
        # Build J·v operator using CENTRAL differences:
        #   J·v ≈ (F(x + εv) - F(x - εv)) / (2ε)
        def matvec(v):
            v = np.asarray(v, float)
            return (F_flat(x + eps * v) - F_flat(x - eps * v)) / (2.0 * eps)

        L = LinearOperator((N, N), matvec=matvec, dtype=float)
        # Solve J · d = -F
        d, info = lgmres(L, -F0, atol=target * 0.1, rtol=1e-12,
                         maxiter=inner_iter)
        x_try = np.clip(x + d, 1e-9, 1-1e-9)
        F_try = F_flat(x_try)
        f_try = float(np.abs(F_try).max())

        # Simple Armijo backtrack if step makes things worse
        alpha = 1.0
        while f_try > f_now and alpha > 1e-3:
            alpha *= 0.5
            x_try = np.clip(x + alpha * d, 1e-9, 1-1e-9)
            F_try = F_flat(x_try)
            f_try = float(np.abs(F_try).max())

        x = x_try
        F0 = F_try
        f_now = f_try
        if f_now < best_local[1]:
            best_local = (x.reshape(G, G, G).copy(), f_now)
        with open(STATUS, "w") as fp:
            fp.write(f"central-NK iter={k+1}/{outer_iter} eps={eps:.0e} "
                     f"alpha={alpha:.3f} Finf={f_now:.3e} "
                     f"best={best_local[1]:.3e}\n")
        print(f"  central-NK iter {k+1:>3} eps={eps:.0e} α={alpha:.3f} "
              f"Finf={f_now:.3e}", flush=True)
        if f_now < target:
            break
    return best_local


# -------- Main driver --------------------------------------------------
print("numba warmup…", flush=True)
_ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

# Warm start from cache
P_init = None
try:
    with open("/home/user/REZN/python/pchip_G11logit_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    same_g = [e for e in cache if abs(e["gammas"][0] - GAMMA) < 1e-9]
    same_g.sort(key=lambda e: abs(e["taus"][0] - TAU))
    if same_g:
        P_init = same_g[0]["P_star"].copy()
        print(f"warm start τ={same_g[0]['taus'][0]:.3f}: "
              f"Finf={finf(P_init):.3e}", flush=True)
except Exception as e:
    print(f"cache load failed: {e}", flush=True)

if P_init is None:
    P_init = rh._nolearning_price(u, taus, gammas, Ws)

best = {"P": P_init.copy(), "finf": finf(P_init)}
print(f"[init] Finf={best['finf']:.3e}", flush=True)


for round_ in range(20):
    print(f"\n=== Round {round_} ===  current best={best['finf']:.3e}",
          flush=True)
    # Step 1: Picard with Aitken
    P_a, f_a = picard_with_aitken(best["P"], n_iter=300)
    print(f"  picard+aitken → Finf={f_a:.3e}", flush=True)
    if f_a < best["finf"]:
        best["P"] = P_a; best["finf"] = f_a
    if best["finf"] < 1e-13:
        break

    # Step 2: Central-difference NK with eps tuned to F-noise
    # Optimal eps ≈ sqrt(noise) — try several
    for eps in [3e-7, 1e-6, 3e-6, 1e-5]:
        P_nk, f_nk = central_nk(best["P"], eps=eps, outer_iter=10,
                                 inner_iter=40, target=1e-13)
        print(f"  central-NK eps={eps:.0e} → Finf={f_nk:.3e}", flush=True)
        if f_nk < best["finf"]:
            best["P"] = P_nk; best["finf"] = f_nk
        if best["finf"] < 1e-13:
            break
    if best["finf"] < 1e-13:
        break

print(f"\n=== FINAL Finf = {best['finf']:.3e} ===", flush=True)
