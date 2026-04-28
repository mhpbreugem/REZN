"""Batch 9 v6 — FD Jacobian (numba f64 Φ stored in f128) + mpmath LU solve.

Track A: replace v5's iterative-refinement (f64 LU + f128 residual) with
a true mpmath LU solve at 40-digit precision (~ float128 precision +
margin). Tests whether the linear solver was the binding constraint.

Pipeline:
  1. Load v5's best iterate (Finf 2.75e-3, 1-R² 21%) from PR_seed_b9_v5_BEST.pkl
  2. Each Newton iter:
     - F = P − Φ(P)  (numba f64 Φ, F kept in f128)
     - J via central FD with h=1e-12, stored in f128
     - Convert J, F to mpmath matrices at 40-digit precision
     - Solve J · dP = −F via mpmath.lu_solve (true high-precision)
     - Convert dP back to float128, Armijo backtrack on ‖F‖∞ in f128
  3. 12 iters max. Save tensor on improvement.

The diagnostic is direct: if mpmath LU breaks through 2.75e-3 where
f64 LU + iter-refine could not, precision was the limit and we earn
the day-long full-f128 Φ rewrite. If mpmath also stalls, conditioning
is fundamental and full f128 won't help.
"""
import time
import pickle
import numpy as np
import mpmath as mp
import rezn_het as rh


SEED_FILE = "/home/user/REZN/python/PR_seed_b9_v5_BEST.pkl"
EPS_OUTER = 1e-9
H_FD       = np.float128("1e-12")
NEWTON_ITERS = 12
HEARTBEAT  = 10.0
MP_DPS     = 40       # mpmath precision = 40 decimal digits


with open(SEED_FILE, "rb") as f:
    seed = pickle.load(f)
P0 = (seed["P_f128"] if "P_f128" in seed else seed["P"]).astype(np.float128).copy()
TAUS = seed["taus"]
GAMMAS = seed["gammas"]
G = seed["G"]
UMAX = seed["umax"]
WS = np.array([1.0, 1.0, 1.0])
u = np.linspace(-UMAX, UMAX, G)


print(f"=== v6 FD-J + mpmath LU (dps={MP_DPS}) ===", flush=True)
print(f"  config: γ={GAMMAS}  τ={TAUS}  G={G}", flush=True)
print(f"  starting from saved iterate: Finf={seed['Finf']:.3e}  "
      f"1-R²={seed['1-R²']:.3e}", flush=True)

mp.mp.dps = MP_DPS


def Phi128(P128):
    return rh._phi_map(P128.astype(np.float64),
                         u, TAUS, GAMMAS, WS).astype(np.float128)


def F_eval(P128):
    return P128 - Phi128(P128)


def build_FD_jacobian(P128, h=H_FD):
    N = P128.size
    J = np.empty((N, N), dtype=np.float128)
    P_flat = P128.reshape(-1)
    t0 = time.time()
    t_last = 0.0
    for k in range(N):
        Pp = P_flat.copy(); Pp[k] += h
        Pm = P_flat.copy(); Pm[k] -= h
        Fp = F_eval(np.clip(Pp.reshape(P128.shape),
                              np.float128(EPS_OUTER),
                              np.float128(1 - EPS_OUTER))).reshape(-1)
        Fm = F_eval(np.clip(Pm.reshape(P128.shape),
                              np.float128(EPS_OUTER),
                              np.float128(1 - EPS_OUTER))).reshape(-1)
        J[:, k] = (Fp - Fm) / (np.float128(2.0) * h)
        elapsed = time.time() - t0
        if elapsed - t_last >= HEARTBEAT:
            print(f"    [FD-J] col {k+1}/{N}  elapsed={elapsed:.1f}s",
                  flush=True)
            t_last = elapsed
    print(f"    [FD-J] complete in {time.time()-t0:.1f}s", flush=True)
    return J


def solve_mpmath(J128, b128):
    """Solve J x = b using mpmath at MP_DPS digit precision."""
    N = J128.shape[0]
    print(f"    converting to mpmath ({MP_DPS} dps)...", flush=True)
    t0 = time.time()
    Jm = mp.matrix(N, N)
    bm = mp.matrix(N, 1)
    for i in range(N):
        for j in range(N):
            Jm[i, j] = mp.mpf(str(J128[i, j]))
        bm[i, 0] = mp.mpf(str(b128[i]))
    print(f"    convert done in {time.time()-t0:.1f}s; "
          f"calling mp.lu_solve (this is the slow step)...",
          flush=True)
    t0 = time.time()
    xm = mp.lu_solve(Jm, bm)
    print(f"    mp.lu_solve done in {time.time()-t0:.1f}s",
          flush=True)
    x128 = np.empty(N, dtype=np.float128)
    for i in range(N):
        x128[i] = np.float128(str(xm[i]))
    return x128


P = np.clip(P0, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
F = F_eval(P)
Finf = float(np.abs(F).max())
print(f"  initial Finf={Finf:.3e}", flush=True)

best_Finf = Finf
P_best = P.copy()

for it in range(NEWTON_ITERS):
    t_iter = time.time()
    print(f"\n=== Newton-FD-mpmath iter {it} ===", flush=True)
    print(f"  building Jacobian (FD, h={float(H_FD):.0e})...",
          flush=True)
    J = build_FD_jacobian(P)

    try:
        cond = float(np.linalg.cond(J.astype(np.float64)))
        print(f"  cond(J) ≈ {cond:.3e} (computed in f64)", flush=True)
    except Exception:
        pass

    print(f"  mpmath solve at {MP_DPS} dps...", flush=True)
    try:
        dP_flat = solve_mpmath(J, (-F).reshape(-1))
    except Exception as ex:
        print(f"    solve FAILED: {ex}", flush=True)
        break
    dP = dP_flat.reshape(P.shape)

    alpha = np.float128(1.0)
    Finf_try = Finf
    accepted = False
    for back in range(12):
        P_try = np.clip(P + alpha * dP,
                          np.float128(EPS_OUTER),
                          np.float128(1 - EPS_OUTER))
        F_try = F_eval(P_try)
        Finf_try = float(np.abs(F_try).max())
        if Finf_try < Finf:
            P = P_try; F = F_try
            accepted = True
            break
        alpha *= np.float128(0.5)

    if accepted and Finf_try < best_Finf:
        best_Finf = Finf_try
        P_best = P.copy()

    ratio = Finf_try / Finf if Finf > 0 else float("nan")
    print(f"  Newton iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
          f"ratio={ratio:.4f}  α={float(alpha):.6f}  "
          f"accepted={accepted}  t={time.time()-t_iter:.1f}s",
          flush=True)

    if Finf_try < 1e-13:
        print(f"  Newton CONVERGED iter {it}", flush=True)
        break

    Finf = Finf_try
    if not accepted:
        print(f"  Newton step rejected. STOP.", flush=True)
        break


P64_final = P_best.astype(np.float64)
Phi64_final = rh._phi_map(P64_final, u, TAUS, GAMMAS, WS)
Finf_final = float(np.abs(P64_final - Phi64_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL  best Finf (f128) = {best_Finf:.3e}  "
      f"on-disk f64 verify = {Finf_final:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)

if best_Finf < seed["Finf"]:
    fn = "/home/user/REZN/python/PR_seed_b9_v6_mpmath.pkl"
    with open(fn, "wb") as f:
        pickle.dump({"P": P64_final, "P_f128": P_best,
                      "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": best_Finf, "1-R²": one_r2,
                      "improved_from": seed["Finf"],
                      "kernel": "linear (rezn_het) + mpmath f128 LU",
                      "label": "opposed γ⟂τ"}, f)
    print(f"  Improved over seed; saved to {fn}", flush=True)
