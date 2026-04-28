"""Batch 9 v5 — FD Jacobian in float128 with dense linear solve.

Different angle from v3 (analytical-J LGMRES) and v4 (scipy.newton_krylov):
build the *full* Jacobian by central finite differences with all
arithmetic in float128, then solve J·dP = −F densely in float128. This
avoids both (a) the analytical-J implementation, which might have any
bug, and (b) Krylov methods, which can stall on ill-conditioned J. With
float128 we can also use a much smaller FD step (h ≈ 1e-15 vs ~1e-8 for
f64) so the Jacobian itself is sharper.

Pipeline:
  1. Load best Picard iterate from PR_seed_b9_v4_opposed_LINEAR.pkl
     (Finf = 3.67e-3, 1-R² = 21.0%, opposed γ⟂τ, linear kernel)
  2. For up to 12 Newton iterations:
     - F = P − Φ(P)         (cast P → f64 for numba Φ, F kept in f128)
     - J via central FD: 2·G³ Φ evaluations, h = 1e-12
     - dP = numpy.linalg.solve(J, −F)   (float128 if numpy preserves it)
     - Armijo backtrack on ‖F‖∞ in f128
  3. Save tensor on convergence below 1e-10.

Cost: G=11, G³=1331. 2662 Φ calls per Newton step ≈ 30 s. Dense solve
N=1331 in numpy float128 ≈ 10-30 s. So ~60 s per Newton iter, ~12 min
for 12 iters.
"""
import time
import pickle
import numpy as np
import rezn_het as rh


SEED_FILE = "/home/user/REZN/python/PR_seed_b9_v4_opposed_LINEAR.pkl"
EPS_OUTER = 1e-9
H_FD       = np.float128("1e-12")     # FD step
NEWTON_ITERS = 12
HEARTBEAT  = 10.0


with open(SEED_FILE, "rb") as f:
    seed = pickle.load(f)
P0 = seed["P"].astype(np.float128).copy()
TAUS = seed["taus"]
GAMMAS = seed["gammas"]
G = seed["G"]
UMAX = seed["umax"]
WS = np.array([1.0, 1.0, 1.0])
u = np.linspace(-UMAX, UMAX, G)

print(f"=== v5 FD-J float128 Newton ===", flush=True)
print(f"  config: γ={GAMMAS}  τ={TAUS}  G={G}", flush=True)
print(f"  starting from saved Picard best: Finf={seed['Finf']:.3e}  "
      f"1-R²={seed['1-R²']:.3e}", flush=True)


def Phi128(P128):
    """Φ called with float128 input/output. Internal cast to f64 for numba."""
    return rh._phi_map(P128.astype(np.float64),
                         u, TAUS, GAMMAS, WS).astype(np.float128)


def F_eval(P128):
    """F = P - Φ(P) in float128."""
    return P128 - Phi128(P128)


def build_FD_jacobian(P128, h=H_FD):
    """Central finite differences. Returns N×N float128 matrix."""
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


P = np.clip(P0, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
F = F_eval(P)
Finf = float(np.abs(F).max())
print(f"  initial Finf={Finf:.3e}", flush=True)

best_Finf = Finf
P_best = P.copy()

for it in range(NEWTON_ITERS):
    t_iter = time.time()
    print(f"\n=== Newton-FD-f128 iter {it} ===", flush=True)
    print(f"  building Jacobian (FD, h={float(H_FD):.0e}, N={G**3})...",
          flush=True)
    J = build_FD_jacobian(P)

    # Check J condition (in float128 if possible, else float64)
    try:
        cond = float(np.linalg.cond(J.astype(np.float64)))
        print(f"  cond(J) ≈ {cond:.3e}  (computed in f64)", flush=True)
    except Exception:
        pass

    print(f"  solving J·dP = −F (float128)...", flush=True)
    t0 = time.time()
    try:
        dP_flat = np.linalg.solve(J, (-F).reshape(-1))
        dP_dtype = dP_flat.dtype
        print(f"    solve done in {time.time()-t0:.1f}s, "
              f"dP dtype={dP_dtype}", flush=True)
    except Exception as ex:
        print(f"    solve FAILED: {ex}", flush=True)
        break
    dP = dP_flat.reshape(P.shape).astype(np.float128)

    # Armijo backtrack
    alpha = np.float128(1.0)
    Finf_try = Finf
    accepted = False
    for back in range(10):
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
          f"ratio={ratio:.4f}  α={float(alpha):.4f}  "
          f"accepted={accepted}  t={time.time()-t_iter:.1f}s",
          flush=True)

    if Finf_try < 1e-13:
        print(f"  Newton CONVERGED iter {it}", flush=True)
        break

    Finf = Finf_try
    if not accepted:
        print(f"  Newton step rejected at all backtracks. STOP.",
              flush=True)
        break


P64_final = P_best.astype(np.float64)
Phi64_final = rh._phi_map(P64_final, u, TAUS, GAMMAS, WS)
Finf_final_64 = float(np.abs(P64_final - Phi64_final).max())
F128_final = P_best - Phi64_final.astype(np.float128)
Finf_final_128 = float(np.abs(F128_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL  best Finf (f128) = {best_Finf:.3e}  "
      f"on-disk f64 verify = {Finf_final_64:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)

if best_Finf < 1e-10 and one_r2 > 0.01:
    fn = "/home/user/REZN/python/PR_seed_b9_v5_FD128_LINEAR.pkl"
    with open(fn, "wb") as f:
        pickle.dump({"P": P64_final, "P_f128": P_best,
                      "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": best_Finf, "1-R²": one_r2,
                      "kernel": "linear (rezn_het._phi_map) + FD-J f128 Newton",
                      "label": "opposed γ=(0.3,3,30) τ=(30,3,0.3)"}, f)
    print(f"  PR! saved to {fn}", flush=True)
elif best_Finf < seed["Finf"]:
    fn = "/home/user/REZN/python/PR_seed_b9_v5_BEST.pkl"
    with open(fn, "wb") as f:
        pickle.dump({"P": P64_final, "P_f128": P_best,
                      "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": best_Finf, "1-R²": one_r2,
                      "improved_from": seed["Finf"],
                      "kernel": "linear + FD-J f128 Newton",
                      "label": "opposed γ⟂τ"}, f)
    print(f"  Improved over seed; saved to {fn}", flush=True)
