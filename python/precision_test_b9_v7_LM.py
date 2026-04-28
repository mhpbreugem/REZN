"""Batch 9 v7 — Levenberg-Marquardt + iterative refinement, transparent.

The diagnosis from v5/v6: at the opposed-γ⟂τ basin the Jacobian condition
number jumps from ~3e9 to ~6e11 in one Newton step, and Armijo rejects
all step sizes. This is a near-singular Jacobian where the pure Newton
direction dP = J⁻¹ · F is dominated by the eigendirection with the
smallest singular value, which need not be a descent direction for ‖F‖.
Higher precision in the linear solve doesn't fix this — the direction
is wrong, not the magnitude.

Levenberg-Marquardt fixes it: solve

    (JᵀJ + λI) dP = -Jᵀ F

with adaptive λ. Small λ → Gauss-Newton (fast near regular fp).
Large λ → steepest descent (always a descent direction).

Inside the LM solve we use iterative refinement with TRANSPARENT
progress: each refinement iter prints ‖r‖, the residual norm in
float128. Solver completes when ‖r‖ < tol or max iters hit.

Pipeline:
  1. Load v5's best (Finf 2.75e-3, 1-R² 21%)
  2. Each LM iter:
     - F, J via FD in f128
     - solve (JᵀJ + λI) · dP = -JᵀF   with iter refinement (f64 LU + f128 residual)
     - Armijo backtrack;   if accepted: λ /= 3
                          else:        λ *= 5; retry up to 4 times
  3. 15 iters max.
"""
import time
import pickle
import numpy as np
import rezn_het as rh


SEED_FILE = "/home/user/REZN/python/PR_seed_b9_v5_BEST.pkl"
EPS_OUTER = 1e-9
H_FD       = np.float128("1e-12")
LM_ITERS   = 15
HEARTBEAT  = 10.0
LAMBDA0    = np.float128("1e-3")
REFINE_TOL = 1e-25
REFINE_MAX = 12


with open(SEED_FILE, "rb") as f:
    seed = pickle.load(f)
P0 = (seed["P_f128"] if "P_f128" in seed else seed["P"]).astype(np.float128).copy()
TAUS = seed["taus"]
GAMMAS = seed["gammas"]
G = seed["G"]
UMAX = seed["umax"]
WS = np.array([1.0, 1.0, 1.0])
u = np.linspace(-UMAX, UMAX, G)


print(f"=== v7 LM + iter-refine (TRANSPARENT) ===", flush=True)
print(f"  config: γ={GAMMAS}  τ={TAUS}  G={G}", flush=True)
print(f"  starting from saved iterate: Finf={seed['Finf']:.3e}  "
      f"1-R²={seed['1-R²']:.3e}", flush=True)


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


def lm_solve(J128, F128, lam):
    """Solve (JᵀJ + λI) · dP = -JᵀF using iterative refinement.
    Builds the normal equations in float128, factorises in float64,
    refines residual in float128. Reports ‖r‖ each refinement iter.
    """
    N = J128.shape[0]
    Ft = F128.reshape(-1)

    # Build A = JᵀJ + λI in float128
    print(f"      [LM] building JᵀJ (N={N})...", flush=True)
    t0 = time.time()
    A = J128.T @ J128
    A_diag = np.arange(N)
    A[A_diag, A_diag] += lam
    rhs = -(J128.T @ Ft)
    print(f"      [LM] normal eqns built in {time.time()-t0:.1f}s",
          flush=True)

    # Factorise A in float64
    A64 = A.astype(np.float64)
    rhs64 = rhs.astype(np.float64)
    print(f"      [LM] solving f64 LU...", flush=True)
    t0 = time.time()
    try:
        dP = np.linalg.solve(A64, rhs64).astype(np.float128)
    except Exception as ex:
        print(f"      [LM] f64 LU FAILED: {ex}", flush=True)
        return None
    r0 = float(np.abs(rhs - A @ dP).max())
    print(f"      [LM] f64 LU done in {time.time()-t0:.1f}s, "
          f"‖r‖∞={r0:.3e}", flush=True)

    # Iterative refinement
    for r_it in range(REFINE_MAX):
        r128 = rhs - (A @ dP)
        r_norm = float(np.abs(r128).max())
        if r_norm < REFINE_TOL:
            print(f"      [LM-refine {r_it}] CONVERGED ‖r‖∞={r_norm:.3e}",
                  flush=True)
            break
        delta = np.linalg.solve(A64, r128.astype(np.float64))
        dP = dP + delta.astype(np.float128)
        new_r = float(np.abs(rhs - A @ dP).max())
        rate = new_r / r_norm if r_norm > 0 else float("nan")
        print(f"      [LM-refine {r_it}] ‖r‖∞ {r_norm:.3e} → {new_r:.3e}  "
              f"rate={rate:.3e}",
              flush=True)
        if new_r >= r_norm:  # not converging
            print(f"      [LM-refine {r_it}] STAGNATED — stop",
                  flush=True)
            break

    return dP


P = np.clip(P0, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
F = F_eval(P)
Finf = float(np.abs(F).max())
print(f"  initial Finf={Finf:.3e}", flush=True)

best_Finf = Finf
P_best = P.copy()
lam = LAMBDA0

for it in range(LM_ITERS):
    t_iter = time.time()
    print(f"\n=== LM iter {it} (λ={float(lam):.3e}) ===", flush=True)
    print(f"  building Jacobian...", flush=True)
    J = build_FD_jacobian(P)

    try:
        cond = float(np.linalg.cond(J.astype(np.float64)))
        print(f"  cond(J) ≈ {cond:.3e}", flush=True)
    except Exception:
        pass

    accepted = False
    Finf_try = Finf
    # Adaptive λ: try up to 4 times, increasing λ by 5x each time
    for lam_attempt in range(4):
        print(f"  --- λ attempt {lam_attempt}: λ={float(lam):.3e} ---",
              flush=True)
        dP_flat = lm_solve(J, F, lam)
        if dP_flat is None:
            lam *= np.float128(5.0)
            continue
        dP = dP_flat.reshape(P.shape)

        # Armijo backtrack on ‖F‖∞
        alpha = np.float128(1.0)
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

        if accepted:
            lam = max(np.float128("1e-12"), lam / np.float128(3.0))
            print(f"  λ accepted, decreased to {float(lam):.3e}",
                  flush=True)
            break
        else:
            lam *= np.float128(5.0)
            print(f"  λ rejected, increased to {float(lam):.3e}",
                  flush=True)

    if accepted and Finf_try < best_Finf:
        best_Finf = Finf_try
        P_best = P.copy()

    ratio = Finf_try / Finf if Finf > 0 else float("nan")
    print(f"  LM iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
          f"ratio={ratio:.4f}  α={float(alpha):.6f}  "
          f"accepted={accepted}  t={time.time()-t_iter:.1f}s",
          flush=True)

    if Finf_try < 1e-13:
        print(f"  LM CONVERGED iter {it}", flush=True)
        break

    Finf = Finf_try
    if not accepted:
        print(f"  LM step rejected at all λ attempts. STOP.",
              flush=True)
        break


P64_final = P_best.astype(np.float64)
Phi64_final = rh._phi_map(P64_final, u, TAUS, GAMMAS, WS)
Finf_final = float(np.abs(P64_final - Phi64_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL  best Finf (f128) = {best_Finf:.3e}  "
      f"on-disk f64 verify = {Finf_final:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)

if best_Finf < seed["Finf"]:
    fn = "/home/user/REZN/python/PR_seed_b9_v7_LM.pkl"
    with open(fn, "wb") as f:
        pickle.dump({"P": P64_final, "P_f128": P_best,
                      "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": best_Finf, "1-R²": one_r2,
                      "improved_from": seed["Finf"],
                      "kernel": "linear (rezn_het) + LM + iter-refine",
                      "label": "opposed γ⟂τ"}, f)
    print(f"  Improved over seed; saved to {fn}", flush=True)
