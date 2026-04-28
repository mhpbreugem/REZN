"""Batch 9 v8 — Truncated-SVD pseudoinverse Newton.

The v7 diagnostic confirmed:
  • J has effective rank ~97 of 1331 (1234 singular values below σ_max·1e-6)
  • F is entirely in range(J) — fixed point IS reachable
  • Newton blows up because smallest σ_i amplify tiny F-components into
    huge dP entries that destabilise

Truncated-SVD fix: project the Newton step onto the top-K singular
directions only. dP_TSVD = V · diag(1/σ_i if σ_i > σ_max·rcond else 0) · U^T · (-F)

Adaptive rcond:
  - start at 1e-6 (≈ effective-rank subspace per the diagnostic)
  - if Armijo rejects: rcond *= 10 (more aggressive truncation)
  - if accepted: rcond /= 3 (more directions next iter)

Plus iterative refinement to push the linear-solve residual below f128.

Pipeline:
  1. Load v7 best iterate (Finf=4.79e-4, 1-R²=21%)
  2. Up to 15 Newton-TSVD iterations
  3. Save tensor on improvement.
"""
import time
import pickle
import numpy as np
import rezn_het as rh


SEED_FILE = "/home/user/REZN/python/PR_seed_b9_v7_LM.pkl"
EPS_OUTER = 1e-9
H_FD       = np.float128("1e-12")
NEWTON_ITERS = 15
HEARTBEAT  = 10.0
RCOND_INIT = 1e-6
RCOND_MIN  = 1e-12
RCOND_MAX  = 1e-2
REFINE_MAX = 6


with open(SEED_FILE, "rb") as f:
    seed = pickle.load(f)
P0 = (seed["P_f128"] if "P_f128" in seed else seed["P"]).astype(np.float128).copy()
TAUS = seed["taus"]
GAMMAS = seed["gammas"]
G = seed["G"]
UMAX = seed["umax"]
WS = np.array([1.0, 1.0, 1.0])
u = np.linspace(-UMAX, UMAX, G)


print(f"=== v8 Truncated-SVD Newton ===", flush=True)
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


def tsvd_solve(J128, b128, rcond):
    """Truncated-SVD pseudoinverse with iterative refinement.
    Returns (dP_f128, rank, ‖r_final‖)."""
    N = J128.shape[0]
    J64 = J128.astype(np.float64)
    b64 = b128.reshape(-1).astype(np.float64)

    print(f"      [TSVD] computing SVD...", flush=True)
    t0 = time.time()
    U, sigma, Vt = np.linalg.svd(J64, full_matrices=False)
    print(f"      [TSVD] SVD done in {time.time()-t0:.1f}s; "
          f"σ_max={sigma[0]:.3e}, σ_min={sigma[-1]:.3e}", flush=True)

    threshold = sigma[0] * rcond
    keep = sigma > threshold
    rank = int(keep.sum())
    inv_sigma = np.where(keep, 1.0 / sigma, 0.0)
    print(f"      [TSVD] rcond={rcond:.0e}, threshold={threshold:.3e}, "
          f"rank kept={rank}/{N}",
          flush=True)

    # Initial pseudoinverse solve
    Utb = U.T @ b64
    dP_64 = (Vt.T * inv_sigma) @ Utb
    dP_128 = dP_64.astype(np.float128)

    # Iterative refinement using full f128 residual
    b128_flat = b128.reshape(-1)
    for r_it in range(REFINE_MAX):
        r128 = b128_flat - (J128 @ dP_128)
        r_norm = float(np.abs(r128).max())
        if r_norm < 1e-22:
            print(f"      [TSVD-refine {r_it}] CONVERGED ‖r‖∞={r_norm:.3e}",
                  flush=True)
            break
        # Project residual via the same TSVD pseudoinverse
        Utr = U.T @ r128.astype(np.float64)
        delta = (Vt.T * inv_sigma) @ Utr
        dP_new = dP_128 + delta.astype(np.float128)
        new_r = float(np.abs(b128_flat - J128 @ dP_new).max())
        rate = new_r / r_norm if r_norm > 0 else float("nan")
        print(f"      [TSVD-refine {r_it}] ‖r‖∞ {r_norm:.3e} → {new_r:.3e}  "
              f"rate={rate:.3e}",
              flush=True)
        if new_r >= r_norm:
            print(f"      [TSVD-refine {r_it}] STAGNATED — stop",
                  flush=True)
            break
        dP_128 = dP_new
    final_r = float(np.abs(b128_flat - J128 @ dP_128).max())

    return dP_128, rank, final_r


P = np.clip(P0, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
F = F_eval(P)
Finf = float(np.abs(F).max())
print(f"  initial Finf={Finf:.3e}", flush=True)

best_Finf = Finf
P_best = P.copy()
rcond = RCOND_INIT

for it in range(NEWTON_ITERS):
    t_iter = time.time()
    print(f"\n=== TSVD iter {it} (rcond={rcond:.0e}) ===", flush=True)
    print(f"  building Jacobian...", flush=True)
    J = build_FD_jacobian(P)

    accepted = False
    Finf_try = Finf
    last_rank = -1

    for rcond_attempt in range(5):
        print(f"  --- rcond attempt {rcond_attempt}: rcond={rcond:.0e} ---",
              flush=True)
        dP_flat, rank, r_norm = tsvd_solve(J, -F, rcond)
        last_rank = rank
        dP = dP_flat.reshape(P.shape)

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
            rcond = max(RCOND_MIN, rcond / 3.0)
            print(f"  TSVD accepted, rcond decreased to {rcond:.0e}",
                  flush=True)
            break
        else:
            rcond = min(RCOND_MAX, rcond * 10.0)
            print(f"  TSVD rejected, rcond increased to {rcond:.0e} "
                  f"(more aggressive truncation)",
                  flush=True)

    if accepted and Finf_try < best_Finf:
        best_Finf = Finf_try
        P_best = P.copy()

    ratio = Finf_try / Finf if Finf > 0 else float("nan")
    print(f"  TSVD iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
          f"ratio={ratio:.4f}  α={float(alpha):.6f}  rank={last_rank}  "
          f"accepted={accepted}  t={time.time()-t_iter:.1f}s",
          flush=True)

    if Finf_try < 1e-13:
        print(f"  TSVD CONVERGED iter {it}", flush=True)
        break

    Finf = Finf_try
    if not accepted:
        print(f"  TSVD step rejected at all rcond attempts. STOP.",
              flush=True)
        break


P64_final = P_best.astype(np.float64)
Phi64_final = rh._phi_map(P64_final, u, TAUS, GAMMAS, WS)
Finf_final = float(np.abs(P64_final - Phi64_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL  best Finf = {best_Finf:.3e}  "
      f"on-disk f64 verify = {Finf_final:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)

if best_Finf < seed["Finf"]:
    fn = "/home/user/REZN/python/PR_seed_b9_v8_TSVD.pkl"
    with open(fn, "wb") as f:
        pickle.dump({"P": P64_final, "P_f128": P_best,
                      "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": best_Finf, "1-R²": one_r2,
                      "improved_from": seed["Finf"],
                      "kernel": "linear (rezn_het) + TSVD pseudoinverse",
                      "label": "opposed γ⟂τ"}, f)
    print(f"  Improved over seed; saved to {fn}", flush=True)
