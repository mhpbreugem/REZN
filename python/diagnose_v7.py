"""Diagnose WHY J=(I-DΦ) is ill-conditioned at the v7 best iterate.

Three questions:
  Q1. Are cells stuck at the [1e-9, 1-1e-9] clip boundary?
      Cells at the clip have zero rows in J (any positive perturbation
      gets clipped back) — that creates exact rank deficiency.
  Q2. Is the residual F concentrated in a few cells, particularly at
      the clipped cells?
  Q3. Singular-value spectrum of J — how many σ_i are essentially zero?
      How does ‖F‖ project onto the column space vs null space of J?

Inputs: PR_seed_b9_v7_LM.pkl (best iterate, Finf=4.79e-4, 1-R²=21%)
"""
import time
import pickle
import numpy as np
import rezn_het as rh


SEED = "/home/user/REZN/python/PR_seed_b9_v7_LM.pkl"
EPS_OUTER = 1e-9
H_FD = np.float128("1e-12")


with open(SEED, "rb") as f:
    s = pickle.load(f)
P = s["P_f128"].astype(np.float128).copy()
TAUS = s["taus"]; GAMMAS = s["gammas"]; G = s["G"]; UMAX = s["umax"]
WS = np.array([1.0, 1.0, 1.0])
u = np.linspace(-UMAX, UMAX, G)
N = G ** 3

print(f"=== Diagnose v7 best iterate ===", flush=True)
print(f"  config: γ={GAMMAS}  τ={TAUS}  G={G}  N={N}", flush=True)
print(f"  Finf (saved) = {s['Finf']:.3e}  1-R² = {s['1-R²']:.3e}",
      flush=True)


# Q1. Cell distribution
P_flat = P.reshape(-1).astype(np.float64)
near_lo = (P_flat <= 10.0 * EPS_OUTER).sum()
at_lo   = (P_flat <= 2.0 * EPS_OUTER).sum()
near_hi = (P_flat >= 1.0 - 10.0 * EPS_OUTER).sum()
at_hi   = (P_flat >= 1.0 - 2.0 * EPS_OUTER).sum()
print(f"\nQ1. Cell distribution (clip = [{EPS_OUTER}, {1-EPS_OUTER}]):",
      flush=True)
print(f"    cells AT lower clip (≤{2*EPS_OUTER:.0e}):     {at_lo} / {N}",
      flush=True)
print(f"    cells near lower (≤{10*EPS_OUTER:.0e}):      {near_lo} / {N}",
      flush=True)
print(f"    cells AT upper clip (≥{1-2*EPS_OUTER:.7f}): {at_hi} / {N}",
      flush=True)
print(f"    cells near upper (≥{1-10*EPS_OUTER:.7f}):  {near_hi} / {N}",
      flush=True)
print(f"    P quantiles  min/1%/50%/99%/max  =  "
      f"{P_flat.min():.3e} / {np.quantile(P_flat, 0.01):.3e} / "
      f"{np.quantile(P_flat, 0.50):.4f} / "
      f"{np.quantile(P_flat, 0.99):.4f} / {P_flat.max():.4f}",
      flush=True)


# Compute F
def Phi128(P128):
    return rh._phi_map(P128.astype(np.float64),
                         u, TAUS, GAMMAS, WS).astype(np.float128)

F = (P - Phi128(P)).reshape(-1)
F_abs = np.abs(F).astype(np.float64)
Finf = float(F_abs.max())
print(f"\nQ2. Residual distribution (Finf={Finf:.3e}):", flush=True)
print(f"    |F| quantiles  50%/90%/99%/max  =  "
      f"{np.quantile(F_abs, 0.50):.3e} / "
      f"{np.quantile(F_abs, 0.90):.3e} / "
      f"{np.quantile(F_abs, 0.99):.3e} / {F_abs.max():.3e}",
      flush=True)
top10_idx = np.argsort(F_abs)[-10:][::-1]
print(f"    Top-10 |F| cells (flat-idx, |F|, P-value, near-clip?):",
      flush=True)
for idx in top10_idx:
    pv = float(P_flat[idx])
    clip_status = ("LO!" if pv < 10*EPS_OUTER
                   else "HI!" if pv > 1 - 10*EPS_OUTER
                   else "ok")
    i, j, l = np.unravel_index(idx, (G, G, G))
    print(f"      ({i},{j},{l})   |F|={F_abs[idx]:.3e}  P={pv:.4f}  {clip_status}",
          flush=True)


# Q3. SVD of J via FD
print(f"\nQ3. Building FD Jacobian (cond + SVD)...", flush=True)
t0 = time.time()
J = np.empty((N, N), dtype=np.float128)
for k in range(N):
    Pp = P_flat.copy().astype(np.float128); Pp[k] += H_FD
    Pm = P_flat.copy().astype(np.float128); Pm[k] -= H_FD
    Pp = np.clip(Pp, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
    Pm = np.clip(Pm, np.float128(EPS_OUTER), np.float128(1 - EPS_OUTER))
    Fp = (Pp.reshape(P.shape) - Phi128(Pp.reshape(P.shape))).reshape(-1)
    Fm = (Pm.reshape(P.shape) - Phi128(Pm.reshape(P.shape))).reshape(-1)
    J[:, k] = (Fp - Fm) / (np.float128(2.0) * H_FD)
    if (k + 1) % 200 == 0:
        print(f"    FD col {k+1}/{N}  elapsed={time.time()-t0:.1f}s",
              flush=True)
print(f"    FD-J built in {time.time()-t0:.1f}s", flush=True)

J64 = J.astype(np.float64)
print(f"    computing SVD of J in float64...", flush=True)
t0 = time.time()
U, sigma, Vt = np.linalg.svd(J64, full_matrices=False)
print(f"    SVD done in {time.time()-t0:.1f}s", flush=True)

print(f"\n  Singular value spectrum (sigma_max = {sigma[0]:.3e}):", flush=True)
for q in (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0):
    rank = int(round(q * (N - 1)))
    print(f"    σ[{rank:4d}] = {sigma[rank]:.3e}  "
          f"(σ_max/σ = {sigma[0]/max(sigma[rank], 1e-300):.3e})",
          flush=True)
n_below_1em10 = (sigma < 1e-10 * sigma[0]).sum()
n_below_1em8  = (sigma < 1e-8 * sigma[0]).sum()
n_below_1em6  = (sigma < 1e-6 * sigma[0]).sum()
print(f"    # σ_i / σ_max < 1e-6: {n_below_1em6} / {N}", flush=True)
print(f"    # σ_i / σ_max < 1e-8: {n_below_1em8} / {N}", flush=True)
print(f"    # σ_i / σ_max < 1e-10: {n_below_1em10} / {N}", flush=True)
print(f"    cond(J) = σ_max / σ_min = "
      f"{sigma[0]/max(sigma[-1], 1e-300):.3e}", flush=True)


# Project F onto column space (image) vs null space of J
F64 = F.astype(np.float64)
F_in_range = U.T @ F64                            # F components in range(J) basis
F_in_null  = F64 - U @ (U.T @ F64)                # orthogonal to range(J)
norm_in    = float(np.linalg.norm(F_in_range))
norm_out   = float(np.linalg.norm(F_in_null))
print(f"\n  ‖F‖_in_range(J) = {norm_in:.3e}", flush=True)
print(f"  ‖F‖_orth_to_range(J) = {norm_out:.3e}  ← if non-zero, "
      f"residual has component J cannot reduce", flush=True)
total = float(np.linalg.norm(F64))
print(f"  ‖F‖_total = {total:.3e}", flush=True)
print(f"  fraction of ‖F‖ outside range(J): {norm_out/total:.3e}",
      flush=True)


# Per-singular-value contribution: how is F decomposed?
contributions = (U.T @ F64).astype(np.float64)
contrib_abs = np.abs(contributions)
print(f"\n  F decomposition by singular vector:", flush=True)
print(f"    largest |c_i|·σ_i (most reducible):",
      flush=True)
top_reducible = np.argsort(np.abs(contributions) * sigma)[-5:][::-1]
for idx in top_reducible:
    print(f"      σ[{idx}]={sigma[idx]:.3e}  "
          f"|c|={contrib_abs[idx]:.3e}  "
          f"|c|·σ={contrib_abs[idx]*sigma[idx]:.3e}",
          flush=True)
print(f"    largest |c_i|/σ_i (hardest to reduce, drives Newton step magnitude):",
      flush=True)
worst = np.argsort(np.abs(contributions) / np.maximum(sigma, 1e-30))[-5:][::-1]
for idx in worst:
    print(f"      σ[{idx}]={sigma[idx]:.3e}  "
          f"|c|={contrib_abs[idx]:.3e}  "
          f"|c|/σ={contrib_abs[idx]/max(sigma[idx], 1e-30):.3e}",
          flush=True)
