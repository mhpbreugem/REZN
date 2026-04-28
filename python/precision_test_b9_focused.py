"""Batch 9 focused — γ=3 τ=4 (the cleanest stalled PR-suspect from
batch 7 at Finf=5.9e-6, 1-R²=2.9%) with live per-phase Finf reporting.

Live updates:
  - Picard:    every ~60 s of wall-clock
  - Newton:    after every iteration (~23 s/iter, well under 60 s)
  - Hybrid:    after every iteration (~23 s/iter)
  - Newton:    also writes a status file each iter for tail -f

Goal: see if hybrid f128 can break past the Newton-f64 stall, and
how it compares to the FR-control linear-rate result we just observed.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


G = 11
UMAX = 2.0
TAUS = np.array([4.0, 4.0, 4.0])         # τ = 4 (PR-suspect)
GAMMAS = np.array([3.0, 3.0, 3.0])
WS = np.array([1.0, 1.0, 1.0])
EPS_OUTER = 1e-9
LGMRES_TOL = 1e-13
LGMRES_MAXITER = 60


u = np.linspace(-UMAX, UMAX, G)
P0 = rh._nolearning_price(u, TAUS, GAMMAS, WS)


# JIT warm
print("=== JIT warm ===", flush=True)
t0 = time.time()
_phi = rp._phi_map_pchip(P0, u, TAUS, GAMMAS, WS)
_p = pj.precompute(P0, u, TAUS, GAMMAS, WS)
_v = pj.J_dot_v(P0, _p)
print(f"  JIT warm: {time.time()-t0:.1f}s", flush=True)


# Picard 300 with 60s heartbeat
print("=== Picard burn-in (300 iters, heartbeat ≥60 s) ===", flush=True)
P_warm = P0.copy()
t_start = time.time()
t_last = 0.0
for it in range(300):
    Phi = rp._phi_map_pchip(P_warm, u, TAUS, GAMMAS, WS)
    P_new = np.clip(0.3 * Phi + 0.7 * P_warm, EPS_OUTER, 1 - EPS_OUTER)
    Finf_p = float(np.abs(P_new - P_warm).max())
    P_warm = P_new
    elapsed = time.time() - t_start
    if elapsed - t_last > 60.0 or it == 299 or Finf_p < 1e-13:
        print(f"  Picard iter {it+1}/300: Finf={Finf_p:.3e}  "
              f"elapsed={elapsed:.1f}s", flush=True)
        t_last = elapsed
    if Finf_p < 1e-13:
        print(f"  Picard CONVERGED at iter {it+1}", flush=True)
        break

Phi_warm = rp._phi_map_pchip(P_warm, u, TAUS, GAMMAS, WS)
F_warm = P_warm - Phi_warm
print(f"  Picard final: ‖P − Φ(P)‖∞ = {float(np.abs(F_warm).max()):.3e}",
      flush=True)


# float64 Newton, 10 iters, verbose
print("=== Newton-f64 (10 iters, verbose) ===", flush=True)
t0 = time.time()
res64 = pj.solve_newton(
    G, TAUS, GAMMAS, umax=UMAX, P_init=P_warm,
    maxiters=10, abstol=1e-15,
    lgmres_tol=LGMRES_TOL, lgmres_maxiter=LGMRES_MAXITER,
    verbose=True,
    status_path="/home/user/REZN/python/b9_focused.status",
    status_every=1, status_prefix="newton-f64")
print(f"  Newton-f64 done in {time.time()-t0:.1f}s, "
      f"best_Finf={res64['best_Finf']:.3e}", flush=True)


# Hybrid f128 (12 iters), per-iter print
print("=== Hybrid f128 (12 iters, per-iter Finf) ===", flush=True)
P128 = res64["P_star"].astype(np.float128)
eps128 = np.float128(EPS_OUTER)
one_m_eps128 = np.float128(1.0 - EPS_OUTER)
N = G ** 3

for it in range(12):
    t_iter = time.time()
    P64 = P128.astype(np.float64)
    Phi64 = rp._phi_map_pchip(P64, u, TAUS, GAMMAS, WS)
    F128 = P128 - Phi64.astype(np.float128)
    Finf = float(np.abs(F128).max())

    if Finf < 1e-13:
        print(f"  hybrid CONVERGED iter {it}: Finf={Finf:.3e}", flush=True)
        break

    precomp = pj.precompute(P64, u, TAUS, GAMMAS, WS)

    def matvec(v, _pre=precomp):
        return pj.J_dot_v(v.reshape(P64.shape), _pre).reshape(-1)

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    F64 = F128.astype(np.float64)
    dP_flat, _ = lgmres(op, -F64.reshape(-1),
                          rtol=LGMRES_TOL, atol=0.0,
                          maxiter=LGMRES_MAXITER)
    dP128 = dP_flat.reshape(P128.shape).astype(np.float128)

    alpha = np.float128(1.0)
    Finf_try = Finf
    for _ in range(8):
        P_try = np.clip(P128 + alpha * dP128, eps128, one_m_eps128)
        Phi_try = rp._phi_map_pchip(P_try.astype(np.float64),
                                      u, TAUS, GAMMAS, WS)
        F_try = P_try - Phi_try.astype(np.float128)
        Finf_try = float(np.abs(F_try).max())
        if Finf_try < Finf:
            P128 = P_try
            break
        alpha *= np.float128(0.5)
    else:
        # Picard fallback
        P128 = np.clip(Phi64.astype(np.float128), eps128, one_m_eps128)

    ratio = Finf_try / Finf if Finf > 0 else float("nan")
    print(f"  hybrid iter {it}: Finf {Finf:.3e} → {Finf_try:.3e}  "
          f"ratio={ratio:.3f}  alpha={float(alpha):.4f}  "
          f"t_iter={time.time()-t_iter:.1f}s",
          flush=True)

P64_final = P128.astype(np.float64)
Phi64_final = rp._phi_map_pchip(P64_final, u, TAUS, GAMMAS, WS)
F_final = P128 - Phi64_final.astype(np.float128)
Finf_final = float(np.abs(F_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL: Newton-f64 best = {res64['best_Finf']:.3e}, "
      f"hybrid final = {Finf_final:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)
