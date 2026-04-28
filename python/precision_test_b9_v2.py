"""Batch 9 v2 — γ=3 τ=4 with damping 0.5, Picard 2000, hybrid Newton.

Live updates every 10 s in every phase:
  - Picard: time-check inside the loop emits Finf when ≥10 s passed
  - Newton-f64: lgmres callback emits Krylov residual ≥10 s tick
  - Hybrid f128: same; per-iter Finf print is also natural ~23 s
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


G = 11
UMAX = 2.0
TAUS = np.array([4.0, 4.0, 4.0])
GAMMAS = np.array([3.0, 3.0, 3.0])
WS = np.array([1.0, 1.0, 1.0])
EPS_OUTER = 1e-9
LGMRES_TOL = 1e-13
LGMRES_MAXITER = 100
HEARTBEAT = 10.0


u = np.linspace(-UMAX, UMAX, G)
P0 = rh._nolearning_price(u, TAUS, GAMMAS, WS)


# JIT warm
print("=== JIT warm ===", flush=True)
t0 = time.time()
_phi = rp._phi_map_pchip(P0, u, TAUS, GAMMAS, WS)
_p = pj.precompute(P0, u, TAUS, GAMMAS, WS)
_v = pj.J_dot_v(P0, _p)
print(f"  JIT warm: {time.time()-t0:.1f}s", flush=True)


def picard(P_init, alpha, maxiter, abstol):
    """Damped Picard with 10s heartbeat. Returns final P and last residual."""
    print(f"=== Picard damping α={alpha}, {maxiter} iters, "
          f"heartbeat {HEARTBEAT:.0f}s ===", flush=True)
    P = P_init.copy()
    t_start = time.time()
    t_last = 0.0
    for it in range(maxiter):
        Phi = rp._phi_map_pchip(P, u, TAUS, GAMMAS, WS)
        F = P - Phi
        Finf_resid = float(np.abs(F).max())
        P = np.clip(alpha * Phi + (1 - alpha) * P,
                    EPS_OUTER, 1 - EPS_OUTER)
        elapsed = time.time() - t_start
        if (elapsed - t_last) >= HEARTBEAT or it == maxiter - 1 \
                or Finf_resid < abstol:
            print(f"  Picard iter {it+1}/{maxiter}: "
                  f"resid={Finf_resid:.3e}  elapsed={elapsed:.1f}s",
                  flush=True)
            t_last = elapsed
        if Finf_resid < abstol:
            print(f"  Picard CONVERGED at iter {it+1}", flush=True)
            break
    Phi_final = rp._phi_map_pchip(P, u, TAUS, GAMMAS, WS)
    Finf_final = float(np.abs(P - Phi_final).max())
    print(f"  Picard done: final resid={Finf_final:.3e}  "
          f"total={time.time()-t_start:.1f}s",
          flush=True)
    return P, Finf_final


class HeartbeatCallback:
    """lgmres callback that prints residual norm every HEARTBEAT seconds."""
    def __init__(self, label):
        self.label = label
        self.t0 = time.time()
        self.t_last = 0.0
        self.n = 0

    def __call__(self, xk):
        self.n += 1
        elapsed = time.time() - self.t0
        if (elapsed - self.t_last) >= HEARTBEAT:
            try:
                rn = float(np.linalg.norm(xk))
            except Exception:
                rn = float("nan")
            print(f"    [{self.label}] krylov iter {self.n}  "
                  f"‖xk‖={rn:.3e}  elapsed={elapsed:.1f}s", flush=True)
            self.t_last = elapsed


def newton_step_f64(P, label):
    """One float64 Newton step with lgmres heartbeat."""
    Phi = rp._phi_map_pchip(P, u, TAUS, GAMMAS, WS)
    F = P - Phi
    Finf = float(np.abs(F).max())
    precomp = pj.precompute(P, u, TAUS, GAMMAS, WS)
    N = G ** 3

    def matvec(v, _pre=precomp):
        return pj.J_dot_v(v.reshape(P.shape), _pre).reshape(-1)

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    cb = HeartbeatCallback(label)
    dP_flat, _ = lgmres(op, -F.reshape(-1),
                          rtol=LGMRES_TOL, atol=0.0,
                          maxiter=LGMRES_MAXITER, callback=cb)
    dP = dP_flat.reshape(P.shape)

    alpha = 1.0
    Finf_try = Finf
    for _ in range(8):
        P_try = np.clip(P + alpha * dP, EPS_OUTER, 1 - EPS_OUTER)
        Phi_try = rp._phi_map_pchip(P_try, u, TAUS, GAMMAS, WS)
        Finf_try = float(np.abs(P_try - Phi_try).max())
        if Finf_try < Finf:
            P = P_try
            break
        alpha *= 0.5
    else:
        P = np.clip(Phi, EPS_OUTER, 1 - EPS_OUTER)
    return P, Finf, Finf_try, alpha


def newton_step_hybrid(P128, label):
    """One hybrid f64-Φ / f128-state Newton step with lgmres heartbeat."""
    eps128 = np.float128(EPS_OUTER)
    one_m_eps128 = np.float128(1.0 - EPS_OUTER)
    P64 = P128.astype(np.float64)
    Phi64 = rp._phi_map_pchip(P64, u, TAUS, GAMMAS, WS)
    F128 = P128 - Phi64.astype(np.float128)
    Finf = float(np.abs(F128).max())
    precomp = pj.precompute(P64, u, TAUS, GAMMAS, WS)
    N = G ** 3

    def matvec(v, _pre=precomp, _shape=P64.shape):
        return pj.J_dot_v(v.reshape(_shape), _pre).reshape(-1)

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    F64 = F128.astype(np.float64)
    cb = HeartbeatCallback(label)
    dP_flat, _ = lgmres(op, -F64.reshape(-1),
                          rtol=LGMRES_TOL, atol=0.0,
                          maxiter=LGMRES_MAXITER, callback=cb)
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
        P128 = np.clip(Phi64.astype(np.float128), eps128, one_m_eps128)
    return P128, Finf, Finf_try, float(alpha)


# Picard burn-in: damping 0.5, 2000 iters
P_warm, Finf_picard = picard(P0, alpha=0.5, maxiter=2000, abstol=1e-13)


# Float64 Newton (8 iters), per-iter print
print(f"=== Newton-f64 (8 iters) ===", flush=True)
P64 = P_warm
for it in range(8):
    t_iter = time.time()
    P64, Finf_in, Finf_out, alpha = newton_step_f64(P64, f"f64 iter {it}")
    print(f"  newton-f64 iter {it}: {Finf_in:.3e} → {Finf_out:.3e}  "
          f"ratio={Finf_out/Finf_in:.3f}  α={alpha:.4f}  "
          f"t={time.time()-t_iter:.1f}s", flush=True)
    if Finf_out < 1e-13:
        print(f"  newton-f64 CONVERGED iter {it}", flush=True)
        break


# Hybrid f128 (12 iters)
print(f"=== Hybrid f128 (12 iters) ===", flush=True)
P128 = P64.astype(np.float128)
for it in range(12):
    t_iter = time.time()
    P128, Finf_in, Finf_out, alpha = newton_step_hybrid(
        P128, f"hybr iter {it}")
    print(f"  hybrid iter {it}: {Finf_in:.3e} → {Finf_out:.3e}  "
          f"ratio={Finf_out/Finf_in:.3f}  α={alpha:.4f}  "
          f"t={time.time()-t_iter:.1f}s", flush=True)
    if Finf_out < 1e-13:
        print(f"  hybrid CONVERGED iter {it}", flush=True)
        break

P64_final = P128.astype(np.float64)
Phi64_final = rp._phi_map_pchip(P64_final, u, TAUS, GAMMAS, WS)
Finf_final_64 = float(np.abs(P64_final - Phi64_final).max())
F128_final = P128 - Phi64_final.astype(np.float128)
Finf_final_128 = float(np.abs(F128_final).max())
one_r2 = rh.one_minus_R2(P64_final, u, TAUS)
print(f"\nFINAL  Picard: {Finf_picard:.3e}  "
      f"hybrid f64-norm: {Finf_final_64:.3e}  "
      f"hybrid f128-norm: {Finf_final_128:.3e}", flush=True)
print(f"       1-R² = {one_r2:.3e}", flush=True)
