"""Batch 9 minimal — one config, lean iterations, time per phase.

Goal: prove the hybrid float128 plumbing works in <2 min on the FR
control (γ=3 τ=3). Once timing is honest, scale up.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


G = 11
UMAX = 2.0
TAUS = np.array([3.0, 3.0, 3.0])
GAMMAS = np.array([3.0, 3.0, 3.0])
WS = np.array([1.0, 1.0, 1.0])
EPS_OUTER = 1e-9


def hybrid_newton_step(P128, u, taus, gammas, Ws, lgmres_tol=1e-13,
                        lgmres_maxiter=60):
    """One hybrid Newton step. Returns (P128_new, Finf_before, t_phi, t_J, t_lgmres, t_step)."""
    eps128 = np.float128(EPS_OUTER)
    one_m_eps128 = np.float128(1.0 - EPS_OUTER)

    t0 = time.time()
    P64 = P128.astype(np.float64)
    Phi64 = rp._phi_map_pchip(P64, u, taus, gammas, Ws)
    F128 = P128 - Phi64.astype(np.float128)
    Finf = float(np.abs(F128).max())
    t_phi = time.time() - t0

    t0 = time.time()
    precomp = pj.precompute(P64, u, taus, gammas, Ws)
    t_J = time.time() - t0

    N = G ** 3
    def matvec(v, _pre=precomp):
        return pj.J_dot_v(v.reshape(P64.shape), _pre).reshape(-1)
    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    t0 = time.time()
    F64 = F128.astype(np.float64)
    dP_flat, _ = lgmres(op, -F64.reshape(-1),
                          rtol=lgmres_tol, atol=0.0,
                          maxiter=lgmres_maxiter)
    t_lgmres = time.time() - t0

    t0 = time.time()
    dP128 = dP_flat.reshape(P128.shape).astype(np.float128)
    alpha = np.float128(1.0)
    accepted = False
    for _ in range(6):
        P_try = np.clip(P128 + alpha * dP128, eps128, one_m_eps128)
        Phi_try = rp._phi_map_pchip(P_try.astype(np.float64),
                                      u, taus, gammas, Ws)
        F_try = P_try - Phi_try.astype(np.float128)
        Finf_try = float(np.abs(F_try).max())
        if Finf_try < Finf:
            P128 = P_try
            accepted = True
            break
        alpha *= np.float128(0.5)
    if not accepted:
        P128 = np.clip(Phi64.astype(np.float128), eps128, one_m_eps128)
    t_step = time.time() - t0

    return P128, Finf, t_phi, t_J, t_lgmres, t_step


print(f"=== Minimal hybrid f128 timing test, γ=3 τ=3 FR ctrl, G={G} ===",
      flush=True)
u = np.linspace(-UMAX, UMAX, G)
P0 = rh._nolearning_price(u, TAUS, GAMMAS, WS)

# JIT warmup: one Picard iter then one Newton-ish call
t0 = time.time()
res_warm = rp.solve_picard_pchip(G, TAUS, GAMMAS, umax=UMAX,
                                   maxiters=1, abstol=1e-3, alpha=0.3,
                                   P_init=P0)
print(f"  JIT warm Picard 1 iter: {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
_p = pj.precompute(P0, u, TAUS, GAMMAS, WS)
_v = pj.J_dot_v(P0, _p)
print(f"  JIT warm precompute+J·v: {time.time()-t0:.2f}s", flush=True)

# Picard burn-in (50 iters, abstol 1e-10 — realistic)
t0 = time.time()
res_p = rp.solve_picard_pchip(G, TAUS, GAMMAS, umax=UMAX,
                                maxiters=50, abstol=1e-10, alpha=0.3,
                                P_init=P0)
P_warm = res_p["P_star"]
Finf_picard = float(np.abs(res_p["residual"]).max())
print(f"  Picard 50 iters: {time.time()-t0:.2f}s, "
      f"Finf={Finf_picard:.3e}", flush=True)

# float64 Newton (5 iters)
t0 = time.time()
res64 = pj.solve_newton(G, TAUS, GAMMAS, umax=UMAX,
                          P_init=P_warm, maxiters=5, abstol=1e-15,
                          lgmres_tol=1e-13, lgmres_maxiter=60)
print(f"  Newton-f64 5 iters: {time.time()-t0:.2f}s, "
      f"best_Finf={res64['best_Finf']:.3e}", flush=True)

# Hybrid float128 — 5 iters
P128 = res64["P_star"].astype(np.float128)
print(f"  starting hybrid f128 from Finf={res64['best_Finf']:.3e}",
      flush=True)
for it in range(5):
    P128, Finf, t_phi, t_J, t_lgmres, t_step = hybrid_newton_step(
        P128, u, TAUS, GAMMAS, WS)
    print(f"    hybrid iter {it}: Finf={Finf:.3e}  "
          f"t_phi={t_phi*1000:.0f}ms  t_J={t_J*1000:.0f}ms  "
          f"t_lgmres={t_lgmres*1000:.0f}ms  t_step={t_step*1000:.0f}ms",
          flush=True)

print(f"\nFINAL: f64_best={res64['best_Finf']:.3e}  hybr_final={Finf:.3e}",
      flush=True)
