"""Batch 9 v3 — adaptive Picard with α-shrink on oscillation and tiny
perturbation on stall, then Newton-f64, then hybrid f128. Target γ=3 τ=4.

Adaptive Picard rules (10 s heartbeat):
  - α starts at 0.20
  - sliding window of last 60 residuals
  - if window's recent max exceeds recent min by >5%  →  α *= 0.7
    (oscillation detected; floor α at 0.02)
  - if residual decreased monotonically for 200 iters  →  α *= 1.05
    (relax back toward initial; cap at 0.20)
  - stall detect: if min over last 800 iters ≥ 0.97 × min over prior 800
    AND we haven't perturbed in 800 iters  →  perturb P with σ = 1e-6 ·
    typical scale (mean of P over support)
  - print: every 10 s wall clock OR every α change OR every perturb
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


print("=== JIT warm ===", flush=True)
t0 = time.time()
_phi = rp._phi_map_pchip(P0, u, TAUS, GAMMAS, WS)
_p = pj.precompute(P0, u, TAUS, GAMMAS, WS)
_v = pj.J_dot_v(P0, _p)
print(f"  JIT warm: {time.time()-t0:.1f}s", flush=True)


def picard_adaptive(P_init, alpha0=0.20, alpha_min=0.02, alpha_max=0.20,
                     maxiter=20000, abstol=1e-13,
                     window=60, mono_relax=200,
                     stall_window=800, stall_ratio=0.97,
                     perturb_sigma=1e-6, perturb_seed=42):
    print(f"=== Adaptive Picard: α₀={alpha0}, max {maxiter} iters ===",
          flush=True)
    rng = np.random.default_rng(perturb_seed)
    P = P_init.copy()
    alpha = alpha0
    resid_hist = []
    decr_streak = 0
    last_perturb = -10**6
    n_perturb = 0
    n_alpha_changes = 0
    t_start = time.time()
    t_last_print = -1.0

    for it in range(maxiter):
        Phi = rp._phi_map_pchip(P, u, TAUS, GAMMAS, WS)
        F = P - Phi
        Finf = float(np.abs(F).max())
        resid_hist.append(Finf)

        if Finf < abstol:
            elapsed = time.time() - t_start
            print(f"  Picard CONVERGED iter {it+1}: resid={Finf:.3e}  "
                  f"elapsed={elapsed:.1f}s  α={alpha:.4f}",
                  flush=True)
            break

        # Adaptive α
        alpha_changed = False
        if it >= window:
            recent = resid_hist[-window:]
            r_min = min(recent)
            r_max = max(recent[-window // 2:])
            if r_min > 0 and r_max / r_min > 1.05:
                # oscillation → shrink α
                new_a = max(alpha_min, alpha * 0.7)
                if new_a < alpha:
                    alpha = new_a
                    decr_streak = 0
                    alpha_changed = True
                    n_alpha_changes += 1
        if it > 0 and Finf < resid_hist[-2]:
            decr_streak += 1
            if decr_streak >= mono_relax and alpha < alpha_max:
                new_a = min(alpha_max, alpha * 1.05)
                if new_a > alpha:
                    alpha = new_a
                    decr_streak = 0
                    alpha_changed = True
                    n_alpha_changes += 1
        else:
            decr_streak = 0

        # Stall detect → perturb
        perturbed = False
        if (it >= 2 * stall_window and
                (it - last_perturb) >= stall_window):
            recent_min = min(resid_hist[-stall_window:])
            prior_min = min(resid_hist[-2 * stall_window:-stall_window])
            if (prior_min > 0 and
                    recent_min >= stall_ratio * prior_min):
                scale = float(np.mean(P[(P > EPS_OUTER * 10) &
                                          (P < 1 - EPS_OUTER * 10)]))
                noise = perturb_sigma * scale * rng.standard_normal(P.shape)
                P = np.clip(P + noise, EPS_OUTER, 1 - EPS_OUTER)
                last_perturb = it
                n_perturb += 1
                perturbed = True

        # Damped Picard step (after possible perturb)
        if not perturbed:
            P = np.clip(alpha * Phi + (1 - alpha) * P,
                          EPS_OUTER, 1 - EPS_OUTER)

        # Heartbeat / event print
        elapsed = time.time() - t_start
        if (elapsed - t_last_print) >= HEARTBEAT or alpha_changed or perturbed \
                or it == maxiter - 1:
            tag = ""
            if alpha_changed: tag += f" α→{alpha:.4f}"
            if perturbed:     tag += f" PERTURB σ={perturb_sigma:.0e}"
            print(f"  Picard iter {it+1}/{maxiter}: resid={Finf:.3e}  "
                  f"α={alpha:.4f}  decr={decr_streak}  "
                  f"perturbs={n_perturb}  α-chg={n_alpha_changes}  "
                  f"elapsed={elapsed:.1f}s{tag}",
                  flush=True)
            t_last_print = elapsed

    Phi_final = rp._phi_map_pchip(P, u, TAUS, GAMMAS, WS)
    Finf_final = float(np.abs(P - Phi_final).max())
    print(f"  Picard done: final resid={Finf_final:.3e}  "
          f"total={time.time()-t_start:.1f}s  "
          f"perturbs={n_perturb}  α-chg={n_alpha_changes}",
          flush=True)
    return P, Finf_final


class HeartbeatCallback:
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


P_warm, Finf_picard = picard_adaptive(P0)


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
