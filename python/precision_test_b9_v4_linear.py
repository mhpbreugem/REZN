"""Batch 9 v4 — LINEAR (older) contour kernel on opposed-heterogeneous γ⟂τ.

Hypothesis: PCHIP's monotone-cubic interpolation may have non-smooth
derivative behavior that prevents Newton from polishing PR fixed points
to machine precision. The older piecewise-linear kernel (rezn_het) has
a different (simpler) basin structure and may admit a true PR fp at
machine precision even where PCHIP cannot.

Same opposed-γ⟂τ config as v3:
    γ = (0.3, 3, 30)   τ = (30, 3, 0.3)

Pipeline:
    1. Adaptive Picard (rh._phi_map, 20000 iters, α-shrink/relax,
       perturb on stall, best-iter tracking)
    2. scipy.optimize.newton_krylov polish from best Picard iterate
       (FD Jacobian, no analytic Jacobian needed)
    3. Report Finf, 1-R², save tensor if it converges with PR > 1%

Heartbeat every 10 s with 1-R².
"""
import time
import pickle
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence
import rezn_het as rh


G = 11
UMAX = 2.0
GAMMAS = np.array([0.3, 3.0, 30.0])
TAUS   = np.array([30.0, 3.0, 0.3])
WS = np.array([1.0, 1.0, 1.0])
EPS_OUTER = 1e-9
HEARTBEAT = 10.0


u = np.linspace(-UMAX, UMAX, G)
P0 = rh._nolearning_price(u, TAUS, GAMMAS, WS)


print("=== JIT warm (linear kernel) ===", flush=True)
t0 = time.time()
_phi = rh._phi_map(P0, u, TAUS, GAMMAS, WS)
print(f"  JIT warm: {time.time()-t0:.1f}s", flush=True)


def picard_adaptive(P_init, alpha0=0.05, alpha_min=0.005, alpha_max=0.15,
                     maxiter=20000, abstol=1e-13,
                     slope_window=100, mono_relax=400,
                     stall_window=500, stall_ratio=0.95,
                     perturb_sigma=1e-6, perturb_seed=42):
    print(f"=== Adaptive Picard (LINEAR): α₀={alpha0}, "
          f"max {maxiter} iters ===", flush=True)
    rng = np.random.default_rng(perturb_seed)
    P = P_init.copy()
    P_best = P.copy()
    Finf_best = float("inf")
    alpha = alpha0
    resid_hist = []
    decr_streak = 0
    last_perturb = -10**6
    last_alpha_chg = -10**6
    n_perturb = 0
    n_alpha_chg = 0
    t_start = time.time()
    t_last_print = -1.0

    for it in range(maxiter):
        Phi = rh._phi_map(P, u, TAUS, GAMMAS, WS)
        F = P - Phi
        Finf = float(np.abs(F).max())
        resid_hist.append(Finf)

        if Finf < 0.5 and Finf < Finf_best:
            Finf_best = Finf
            P_best = P.copy()

        if Finf < abstol:
            elapsed = time.time() - t_start
            print(f"  Picard CONVERGED iter {it+1}: resid={Finf:.3e}  "
                  f"elapsed={elapsed:.1f}s  α={alpha:.4f}",
                  flush=True)
            break

        # α adapt
        alpha_changed = False
        if (it >= slope_window and (it - last_alpha_chg) >= 50):
            window = np.asarray(resid_hist[-slope_window:])
            log_w = np.log(np.maximum(window, 1e-300))
            xs = np.arange(slope_window, dtype=float)
            slope = float(np.polyfit(xs, log_w, 1)[0])
            if slope >= 0.0 and alpha > alpha_min:
                new_a = max(alpha_min, alpha * 0.7)
                if new_a < alpha:
                    alpha = new_a
                    last_alpha_chg = it
                    alpha_changed = True
                    n_alpha_chg += 1
                    decr_streak = 0
        if it > 0 and Finf < resid_hist[-2]:
            decr_streak += 1
            if (decr_streak >= mono_relax and alpha < alpha_max
                    and (it - last_alpha_chg) >= 50):
                new_a = min(alpha_max, alpha * 1.05)
                if new_a > alpha:
                    alpha = new_a
                    last_alpha_chg = it
                    alpha_changed = True
                    n_alpha_chg += 1
                    decr_streak = 0
        else:
            decr_streak = 0

        # stall → perturb
        perturbed = False
        if (it >= 2 * stall_window and (it - last_perturb) >= stall_window):
            recent_min = min(resid_hist[-stall_window:])
            prior_min = min(resid_hist[-2 * stall_window:-stall_window])
            if (prior_min > 0 and recent_min >= stall_ratio * prior_min):
                support = (P > EPS_OUTER * 10) & (P < 1 - EPS_OUTER * 10)
                scale = float(np.mean(P[support])) if support.any() else 0.5
                noise = perturb_sigma * scale * rng.standard_normal(P.shape)
                P = np.clip(P + noise, EPS_OUTER, 1 - EPS_OUTER)
                last_perturb = it
                n_perturb += 1
                perturbed = True

        if not perturbed:
            P = np.clip(alpha * Phi + (1 - alpha) * P,
                          EPS_OUTER, 1 - EPS_OUTER)

        elapsed = time.time() - t_start
        if ((elapsed - t_last_print) >= HEARTBEAT
                or alpha_changed or perturbed
                or it == maxiter - 1):
            tag = ""
            if alpha_changed: tag += f" α→{alpha:.4f}"
            if perturbed:     tag += f" PERTURB σ={perturb_sigma:.0e}"
            try:
                one_r2 = rh.one_minus_R2(P, u, TAUS)
            except Exception:
                one_r2 = float("nan")
            print(f"  Picard iter {it+1}/{maxiter}: resid={Finf:.3e}  "
                  f"1-R²={one_r2:.3e}  α={alpha:.4f}  "
                  f"decr={decr_streak}  perturbs={n_perturb}  "
                  f"α-chg={n_alpha_chg}  elapsed={elapsed:.1f}s{tag}",
                  flush=True)
            t_last_print = elapsed

    Phi_final = rh._phi_map(P, u, TAUS, GAMMAS, WS)
    Finf_final = float(np.abs(P - Phi_final).max())
    print(f"  Picard done: final resid={Finf_final:.3e}  "
          f"best non-sat resid={Finf_best:.3e}  "
          f"total={time.time()-t_start:.1f}s  "
          f"perturbs={n_perturb}  α-chg={n_alpha_chg}",
          flush=True)
    if Finf_best < Finf_final:
        try:
            one_r2_best = rh.one_minus_R2(P_best, u, TAUS)
            print(f"  Using BEST iterate: resid={Finf_best:.3e}  "
                  f"1-R²={one_r2_best:.3e}", flush=True)
        except Exception:
            pass
        return P_best, Finf_best
    return P, Finf_final


P_warm, Finf_picard = picard_adaptive(P0)


# Newton-Krylov polish via scipy (FD Jacobian, works for any Φ)
print("=== scipy.newton_krylov polish ===", flush=True)
def F(P_flat):
    P = P_flat.reshape((G, G, G))
    Pc = np.clip(P, EPS_OUTER, 1 - EPS_OUTER)
    Phi = rh._phi_map(Pc, u, TAUS, GAMMAS, WS)
    return (Pc - Phi).reshape(-1)

t0 = time.time()
try:
    sol_flat = newton_krylov(F, P_warm.reshape(-1), method="lgmres",
                              f_tol=1e-12, maxiter=30, verbose=True,
                              inner_maxiter=80,
                              line_search="armijo")
    P_sol = sol_flat.reshape((G, G, G))
    Phi_sol = rh._phi_map(P_sol, u, TAUS, GAMMAS, WS)
    Finf_sol = float(np.abs(P_sol - Phi_sol).max())
    print(f"  newton_krylov done in {time.time()-t0:.1f}s, "
          f"Finf={Finf_sol:.3e}", flush=True)
except NoConvergence as ex:
    P_sol = ex.args[0].reshape((G, G, G))
    Phi_sol = rh._phi_map(P_sol, u, TAUS, GAMMAS, WS)
    Finf_sol = float(np.abs(P_sol - Phi_sol).max())
    print(f"  newton_krylov NoConvergence in {time.time()-t0:.1f}s, "
          f"Finf={Finf_sol:.3e}", flush=True)
except Exception as ex:
    P_sol = P_warm
    Finf_sol = Finf_picard
    print(f"  newton_krylov FAILED: {ex} (using Picard warm)",
          flush=True)

one_r2 = rh.one_minus_R2(P_sol, u, TAUS)
print(f"\nFINAL  Picard best: {Finf_picard:.3e}  "
      f"NK final: {Finf_sol:.3e}  "
      f"1-R² = {one_r2:.3e}", flush=True)

if Finf_sol < 1e-10 and one_r2 > 0.01:
    fn = ("/home/user/REZN/python/PR_seed_b9_v4_linear_"
           "g0.3-3-30_t30-3-0.3.pkl")
    with open(fn, "wb") as f:
        pickle.dump({"P": P_sol, "taus": TAUS, "gammas": GAMMAS,
                      "G": G, "umax": UMAX,
                      "Finf": Finf_sol, "1-R²": one_r2,
                      "kernel": "linear",
                      "label": "opposed γ=(0.3,3,30) τ=(30,3,0.3)"}, f)
    print(f"  PR! saved to {fn}", flush=True)
