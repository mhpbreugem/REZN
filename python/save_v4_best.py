"""Save the v4 best iterate by re-running Picard and dumping the result.

The v4 Picard run produced its best at Finf=3.67e-3, 1-R²=21.0% with the
opposed-heterogeneous config. Re-runs are deterministic (seed=42), so we
can reproduce and pickle the tensor.
"""
import pickle
import numpy as np
import rezn_het as rh

G = 11
UMAX = 2.0
GAMMAS = np.array([0.3, 3.0, 30.0])
TAUS   = np.array([30.0, 3.0, 0.3])
WS = np.array([1.0, 1.0, 1.0])
EPS_OUTER = 1e-9

u = np.linspace(-UMAX, UMAX, G)
P0 = rh._nolearning_price(u, TAUS, GAMMAS, WS)


def picard_adaptive(P_init, alpha0=0.05, alpha_min=0.005, alpha_max=0.15,
                     maxiter=20000, slope_window=100, mono_relax=400,
                     stall_window=500, stall_ratio=0.95,
                     perturb_sigma=1e-6, perturb_seed=42):
    rng = np.random.default_rng(perturb_seed)
    P = P_init.copy()
    P_best = P.copy()
    Finf_best = float("inf")
    alpha = alpha0
    resid_hist = []
    decr_streak = 0
    last_perturb = -10**6
    last_alpha_chg = -10**6
    for it in range(maxiter):
        Phi = rh._phi_map(P, u, TAUS, GAMMAS, WS)
        F = P - Phi
        Finf = float(np.abs(F).max())
        resid_hist.append(Finf)
        if Finf < 0.5 and Finf < Finf_best:
            Finf_best = Finf; P_best = P.copy()
        # α adapt (slope/relax)
        if it >= slope_window and (it - last_alpha_chg) >= 50:
            window = np.asarray(resid_hist[-slope_window:])
            log_w = np.log(np.maximum(window, 1e-300))
            xs = np.arange(slope_window, dtype=float)
            slope = float(np.polyfit(xs, log_w, 1)[0])
            if slope >= 0 and alpha > alpha_min:
                alpha = max(alpha_min, alpha * 0.7); last_alpha_chg = it; decr_streak = 0
        if it > 0 and Finf < resid_hist[-2]:
            decr_streak += 1
            if (decr_streak >= mono_relax and alpha < alpha_max
                    and (it - last_alpha_chg) >= 50):
                alpha = min(alpha_max, alpha * 1.05); last_alpha_chg = it; decr_streak = 0
        else:
            decr_streak = 0
        # stall → perturb
        perturbed = False
        if it >= 2 * stall_window and (it - last_perturb) >= stall_window:
            recent_min = min(resid_hist[-stall_window:])
            prior_min = min(resid_hist[-2 * stall_window:-stall_window])
            if prior_min > 0 and recent_min >= stall_ratio * prior_min:
                support = (P > EPS_OUTER * 10) & (P < 1 - EPS_OUTER * 10)
                scale = float(np.mean(P[support])) if support.any() else 0.5
                P = np.clip(P + perturb_sigma * scale * rng.standard_normal(P.shape),
                              EPS_OUTER, 1 - EPS_OUTER)
                last_perturb = it; perturbed = True
        if not perturbed:
            P = np.clip(alpha * Phi + (1 - alpha) * P,
                          EPS_OUTER, 1 - EPS_OUTER)
    return P_best, Finf_best


print("Re-running v4 Picard to recover best iterate...", flush=True)
P_best, Finf_best = picard_adaptive(P0)
one_r2 = rh.one_minus_R2(P_best, u, TAUS)
print(f"Recovered: Finf={Finf_best:.3e}  1-R²={one_r2:.3e}", flush=True)

fn = "/home/user/REZN/python/PR_seed_b9_v4_opposed_LINEAR.pkl"
with open(fn, "wb") as f:
    pickle.dump({"P": P_best, "taus": TAUS, "gammas": GAMMAS,
                  "G": G, "umax": UMAX,
                  "Finf": Finf_best, "1-R²": one_r2,
                  "kernel": "linear (rezn_het._phi_map)",
                  "label": "opposed γ=(0.3,3,30) τ=(30,3,0.3)",
                  "note": ("Best non-saturated iterate from adaptive Picard, "
                            "20000 iters. Newton-Krylov could not polish "
                            "below 3.67e-3. PR signal of ~21% is robust "
                            "across saturation/perturb cycles.")}, f)
print(f"Saved → {fn}", flush=True)
