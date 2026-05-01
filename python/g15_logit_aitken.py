"""G=15 polish: logit-space iteration + Aitken Δ² acceleration.

Φ has fixed points μ* with same fixed points as the logit-space map
   Ψ(λ) := logit(Φ(σ(λ)))
(σ = sigmoid, Λ; bijection between μ and λ).

Aitken Δ² extrapolation: from λ_0, λ_1, λ_2 compute
   λ̂ = λ_2 - (λ_2 - λ_1)² / (λ_2 - 2λ_1 + λ_0)   (elementwise)

If Picard converges geometrically (residual ~ ρ^n) then Aitken extracts
the limit ρ^∞ directly. Each Aitken cycle doubles effective precision.

Cycle: 3 Picard steps in λ → Aitken extrapolate → repeat.
"""
import time, json, warnings, math
import numpy as np
from scipy.special import expit, logit as sp_logit

from posterior_method_v3 import (
    init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
G = 15
TAU = 2.0
GAMMA = 0.5
UMAX = 4.0
LOGIT_CLIP = 30.0  # |logit| ≤ 30 (μ ∈ [σ(-30), σ(30)])


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def Psi(lam, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """Logit-space map: Ψ(λ) = logit(Φ(σ(λ)))."""
    mu = expit(np.clip(lam, -LOGIT_CLIP, LOGIT_CLIP))
    mu = np.clip(mu, EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    cand = pava_2d(cand)
    cand = np.clip(cand, EPS, 1 - EPS)
    lam_new = sp_logit(cand)
    return np.clip(lam_new, -LOGIT_CLIP, LOGIT_CLIP), active


def measure_residual(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()),
        "med": float(np.median(F[active])),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def measure_logit_residual(lam, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    lam_new, active = Psi(lam, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(lam_new - lam)
    return {
        "max_logit": float(F[active].max()),
        "med_logit": float(np.median(F[active])),
    }


def aitken_extrapolate(lam0, lam1, lam2, threshold=1e-30):
    """Aitken Δ² extrapolation, elementwise."""
    delta1 = lam2 - lam1
    delta0 = lam1 - lam0
    denom = delta1 - delta0   # lam2 - 2 lam1 + lam0
    out = lam2.copy()
    safe = np.abs(denom) > threshold
    out[safe] = lam2[safe] - delta1[safe]**2 / denom[safe]
    return np.clip(out, -LOGIT_CLIP, LOGIT_CLIP)


# Load G=15 strict
ck = np.load(f"{RESULTS_DIR}/posterior_v3_strict_G15.npz")
mu0 = ck["mu"]; u_grid = ck["u_grid"]; p_grid = ck["p_grid"]
p_lo = ck["p_lo"]; p_hi = ck["p_hi"]

print(f"=== G={G} logit-space + Aitken Δ² ===\n", flush=True)
d0 = measure_residual(mu0, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"Initial (G=15 strict): max={d0['max']:.3e}, med={d0['med']:.3e}",
      flush=True)

lam = sp_logit(np.clip(mu0, EPS, 1 - EPS))
print(f"Logit range: [{lam.min():.2f}, {lam.max():.2f}]\n", flush=True)

# Iterate Aitken cycles
N_CYCLES = 30
history = [{"cycle": 0, "max": d0["max"], "med": d0["med"]}]
for cyc in range(1, N_CYCLES + 1):
    t0 = time.time()
    # 3 Picard steps in λ
    lam0_c = lam.copy()
    lam1_c, _ = Psi(lam0_c, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    lam2_c, _ = Psi(lam1_c, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    # Aitken extrapolate
    lam_new = aitken_extrapolate(lam0_c, lam1_c, lam2_c)
    # PAVA project (in μ space) for monotonicity
    mu_new = expit(lam_new)
    mu_new = pava_2d(mu_new)
    mu_new = np.clip(mu_new, EPS, 1 - EPS)
    lam = sp_logit(mu_new)
    d = measure_residual(mu_new, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
    dlogit = measure_logit_residual(lam, u_grid, p_grid, p_lo, p_hi,
                                       TAU, GAMMA)
    print(f"  cycle {cyc:>2}: μ-max={d['max']:.3e}, μ-med={d['med']:.3e}, "
          f"λ-max={dlogit['max_logit']:.3e}, "
          f"u/p={d['u_viol']}/{d['p_viol']}, t={time.time()-t0:.1f}s",
          flush=True)
    history.append({"cycle": cyc, "max": d["max"], "med": d["med"],
                      "logit_max": dlogit["max_logit"]})

mu_final = expit(lam)
mu_final = pava_2d(mu_final)
mu_final = np.clip(mu_final, EPS, 1 - EPS)
d_final = measure_residual(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
r2, slope, _ = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\nFINAL: μ-max={d_final['max']:.3e}, μ-med={d_final['med']:.3e}, "
      f"u/p={d_final['u_viol']}/{d_final['p_viol']}", flush=True)
print(f"      1-R²={r2:.6e}, slope={slope:.6f}", flush=True)

np.savez(f"{RESULTS_DIR}/posterior_v3_G15_aitken.npz",
         mu=mu_final, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
with open(f"{RESULTS_DIR}/posterior_v3_G15_aitken_history.json", "w") as f:
    json.dump({"params": {"G": G, "tau": TAU, "gamma": GAMMA,
                            "method": "logit_space_picard + Aitken"},
                "history": history,
                "final": {"max": d_final["max"], "med": d_final["med"],
                          "1-R^2": r2, "slope": slope}}, f, indent=2)
print(f"\nSaved {RESULTS_DIR}/posterior_v3_G15_aitken.{{npz,history.json}}",
      flush=True)
