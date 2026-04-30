"""Polish PAVA-Cesaro warm start to med ≤ 1e-13.

Strategy:
1. Use PAVA-Cesaro μ as warm seed (u-monotone, p-monotone, med ~1e-3)
2. NK with gap-reparam-u: solves F(x) = encode(Φ(decode(x))) - x to machine
   precision while preserving u-monotonicity.
3. Check that decode(x_NK) is still p-monotone (warm start should keep us
   in the right basin).
4. If p-violations creep up, fall back to slow Picard-PAVA (5000 iters,
   damping schedule 0.05→0.01→0.005, Cesaro avg over last 2500).
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import (
    encode, decode, pack, unpack, pava_p_only, pava_u_only,
)


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def F_residual_gap_pure(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """F = encode(Φ(decode(x))) - x; pure (no PAVA-p inside)."""
    base, c = unpack(x, Gu, Gp)
    mu = decode(base, c)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    base_new, c_new = encode(cand)
    return pack(base_new - base, c_new - c)


def slow_picard_pava(mu_seed, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                     N_TOTAL=5000, N_AVG=2500):
    """Long Picard-PAVA with damping schedule + Cesaro averaging."""
    mu = pava_2d(mu_seed.copy())
    mu_sum = np.zeros_like(mu)
    n_collected = 0
    for it in range(N_TOTAL):
        # Damping schedule: 0.05 first 1k, 0.01 next 2k, 0.005 last 2k
        if it < 1000:
            alpha = 0.05
        elif it < 3000:
            alpha = 0.01
        else:
            alpha = 0.005
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= N_TOTAL - N_AVG:
            mu_sum += mu
            n_collected += 1
    return pava_2d(mu_sum / n_collected)


def polish_to_eminus13(mu_warm, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                       f_tol=1e-13, log_label=""):
    """Polish a warm seed to med-residual ≤ f_tol via NK gap-reparam.
    Falls back to slow Picard-PAVA if NK fails or p-violations creep up."""
    Gu, Gp = mu_warm.shape
    base0, c0 = encode(mu_warm)
    x0 = pack(base0, c0)

    # NK polish
    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual_gap_pure(x, Gu, Gp, u_grid, p_grid,
                                           p_lo, p_hi, tau, gamma),
            x0, f_tol=f_tol * 100, maxiter=300,   # tol in (base,c) space ~ 100x
            method="lgmres", verbose=False,
        )
        x_final = sol
    except NoConvergence as e:
        nk_status = "noconv"
        x_final = e.args[0] if e.args else x0
    except (ValueError, RuntimeError) as exc:
        nk_status = f"err:{type(exc).__name__}"
        x_final = x0

    base, c = unpack(x_final, Gu, Gp)
    mu_nk = decode(base, c)

    # Check p-monotonicity AND Φ-residual
    cand, active, _ = phi_step(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu_nk)
    res_max = float(F[active].max()) if active.any() else float("nan")
    res_med = float(np.median(F[active])) if active.any() else float("nan")
    n_u = int((np.diff(mu_nk, axis=0) < 0).sum())
    n_p = int((np.diff(mu_nk, axis=1) < 0).sum())
    print(f"  [{log_label}] after NK: max={res_max:.3e}  med={res_med:.3e}  "
          f"u={n_u}, p={n_p}, NK={nk_status}", flush=True)

    if res_med <= f_tol and n_u == 0 and n_p == 0:
        return mu_nk, res_max, res_med, n_u, n_p, "NK polish"

    # Fall back to slow Picard-PAVA
    print(f"  [{log_label}] NK insufficient — trying slow Picard-PAVA polish",
          flush=True)
    mu_slow = slow_picard_pava(mu_warm, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                                 N_TOTAL=5000, N_AVG=2500)
    cand, active, _ = phi_step(mu_slow, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu_slow)
    res_max2 = float(F[active].max()) if active.any() else float("nan")
    res_med2 = float(np.median(F[active])) if active.any() else float("nan")
    n_u2 = int((np.diff(mu_slow, axis=0) < 0).sum())
    n_p2 = int((np.diff(mu_slow, axis=1) < 0).sum())
    print(f"  [{log_label}] after slow Picard: max={res_max2:.3e}  "
          f"med={res_med2:.3e}  u={n_u2}, p={n_p2}", flush=True)
    return mu_slow, res_max2, res_med2, n_u2, n_p2, "slow Picard"


warnings.filterwarnings("ignore", category=RuntimeWarning)
# --- Test at G=14 ---
print("=== Polishing G=14 PAVA-Cesaro to med ≤ 1e-13 ===\n", flush=True)
ck = np.load("results/full_ree/posterior_v3_pava_G14_gamma0.5.npz")
mu_warm = ck["mu"]; u_grid = ck["u_grid"]; p_grid = ck["p_grid"]
p_lo = ck["p_lo"]; p_hi = ck["p_hi"]
print("Warm seed: G=14 PAVA-Cesaro, γ=0.5, τ=2")
cand, active, _ = phi_step(mu_warm, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
F = np.abs(cand - mu_warm)
print(f"  warm: max={F[active].max():.3e}, med={np.median(F[active]):.3e}",
      flush=True)
r2_w, sl_w, _ = measure_R2(mu_warm, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
print(f"  warm: 1-R²={r2_w:.4e}, slope={sl_w:.4f}\n", flush=True)

t0 = time.time()
mu_p, mx, md, nu, np_, method = polish_to_eminus13(
    mu_warm, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5,
    f_tol=1e-13, log_label="G=14"
)
elapsed = time.time() - t0
r2_p, sl_p, _ = measure_R2(mu_p, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
print(f"\nFinal: max={mx:.3e}, med={md:.3e}, u={nu}, p={np_}")
print(f"       method={method}, t={elapsed:.1f}s")
print(f"       1-R²={r2_p:.6e}, slope={sl_p:.4f}")

np.savez("results/full_ree/posterior_v3_G14_polished.npz",
         mu=mu_p, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
print("\nSaved: results/full_ree/posterior_v3_G14_polished.npz")
