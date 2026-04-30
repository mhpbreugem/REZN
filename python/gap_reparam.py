"""Implement Section E gap reparametrization for monotonicity in u.

Reparametrize μ along u-axis as cumulative exp-gaps in logit space:
    base_j = logit(μ(u_1, p_j))
    logit(μ(u_k, p_j)) = logit(μ(u_{k-1}, p_j)) + exp(c_k^j),  k≥2
=> monotone increasing in u by construction, C^∞ everywhere,
   nonsingular Jacobian.

State vector x ∈ ℝ^{G_p × G_u}: x = (base, c).flatten()
- base: shape (G_p,)
- c:    shape (G_p, G_u - 1)

Newton-Krylov on F(x) = T(Φ_μ(reconstruct(x))) - x, where T is the
encode operator from μ to (base, c) with log(max(gap, ε)) clamping.

Optionally adds soft penalty for p-direction monotonicity.
"""
import numpy as np
import time
import warnings
import math
from scipy.optimize import newton_krylov, NoConvergence
from scipy.special import expit, logit as sp_logit

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)

EPS_GAP = 1e-12   # log clamp for non-monotone Φ output


def encode(mu):
    """μ (G_u, G_p) → (base, c).
    base: (G_p,)   c: (G_p, G_u-1)."""
    Gu, Gp = mu.shape
    lm = sp_logit(np.clip(mu, EPS, 1.0 - EPS))   # (G_u, G_p)
    base = lm[0, :]                                # (G_p,)
    gaps = np.diff(lm, axis=0)                     # (G_u-1, G_p) — increment per step
    c = np.log(np.maximum(gaps, EPS_GAP)).T        # (G_p, G_u-1)
    return base, c


def decode(base, c):
    """(base[G_p], c[G_p, G_u-1]) → μ (G_u, G_p)."""
    Gp, Gu_minus_1 = c.shape
    Gu = Gu_minus_1 + 1
    cum = np.concatenate([np.zeros((Gp, 1)), np.cumsum(np.exp(c), axis=1)],
                          axis=1)               # (G_p, G_u)
    logit_mu = base[:, None] + cum               # (G_p, G_u)
    return expit(logit_mu).T                     # (G_u, G_p), monotone in u


def pack(base, c):
    """Flatten (base, c) to vector."""
    return np.concatenate([base, c.ravel()])


def unpack(x, Gu, Gp):
    """Vector → (base, c)."""
    base = x[:Gp]
    c = x[Gp:].reshape(Gp, Gu - 1)
    return base, c


def F_residual_gap(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                    p_pen=0.0):
    """F((base, c)) = encode(Φ(decode(base, c))) - (base, c).

    Optional p-penalty in the residual (soft).
    """
    base, c = unpack(x, Gu, Gp)
    mu = decode(base, c)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    # Encode the φ output with the same gap reparametrization
    base_new, c_new = encode(cand)
    F = pack(base_new - base, c_new - c)

    if p_pen > 0:
        # Add penalty for p-direction monotonicity violations.
        # Penalty: P(μ) = ½ Σ ReLU(μ[i,j] - μ[i,j+1])²
        # Gradient w.r.t. μ adds to the residual after we encode.
        # Since the residual is in (base, c) space, we map back:
        diff_p = mu[:, :-1] - mu[:, 1:]
        viol = np.maximum(diff_p, 0.0)            # (G_u, G_p-1)
        # Gradient w.r.t. μ: g[:,:-1] += viol, g[:,1:] -= viol
        g = np.zeros_like(mu)
        g[:, :-1] += viol
        g[:, 1:] -= viol
        # Convert to gap-space: ∂μ/∂base affects all rows; ∂μ/∂c[k] affects rows ≥k+1.
        # Easier: encode (μ - p_pen * g), but that doesn't compose linearly.
        # Approximate: add p_pen * (encode(mu - g_step) - encode(mu)) ≈ p_pen * (... ).
        # Use a finite-difference-style derivative:
        mu_pen = np.clip(mu - 0.001 * g, EPS, 1 - EPS)
        b_p, c_p = encode(mu_pen)
        b0, c0 = encode(mu)
        F += p_pen * pack(b_p - b0, c_p - c0) / 0.001

    return F


def progress_callback(t_start, log_interval=10.0):
    """Print every `log_interval` seconds during NK."""
    state = {"last_t": t_start, "iter": 0}
    def cb(x, f):
        state["iter"] += 1
        now = time.time()
        if now - state["last_t"] >= log_interval:
            elapsed = now - t_start
            r_max = float(np.max(np.abs(f)))
            r_med = float(np.median(np.abs(f)))
            r_l2 = float(np.sqrt(np.mean(f**2)))
            print(f"  [{elapsed:6.1f}s, NK iter {state['iter']:3d}]  "
                  f"max={r_max:.3e}  med={r_med:.3e}  L2={r_l2:.3e}",
                  flush=True)
            state["last_t"] = now
    return cb


def solve_with_gap_reparam(mu_seed, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                            f_tol=1e-12, maxiter=400, p_pen=0.0,
                            log_interval=10.0):
    Gu, Gp = mu_seed.shape
    base0, c0 = encode(mu_seed)
    x0 = pack(base0, c0)
    t0 = time.time()
    cb = progress_callback(t0, log_interval=log_interval)

    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual_gap(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma, p_pen=p_pen),
            x0, f_tol=f_tol, maxiter=maxiter,
            method="lgmres", verbose=False, callback=cb,
        )
        x_final = sol
    except NoConvergence as e:
        nk_status = "noconv"
        x_final = e.args[0] if e.args else x0
    except (ValueError, RuntimeError) as exc:
        nk_status = f"err:{type(exc).__name__}:{exc}"
        x_final = x0

    base_f, c_f = unpack(x_final, Gu, Gp)
    mu_final = decode(base_f, c_f)

    # Diagnostics
    cand, active, _ = phi_step(mu_final, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F_arr = np.abs(cand - mu_final)
    res_max = float(F_arr[active].max()) if active.any() else float("nan")
    res_med = float(np.median(F_arr[active])) if active.any() else float("nan")
    n_u_viol = int((np.diff(mu_final, axis=0) < 0).sum())
    n_p_viol = int((np.diff(mu_final, axis=1) < 0).sum())
    elapsed = time.time() - t0
    return {
        "mu_final": mu_final,
        "nk_status": nk_status,
        "phi_residual_max": res_max,
        "phi_residual_med": res_med,
        "violations_u": n_u_viol,
        "violations_p": n_p_viol,
        "elapsed_s": elapsed,
    }


# ---- Main: run at G=14, γ=0.5, τ=2 from no-learning seed and report ----
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    G = 14
    UMAX = 4.0
    TAU = 2.0
    GAMMA = 0.5

    print(f"=== Gap reparametrization NK ===")
    print(f"G={G}, UMAX={UMAX}, τ={TAU}, γ={GAMMA}", flush=True)

    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
    mu_seed = np.zeros((G, G))
    for i, u in enumerate(u_grid):
        mu_seed[i, :] = Lam(TAU * u)

    print("\n--- Test 1: cold start, p_pen=0 (u-monotone only) ---", flush=True)
    r = solve_with_gap_reparam(mu_seed, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
                                f_tol=1e-12, maxiter=400, p_pen=0.0,
                                log_interval=10.0)
    r2, slope, _ = measure_R2(r["mu_final"], u_grid, p_grid, p_lo, p_hi,
                               TAU, GAMMA)
    print(f"NK={r['nk_status']}  Φ-resid max={r['phi_residual_max']:.3e}, "
          f"med={r['phi_residual_med']:.3e}  "
          f"u-viol={r['violations_u']}, p-viol={r['violations_p']}  "
          f"1-R²={r2:.4e}, slope={slope:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)
    np.savez(f"results/full_ree/posterior_v3_G{G}_gap_reparam.npz",
             mu=r["mu_final"], u_grid=u_grid, p_grid=p_grid,
             p_lo=p_lo, p_hi=p_hi)
    print(f"\nSaved μ to results/full_ree/posterior_v3_G{G}_gap_reparam.npz")

    print("\n--- Test 2: warm from PAVA-Cesaro, p_pen=0 ---", flush=True)
    ck = np.load(f"results/full_ree/posterior_v3_G{G}_PAVA_cesaro_mu.npz")
    r2_seed, slope_seed, _ = measure_R2(ck["mu"], u_grid, p_grid, p_lo, p_hi,
                                          TAU, GAMMA)
    print(f"  PAVA-Cesaro seed: 1-R²={r2_seed:.4e}, slope={slope_seed:.4f}",
          flush=True)
    r = solve_with_gap_reparam(ck["mu"], u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
                                f_tol=1e-12, maxiter=400, p_pen=0.0,
                                log_interval=10.0)
    r2, slope, _ = measure_R2(r["mu_final"], u_grid, p_grid, p_lo, p_hi,
                               TAU, GAMMA)
    print(f"NK={r['nk_status']}  Φ-resid max={r['phi_residual_max']:.3e}, "
          f"med={r['phi_residual_med']:.3e}  "
          f"u-viol={r['violations_u']}, p-viol={r['violations_p']}  "
          f"1-R²={r2:.4e}, slope={slope:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)

    print("\n--- Test 3: warm + p_pen=0.1 ---", flush=True)
    r = solve_with_gap_reparam(ck["mu"], u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
                                f_tol=1e-12, maxiter=400, p_pen=0.1,
                                log_interval=10.0)
    r2, slope, _ = measure_R2(r["mu_final"], u_grid, p_grid, p_lo, p_hi,
                               TAU, GAMMA)
    print(f"NK={r['nk_status']}  Φ-resid max={r['phi_residual_max']:.3e}, "
          f"med={r['phi_residual_med']:.3e}  "
          f"u-viol={r['violations_u']}, p-viol={r['violations_p']}  "
          f"1-R²={r2:.4e}, slope={slope:.4f}  t={r['elapsed_s']:.1f}s",
          flush=True)
