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


def pava_p_only(mu):
    """Project μ to be non-decreasing in p (column-wise across each row).
    L²-closest non-decreasing array per row."""
    out = mu.copy()
    Gu, Gp = mu.shape
    for i in range(Gu):
        # PAVA / monotone regression on out[i, :]
        y = out[i, :].copy()
        w = np.ones(Gp)
        # Pool-adjacent-violators
        means = list(y)
        weights = list(w)
        while True:
            done = True
            j = 0
            while j < len(means) - 1:
                if means[j] > means[j+1]:
                    new_w = weights[j] + weights[j+1]
                    new_m = (weights[j]*means[j] + weights[j+1]*means[j+1]) / new_w
                    means[j] = new_m
                    weights[j] = new_w
                    means.pop(j+1)
                    weights.pop(j+1)
                    done = False
                else:
                    j += 1
            if done:
                break
        # Spread blocks back out
        result = np.empty(Gp)
        idx = 0
        for k in range(len(means)):
            wk = int(weights[k])
            result[idx:idx+wk] = means[k]
            idx += wk
        out[i, :] = result
    return out


def F_residual_gap(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                    pava_p=False):
    """F((base, c)) = encode(Φ(decode(base, c))) - (base, c).

    With pava_p=True: project Φ-output to p-monotone before encoding.
    Then any FP has both u-monotonicity (from gap reparam) and
    p-monotonicity (from PAVA in p).
    """
    base, c = unpack(x, Gu, Gp)
    mu = decode(base, c)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    if pava_p:
        cand = pava_p_only(cand)
    # Encode the (possibly p-projected) φ output with gap reparametrization
    base_new, c_new = encode(cand)
    return pack(base_new - base, c_new - c)


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
                            f_tol=1e-12, maxiter=400, pava_p=False,
                            log_interval=10.0):
    Gu, Gp = mu_seed.shape
    if pava_p:
        # Make sure the seed is p-monotone too
        mu_seed = pava_p_only(mu_seed)
    base0, c0 = encode(mu_seed)
    x0 = pack(base0, c0)
    t0 = time.time()
    cb = progress_callback(t0, log_interval=log_interval)

    nk_status = "ok"
    try:
        sol = newton_krylov(
            lambda x: F_residual_gap(x, Gu, Gp, u_grid, p_grid, p_lo, p_hi,
                                       tau, gamma, pava_p=pava_p),
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

    def report(label, r):
        r2, slope, _ = measure_R2(r["mu_final"], u_grid, p_grid, p_lo, p_hi,
                                   TAU, GAMMA)
        print(f"{label}: NK={r['nk_status']}  "
              f"Φ-resid max={r['phi_residual_max']:.3e}, "
              f"med={r['phi_residual_med']:.3e}  "
              f"u-viol={r['violations_u']}, p-viol={r['violations_p']}  "
              f"1-R²={r2:.4e}, slope={slope:.4f}  t={r['elapsed_s']:.1f}s",
              flush=True)
        return r2, slope

    print("\n--- Test A: cold + gap-reparam u + PAVA-p (hybrid) ---",
          flush=True)
    r = solve_with_gap_reparam(mu_seed, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
                                f_tol=1e-9, maxiter=200, pava_p=True,
                                log_interval=10.0)
    rA = report("Test A", r)
    np.savez(f"results/full_ree/posterior_v3_G{G}_gap_pava_hybrid_cold.npz",
             mu=r["mu_final"], u_grid=u_grid, p_grid=p_grid,
             p_lo=p_lo, p_hi=p_hi)

    print("\n--- Test B: warm from PAVA-Cesaro + hybrid ---", flush=True)
    ck = np.load(f"results/full_ree/posterior_v3_G{G}_PAVA_cesaro_mu.npz")
    r2_seed, slope_seed, _ = measure_R2(ck["mu"], u_grid, p_grid, p_lo, p_hi,
                                          TAU, GAMMA)
    print(f"  PAVA-Cesaro seed: 1-R²={r2_seed:.4e}, slope={slope_seed:.4f}",
          flush=True)
    r = solve_with_gap_reparam(ck["mu"], u_grid, p_grid, p_lo, p_hi, TAU, GAMMA,
                                f_tol=1e-9, maxiter=200, pava_p=True,
                                log_interval=10.0)
    rB = report("Test B", r)
    np.savez(f"results/full_ree/posterior_v3_G{G}_gap_pava_hybrid_warm.npz",
             mu=r["mu_final"], u_grid=u_grid, p_grid=p_grid,
             p_lo=p_lo, p_hi=p_hi)

    print(f"\n=== SUMMARY ===")
    print(f"  Cold seed  1-R²={rA[0]:.4e}  slope={rA[1]:.4f}")
    print(f"  Warm PAVA  1-R²={rB[0]:.4e}  slope={rB[1]:.4f}")
    print(f"  PAVA seed  1-R²={r2_seed:.4e}  slope={slope_seed:.4f}")
