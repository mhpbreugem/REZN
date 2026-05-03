"""Task 3 ladder version: warm-start chain in τ for γ ∈ {0.5, 1.0, 4.0}.

Per FIGURES_TODO + user request: add intermediate τ values (4.3, 4.7,
5.5, 6.0, 8.0, 12.5) and chain warm-starts to break boundary stagnation.

Chain from τ=2 (converged seed) walking down/up:
  Down: 2 → 1.5 → 1.0 → 0.8 → 0.5 → 0.3
  Up:   2 → 3 → 4 → 4.3 → 4.7 → 5 → 5.5 → 6 → 7 → 8 → 10 → 12.5 → 15

For each τ: warm-start mu by interp from previous τ's converged μ
onto current p-grid. Then float64 picard polish + mp50 LM.

Save all ckpts (incl. intermediate τ's).
Skip if ckpt already exists.
"""
import time, json, warnings, os.path
import numpy as np
import mpmath
from mpmath import mp, mpf
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 50

import sys
sys.path.insert(0, "python")
from task3_tau_sweep import (
    G, UMAX, TRIM, H_FD, TARGET, gamma_tag, tau_tag,
    Lam_mp, logit_mp, crra_demand_mp, f_v_mp, interp_mp, phi_step_mp,
    F_mu, F_max, F_med, lm_step, nk_solve,
    Lam, logit, crra_d_f64, mu_at, market_clear,
    weighted_R2_grid, to_floats, save_ckpt,
    init_p_grid_f64, phi_step_f64, EPS_F64, pava_2d_f64,
)

RESULTS_DIR = "results/full_ree"
GAMMAS = [0.5, 1.0, 4.0]

# τ chain: full ladder including intermediates
TAU_CHAIN_DOWN = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3]
TAU_CHAIN_UP = [2.0, 3.0, 4.0, 4.3, 4.7, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0,
                12.5, 15.0]

# τ=2 ckpts (seed for chain)
TAU2_CKPT = {
    0.5: f"{RESULTS_DIR}/posterior_v3_G20_umax5_trim05_mp300.json",
    1.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g100_mp50.json",
    4.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g400_mp50.json",
}


def load_mu_from_ckpt(path):
    with open(path) as f:
        d = json.load(f)
    mu_arr = np.array([[float(s) for s in row] for row in d["mu_strings"]])
    p_grid_arr = np.array([[float(s) for s in row] for row in d["p_grid"]])
    return mu_arr, p_grid_arr


def interp_mu_to_new_pgrid(mu_old, p_old, p_new):
    """Interp mu values from old per-row p-grid to new per-row p-grid.

    Both have same u-grid structure (G rows). p-grids differ per row.
    """
    G = len(mu_old)
    out = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            p = p_new[i, j]
            if p <= p_old[i, 0]: out[i, j] = mu_old[i, 0]
            elif p >= p_old[i, -1]: out[i, j] = mu_old[i, -1]
            else: out[i, j] = float(np.interp(p, p_old[i], mu_old[i]))
    return out


def solve_one_warm(gamma_f, tau_f, mu_warm_arr, p_old_arr, u_grid_np,
                       u_grid_mp):
    """Warm-start solve at (γ, τ). mu_warm_arr is float64 array on p_old_arr.
    Returns (mu_conv, F_max, F_med, history, one_R2, slope, p_grid_np).
    """
    gamma_mp = mpf(str(gamma_f))
    tau_mp = mpf(str(tau_f))
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, tau_f, gamma_f,
                                                       G, trim=TRIM)
    p_grid_mp = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo_mp = [mpf(str(x)) for x in p_lo_np]
    p_hi_mp = [mpf(str(x)) for x in p_hi_np]

    # Interpolate mu from old p-grid to new p-grid
    mu_f = interp_mu_to_new_pgrid(mu_warm_arr, p_old_arr, p_grid_np)
    mu_f = pava_2d_f64(mu_f)

    # Light float64 polish (warm start should already be close)
    cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np, p_lo_np, p_hi_np,
                                       tau_f, gamma_f)
    F0 = float(np.max(np.abs(cand2 - mu_f)[act2]))
    print(f"  Warm interp residual: max={F0:.3e}", flush=True)

    last_status = time.time()
    for round_idx, (n_iter, n_avg, alpha) in enumerate(
            [(2000, 1000, 0.005), (2000, 1000, 0.002),
             (3000, 1500, 0.001)]):
        mu_sum = np.zeros_like(mu_f); n_collected = 0
        for it in range(n_iter):
            cand, active, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                              p_lo_np, p_hi_np, tau_f, gamma_f)
            cand = pava_2d_f64(cand)
            mu_f = alpha * cand + (1 - alpha) * mu_f
            mu_f = np.clip(mu_f, EPS_F64, 1 - EPS_F64)
            if it >= n_iter - n_avg:
                mu_sum += mu_f; n_collected += 1
            if time.time() - last_status > 30:
                cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                                  p_lo_np, p_hi_np,
                                                  tau_f, gamma_f)
                r = float(np.max(np.abs(cand2 - mu_f)[act2]))
                print(f"    polish r{round_idx+1} α={alpha} "
                      f"it {it+1}/{n_iter}: max={r:.3e}", flush=True)
                last_status = time.time()
        mu_f = pava_2d_f64(mu_sum / max(n_collected, 1))
    cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np, p_lo_np, p_hi_np,
                                      tau_f, gamma_f)
    F_f64 = float(np.max(np.abs(cand2 - mu_f)[act2]))
    print(f"  Polish done: max={F_f64:.3e}", flush=True)

    mu_init = [[mpf(str(mu_f[i, j])) for j in range(G)] for i in range(G)]
    tag = f"γ={gamma_f},τ={tau_f}"
    mu_conv, F_max_v, F_med_v, history = nk_solve(
        mu_init, u_grid_mp, p_grid_mp, p_lo_mp, p_hi_mp, tau_mp, gamma_mp, tag)

    mu_arr = to_floats(mu_conv)
    one_R2, slope = weighted_R2_grid(u_grid_np, p_grid_np, mu_arr,
                                          gamma_f, tau_f)
    return (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
            p_grid_np, mu_arr)


def main():
    t_start = time.time()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    u_grid_mp = [mpf(str(x)) for x in u_grid_np]

    for gamma_f in GAMMAS:
        print(f"\n{'='*60}\n=== γ={gamma_f} τ-ladder ===\n{'='*60}",
              flush=True)
        # Walk DOWN from τ=2 ckpt
        print(f"\n--- Walking τ DOWN from 2 ---", flush=True)
        mu_warm, p_old = load_mu_from_ckpt(TAU2_CKPT[gamma_f])
        for tau_f in TAU_CHAIN_DOWN[1:]:  # skip τ=2 itself
            existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                        f"{gamma_tag(gamma_f)}_{tau_tag(tau_f)}_mp50.json")
            if os.path.exists(existing):
                with open(existing) as f:
                    d = json.load(f)
                # Use existing for warm-start to next
                mu_warm, p_old = load_mu_from_ckpt(existing)
                print(f"  Skipping γ={gamma_f}, τ={tau_f} (exists, "
                      f"1-R²={d.get('1-R2_weighted', 'n/a')})", flush=True)
                continue
            print(f"\n=== γ={gamma_f}, τ={tau_f} (warm) ===", flush=True)
            t0 = time.time()
            try:
                (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
                 p_grid_np, mu_arr) = solve_one_warm(gamma_f, tau_f,
                                                        mu_warm, p_old,
                                                        u_grid_np, u_grid_mp)
                save_ckpt(gamma_f, tau_f, mu_conv, u_grid_np, p_grid_np,
                              F_max_v, F_med_v, history, one_R2, slope)
                mu_warm = mu_arr; p_old = p_grid_np
                print(f"  γ={gamma_f}, τ={tau_f}: 1-R²={one_R2:.6e}, "
                      f"slope={slope:.4f} (t={time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  γ={gamma_f}, τ={tau_f} FAILED: {e}")

        # Walk UP from τ=2 ckpt
        print(f"\n--- Walking τ UP from 2 ---", flush=True)
        mu_warm, p_old = load_mu_from_ckpt(TAU2_CKPT[gamma_f])
        for tau_f in TAU_CHAIN_UP[1:]:  # skip τ=2
            existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                        f"{gamma_tag(gamma_f)}_{tau_tag(tau_f)}_mp50.json")
            if os.path.exists(existing):
                with open(existing) as f:
                    d = json.load(f)
                mu_warm, p_old = load_mu_from_ckpt(existing)
                print(f"  Skipping γ={gamma_f}, τ={tau_f} (exists, "
                      f"1-R²={d.get('1-R2_weighted', 'n/a')})", flush=True)
                continue
            print(f"\n=== γ={gamma_f}, τ={tau_f} (warm) ===", flush=True)
            t0 = time.time()
            try:
                (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
                 p_grid_np, mu_arr) = solve_one_warm(gamma_f, tau_f,
                                                        mu_warm, p_old,
                                                        u_grid_np, u_grid_mp)
                save_ckpt(gamma_f, tau_f, mu_conv, u_grid_np, p_grid_np,
                              F_max_v, F_med_v, history, one_R2, slope)
                mu_warm = mu_arr; p_old = p_grid_np
                print(f"  γ={gamma_f}, τ={tau_f}: 1-R²={one_R2:.6e}, "
                      f"slope={slope:.4f} (t={time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  γ={gamma_f}, τ={tau_f} FAILED: {e}")

    print(f"\nTotal: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
