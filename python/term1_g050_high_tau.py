"""Terminal 1: γ=0.5 high-τ chain from τ=3 (skipping bailed τ=4).

Per PARALLEL_SOLVER.md.
Chain: τ=3.0 (good ckpt, max~e-30) → τ=5 → τ=7 → τ=10 → τ=15
(skip τ=4 since it's structurally stuck at boundary).
"""
import time, json, warnings, os.path
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 50

import sys
sys.path.insert(0, "python")
from task3_tau_sweep import (
    G, UMAX, TRIM, H_FD, TARGET, gamma_tag, tau_tag,
    nk_solve, weighted_R2_grid, to_floats, save_ckpt,
    init_p_grid_f64, phi_step_f64, EPS_F64, pava_2d_f64,
)
from task3_tau_ladder import load_mu_from_ckpt, interp_mu_to_new_pgrid

RESULTS_DIR = "results/full_ree"
GAMMA_F = 0.5

# τ chain skipping τ=4
TAU_CHAIN = [3.0, 5.0, 7.0, 10.0, 15.0]
# τ=3 ckpt (already converged)
SEED_CKPT = (f"{RESULTS_DIR}/posterior_v3_G20_umax5_g050_t0300_mp50.json")


def solve_one_warm(gamma_f, tau_f, mu_warm_arr, p_old_arr, u_grid_np,
                       u_grid_mp):
    gamma_mp = mpf(str(gamma_f))
    tau_mp = mpf(str(tau_f))
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, tau_f, gamma_f,
                                                       G, trim=TRIM)
    p_grid_mp = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo_mp = [mpf(str(x)) for x in p_lo_np]
    p_hi_mp = [mpf(str(x)) for x in p_hi_np]

    mu_f = interp_mu_to_new_pgrid(mu_warm_arr, p_old_arr, p_grid_np)
    mu_f = pava_2d_f64(mu_f)

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

    # Load τ=3 seed (always good warm)
    mu_seed, p_seed = load_mu_from_ckpt(SEED_CKPT)
    print(f"Seed: {SEED_CKPT}", flush=True)

    # Use τ=3 as the persistent fallback warm-start (don't chain through bad)
    mu_warm = mu_seed; p_old = p_seed

    for tau_f in TAU_CHAIN[1:]:  # skip τ=3 (seed)
        existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                    f"{gamma_tag(GAMMA_F)}_{tau_tag(tau_f)}_mp50.json")
        if os.path.exists(existing):
            with open(existing) as f:
                d = json.load(f)
            print(f"  Skipping γ=0.5, τ={tau_f} (exists, "
                  f"1-R²={d.get('1-R2_weighted','n/a')})", flush=True)
            continue
        print(f"\n=== γ=0.5, τ={tau_f} (warm from τ=3) ===", flush=True)
        t0 = time.time()
        try:
            (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
             p_grid_np, mu_arr) = solve_one_warm(GAMMA_F, tau_f, mu_warm,
                                                    p_old, u_grid_np, u_grid_mp)
            save_ckpt(GAMMA_F, tau_f, mu_conv, u_grid_np, p_grid_np,
                          F_max_v, F_med_v, history, one_R2, slope)
            print(f"  γ=0.5, τ={tau_f}: 1-R²={one_R2:.6e}, "
                  f"slope={slope:.4f} (t={time.time()-t0:.0f}s)")
            # Update warm only if reasonably converged (else keep τ=3 warm)
            if F_max_v < mpf("0.1"):
                mu_warm = mu_arr; p_old = p_grid_np
                print(f"  → warm-start updated for next τ", flush=True)
            else:
                print(f"  → keeping τ=3 warm-start (this τ bailed)", flush=True)
        except Exception as e:
            print(f"  γ=0.5, τ={tau_f} FAILED: {e}")

    print(f"\nTotal: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
