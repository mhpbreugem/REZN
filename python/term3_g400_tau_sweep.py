"""Terminal 3 REDO: γ=4.0 τ-sweep at G=20 UMAX=5 trim=0.05 mp50.

Per SESSION3_SUMMARY.md: previous Terminal 3 used WRONG solver, needs redo.
γ=4.0 is closest to CARA → should converge FASTEST.

Chain from γ=4.0 τ=2 ckpt (already converged, in g400_mp50.json):
  Down: 2 → 1.5 → 1.0 → 0.8 → 0.5 → 0.3
  Up:   2 → 3 → 4 → 5 → 7 → 10 → 15

Save each to: results/full_ree/posterior_v3_G20_umax5_g400_tXXXX_mp50.json
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
from term1_g050_high_tau import solve_one_warm

RESULTS_DIR = "results/full_ree"
GAMMA_F = 4.0

# τ chain
TAU_CHAIN_DOWN = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3]
TAU_CHAIN_UP = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
SEED_CKPT = f"{RESULTS_DIR}/posterior_v3_G20_umax5_g400_mp50.json"


def main():
    t_start = time.time()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    u_grid_mp = [mpf(str(x)) for x in u_grid_np]

    # Load τ=2 seed
    mu_seed, p_seed = load_mu_from_ckpt(SEED_CKPT)
    print(f"Seed: {SEED_CKPT}", flush=True)
    print(f"=== γ={GAMMA_F} τ-ladder ===", flush=True)

    # Walk DOWN
    print(f"\n--- Walking τ DOWN from 2 ---", flush=True)
    mu_warm = mu_seed; p_old = p_seed
    for tau_f in TAU_CHAIN_DOWN[1:]:
        existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                    f"{gamma_tag(GAMMA_F)}_{tau_tag(tau_f)}_mp50.json")
        if os.path.exists(existing):
            with open(existing) as f:
                d = json.load(f)
            mu_warm, p_old = load_mu_from_ckpt(existing)
            print(f"  Skipping γ={GAMMA_F}, τ={tau_f} (exists, "
                  f"1-R²={d.get('1-R2_weighted','n/a')})", flush=True)
            continue
        print(f"\n=== γ={GAMMA_F}, τ={tau_f} (warm) ===", flush=True)
        t0 = time.time()
        try:
            (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
             p_grid_np, mu_arr) = solve_one_warm(GAMMA_F, tau_f, mu_warm,
                                                    p_old, u_grid_np,
                                                    u_grid_mp)
            save_ckpt(GAMMA_F, tau_f, mu_conv, u_grid_np, p_grid_np,
                          F_max_v, F_med_v, history, one_R2, slope)
            print(f"  γ={GAMMA_F}, τ={tau_f}: 1-R²={one_R2:.6e}, "
                  f"slope={slope:.4f} (t={time.time()-t0:.0f}s)")
            if F_max_v < mpf("0.1"):
                mu_warm = mu_arr; p_old = p_grid_np
        except Exception as e:
            print(f"  γ={GAMMA_F}, τ={tau_f} FAILED: {e}")

    # Walk UP
    print(f"\n--- Walking τ UP from 2 ---", flush=True)
    mu_warm, p_old = load_mu_from_ckpt(SEED_CKPT)
    for tau_f in TAU_CHAIN_UP[1:]:
        existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                    f"{gamma_tag(GAMMA_F)}_{tau_tag(tau_f)}_mp50.json")
        if os.path.exists(existing):
            with open(existing) as f:
                d = json.load(f)
            mu_warm, p_old = load_mu_from_ckpt(existing)
            print(f"  Skipping γ={GAMMA_F}, τ={tau_f} (exists, "
                  f"1-R²={d.get('1-R2_weighted','n/a')})", flush=True)
            continue
        print(f"\n=== γ={GAMMA_F}, τ={tau_f} (warm) ===", flush=True)
        t0 = time.time()
        try:
            (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
             p_grid_np, mu_arr) = solve_one_warm(GAMMA_F, tau_f, mu_warm,
                                                    p_old, u_grid_np,
                                                    u_grid_mp)
            save_ckpt(GAMMA_F, tau_f, mu_conv, u_grid_np, p_grid_np,
                          F_max_v, F_med_v, history, one_R2, slope)
            print(f"  γ={GAMMA_F}, τ={tau_f}: 1-R²={one_R2:.6e}, "
                  f"slope={slope:.4f} (t={time.time()-t0:.0f}s)")
            if F_max_v < mpf("0.1"):
                mu_warm = mu_arr; p_old = p_grid_np
        except Exception as e:
            print(f"  γ={GAMMA_F}, τ={tau_f} FAILED: {e}")

    print(f"\nTotal: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
