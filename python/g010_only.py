"""Just γ=0.1 cold-start (Task 2 final run). Imports machinery from
task2_gamma_sweep.
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 50

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    init_p_grid as init_p_grid_f64, phi_step as phi_step_f64, EPS as EPS_F64,
)
from gap_reparam import pava_p_only, pava_u_only

# Reuse all the helpers from task2
from task2_gamma_sweep import (
    G, UMAX, TRIM, TARGET, MAX_ITERS, gamma_tag,
    Lam_mp, logit_mp, crra_demand_mp, f_v_mp, interp_mp, phi_step_mp,
    F_mu, F_max, F_med, lm_step, nk_solve, save_ckpt, to_floats,
    measure_R2_float,
)

RESULTS_DIR = "results/full_ree"
GAMMA_F = 0.1
TAU_F = 2.0


def pava_2d_f64(mu): return pava_u_only(pava_p_only(mu))


def main():
    t_start = time.time()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, TAU_F, GAMMA_F,
                                                       G, trim=TRIM)
    p_grid_mp = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo_mp = [mpf(str(x)) for x in p_lo_np]
    p_hi_mp = [mpf(str(x)) for x in p_hi_np]
    u_grid_mp = [mpf(str(x)) for x in u_grid_np]
    gamma_mp = mpf(str(GAMMA_F))

    print(f"\n=== γ={GAMMA_F} (cold-start) ===")

    # Cold start
    Lam = lambda z: 1.0/(1.0+np.exp(-z)) if z>=0 else np.exp(z)/(1+np.exp(z))
    mu_f = np.zeros((G, G))
    for i, u in enumerate(u_grid_np):
        mu_f[i, :] = Lam(TAU_F * u)
    mu_f = pava_2d_f64(mu_f)

    # Polish
    print(f"  Float64 picard polish (γ={GAMMA_F})...", flush=True)
    last_status = time.time()
    for round_idx, (n_iter, n_avg, alpha) in enumerate(
            [(2000, 1000, 0.005), (2000, 1000, 0.002),
             (3000, 1500, 0.001)]):
        mu_sum = np.zeros_like(mu_f); n_collected = 0
        for it in range(n_iter):
            cand, active, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                              p_lo_np, p_hi_np,
                                              TAU_F, GAMMA_F)
            cand = pava_2d_f64(cand)
            mu_f = alpha * cand + (1 - alpha) * mu_f
            mu_f = np.clip(mu_f, EPS_F64, 1 - EPS_F64)
            if it >= n_iter - n_avg:
                mu_sum += mu_f; n_collected += 1
            if time.time() - last_status > 30:
                cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                                  p_lo_np, p_hi_np,
                                                  TAU_F, GAMMA_F)
                r = float(np.max(np.abs(cand2 - mu_f)[act2]))
                print(f"    polish r{round_idx+1} α={alpha} "
                      f"it {it+1}/{n_iter}: max={r:.3e}", flush=True)
                last_status = time.time()
        mu_f = pava_2d_f64(mu_sum / max(n_collected, 1))

    cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                      p_lo_np, p_hi_np, TAU_F, GAMMA_F)
    F_f64 = float(np.max(np.abs(cand2 - mu_f)[act2]))
    print(f"  Float64 polish done: max={F_f64:.3e}", flush=True)

    # Cast to mp50
    mu_init = [[mpf(str(mu_f[i, j])) for j in range(G)] for i in range(G)]

    # mp50 LM
    tag = f"γ={GAMMA_F}"
    mu_conv, F_max_v, F_med_v, history = nk_solve(
        mu_init, u_grid_mp, p_grid_mp, p_lo_mp, p_hi_mp, gamma_mp, tag)

    # Measure unweighted (the post-processor will compute weighted)
    mu_arr = to_floats(mu_conv)
    one_R2, slope, n = measure_R2_float(mu_arr, u_grid_np, p_grid_np,
                                            GAMMA_F, TAU_F)
    print(f"  γ={GAMMA_F}: 1-R² (unweighted) = {one_R2:.6e}, slope={slope:.6f}")

    save_ckpt(gamma_mp, mu_conv, u_grid_np, p_grid_np,
                  F_max_v, F_med_v, history)
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
