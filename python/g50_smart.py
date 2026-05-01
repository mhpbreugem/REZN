"""G=50 smart polish: warm-start chain from G=17 (mp200) up to G=50 in steps,
float64 polish at each, then single mp200 NK iter at G=50.

Updates every 10 seconds.
"""
import time, json, warnings, os
import numpy as np
import mpmath
from mpmath import mp, mpf

warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    Lam as Lam_f64, init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64,
)
from gap_reparam import pava_p_only, pava_u_only

RESULTS_DIR = "results/full_ree"

# Wait for G=17 mp200 ckpt
src = f"{RESULTS_DIR}/posterior_v3_G17_mp200.json"
print(f"Waiting for {src}...", flush=True)
last_print = time.time()
while not os.path.exists(src):
    if time.time() - last_print > 60:
        print(f"  Still waiting...", flush=True)
        last_print = time.time()
    time.sleep(5)
print(f"Found G=17 mp200, loading...", flush=True)


def pava_2d_f64(mu): return pava_u_only(pava_p_only(mu))


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, n, na, alpha,
                  status_every=1000):
    """Picard with status updates."""
    mu_sum = np.zeros_like(mu); n_collected = 0
    t0 = time.time()
    last_status = t0
    for it in range(n):
        cand, active, _ = phi_step_f64(mu, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
        cand = pava_2d_f64(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS_F64, 1 - EPS_F64)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
        now = time.time()
        if now - last_status >= 10:
            cand2, act2, _ = phi_step_f64(mu, u_grid, p_grid,
                                              p_lo, p_hi, 2.0, 0.5)
            r_max = float(np.max(np.abs(cand2 - mu)[act2]))
            print(f"    [picard α={alpha}] iter {it+1}/{n}, "
                  f"max={r_max:.3e}, t={now-t0:.0f}s", flush=True)
            last_status = now
    return pava_2d_f64(mu_sum / max(n_collected, 1))


def measure(mu, u_grid, p_grid, p_lo, p_hi):
    cand, active, _ = phi_step_f64(mu, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()),
        "med": float(np.median(F[active])),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def interp_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new); G_old = len(u_old)
    out = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        u_c = np.clip(u_target, u_old[0], u_old[-1])
        ra = np.searchsorted(u_old, u_c)
        rb = max(ra - 1, 0); ra = min(ra, G_old - 1)
        w = (u_c - u_old[rb]) / (u_old[ra] - u_old[rb]) if ra != rb else 1.0
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_b = np.clip(p_target, p_old[rb, 0], p_old[rb, -1])
            mu_b = np.interp(p_b, p_old[rb, :], mu_old[rb, :])
            p_a = np.clip(p_target, p_old[ra, 0], p_old[ra, -1])
            mu_a = np.interp(p_a, p_old[ra, :], mu_old[ra, :])
            out[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(out, EPS_F64, 1 - EPS_F64)


# Load G=17 mp200 (cast to float64)
with open(src) as f:
    state17 = json.load(f)
G17 = 17
mu_17 = np.array([[float(state17["mu_strings"][i][j]) for j in range(G17)]
                      for i in range(G17)])
u_17 = np.array([float(s) for s in state17["u_grid"]])
p_17 = np.array([[float(s) for s in row] for row in state17["p_grid"]])
print(f"G=17 mp200 loaded: F_max={state17['F_max'][:30]}, "
      f"shape={mu_17.shape}", flush=True)

# Continuation through G=18, 22, 28, 35, 42, 50
G_LADDER = [22, 28, 35, 42, 50]
prev_mu, prev_u, prev_p = mu_17, u_17, p_17

for G in G_LADDER:
    print(f"\n{'='*60}\nG = {G}\n{'='*60}", flush=True)
    t_G = time.time()
    UMAX = 4.0
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid_f64(u_grid, 2.0, 0.5, G, trim=0.05)
    mu_warm = interp_to_grid(prev_mu, prev_u, prev_p, u_grid, p_grid)
    mu_warm = pava_2d_f64(mu_warm)
    d0 = measure(mu_warm, u_grid, p_grid, p_lo, p_hi)
    print(f"  Warm interp: max={d0['max']:.3e}, med={d0['med']:.3e}",
          flush=True)

    # 3 Picard rounds
    print(f"  Picard α=0.05 (8000 iters)...", flush=True)
    mu = picard_round(mu_warm, u_grid, p_grid, p_lo, p_hi, 8000, 4000, 0.05)
    print(f"  Picard α=0.01 (5000 iters)...", flush=True)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.01)
    print(f"  Picard α=0.003 (5000 iters)...", flush=True)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.003)
    d_p = measure(mu, u_grid, p_grid, p_lo, p_hi)
    print(f"  After picard: max={d_p['max']:.3e}, med={d_p['med']:.3e}, "
          f"u/p={d_p['u_viol']}/{d_p['p_viol']}, t={time.time()-t_G:.0f}s",
          flush=True)

    # Save float64 result
    np.savez(f"{RESULTS_DIR}/posterior_v3_G{G}_f64_polish.npz",
             mu=mu, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
    prev_mu, prev_u, prev_p = mu, u_grid, p_grid
    print(f"  Saved G={G} float64 polish", flush=True)

print("\n=== G_LADDER UP TO 50 DONE (float64) ===", flush=True)
