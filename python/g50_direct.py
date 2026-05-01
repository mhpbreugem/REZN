"""G=50 direct: warm from G=17 mp200, polish in float64.
Status updates every 10s.
"""
import time, json, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    Lam as Lam_f64, init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64, measure_R2,
)
from gap_reparam import pava_p_only, pava_u_only

RESULTS_DIR = "results/full_ree"
G = 50; UMAX = 4.0


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def interp_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new); G_old = len(u_old)
    out = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u = u_new[i_new]
        u_c = np.clip(u, u_old[0], u_old[-1])
        ra = np.searchsorted(u_old, u_c)
        rb = max(ra - 1, 0); ra = min(ra, G_old - 1)
        w = (u_c - u_old[rb]) / (u_old[ra] - u_old[rb]) if ra != rb else 1.0
        for j_new in range(p_new.shape[1]):
            p = p_new[i_new, j_new]
            p_b = np.clip(p, p_old[rb, 0], p_old[rb, -1])
            mu_b = np.interp(p_b, p_old[rb, :], mu_old[rb, :])
            p_a = np.clip(p, p_old[ra, 0], p_old[ra, -1])
            mu_a = np.interp(p_a, p_old[ra, :], mu_old[ra, :])
            out[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(out, EPS_F64, 1 - EPS_F64)


def picard(mu, u_grid, p_grid, p_lo, p_hi, n, na, alpha,
            label="", status_every=10):
    mu_sum = np.zeros_like(mu); n_collected = 0
    t0 = time.time()
    last_status = t0
    for it in range(n):
        cand, active, _ = phi_step_f64(mu, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS_F64, 1 - EPS_F64)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
        now = time.time()
        if now - last_status >= status_every:
            cand2, act2, _ = phi_step_f64(mu, u_grid, p_grid,
                                              p_lo, p_hi, 2.0, 0.5)
            r_max = float(np.max(np.abs(cand2 - mu)[act2]))
            r_med = float(np.median(np.abs(cand2 - mu)[act2]))
            print(f"  [{label}] iter {it+1}/{n}, max={r_max:.3e}, "
                  f"med={r_med:.3e}, t={now-t0:.0f}s", flush=True)
            last_status = now
    return pava_2d(mu_sum / max(n_collected, 1))


# Load G=17 mp200
print(f"Loading G=17 mp200...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G17_mp200.json") as f:
    state17 = json.load(f)
G17 = 17
mu_17 = np.array([[float(state17["mu_strings"][i][j]) for j in range(G17)]
                      for i in range(G17)])
u_17 = np.array([float(s) for s in state17["u_grid"]])
p_17 = np.array([[float(s) for s in row] for row in state17["p_grid"]])

# G=50 grid
print(f"Building G=50 grid...", flush=True)
u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid_f64(u_grid, 2.0, 0.5, G, trim=0.05)
mu_warm = interp_to_grid(mu_17, u_17, p_17, u_grid, p_grid)
mu_warm = pava_2d(mu_warm)
cand0, act0, _ = phi_step_f64(mu_warm, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
print(f"Warm interp G=17→G=50: max={float(np.max(np.abs(cand0-mu_warm)[act0])):.3e}",
      flush=True)

t_start = time.time()
print(f"\nPicard polish chain (8000 + 5000 + 8000)...", flush=True)
mu = picard(mu_warm, u_grid, p_grid, p_lo, p_hi, 8000, 4000, 0.05,
              label="α=0.05")
mu = picard(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.01, label="α=0.01")
mu = picard(mu, u_grid, p_grid, p_lo, p_hi, 8000, 4000, 0.003, label="α=0.003")

cand, active, _ = phi_step_f64(mu, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
F = np.abs(cand - mu)
res_max = float(F[active].max())
res_med = float(np.median(F[active]))
n_u = int((np.diff(mu, axis=0) < 0).sum())
n_p = int((np.diff(mu, axis=1) < 0).sum())
print(f"\n=== G=50 FINAL (float64) ===")
print(f"  max={res_max:.3e}, med={res_med:.3e}, u/p={n_u}/{n_p}")
print(f"  total t={time.time()-t_start:.0f}s", flush=True)

# Measure 1-R²
print(f"\nMeasuring 1-R² at G={G}...", flush=True)
r2, slope, n = measure_R2(mu, u_grid, p_grid, p_lo, p_hi, 2.0, 0.5)
print(f"  1-R²={r2:.6e}, slope={slope:.6f}, samples={n}", flush=True)

np.savez(f"{RESULTS_DIR}/posterior_v3_G50_f64.npz",
         mu=mu, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
with open(f"{RESULTS_DIR}/posterior_v3_G50_summary.json", "w") as f:
    json.dump({"G": G, "tau": 2.0, "gamma": 0.5,
                "max_residual": res_max,
                "med_residual": res_med,
                "u_viol": n_u, "p_viol": n_p,
                "1-R^2": r2, "slope": slope, "samples": n}, f, indent=2)
print(f"\nSaved {RESULTS_DIR}/posterior_v3_G50_f64.npz")
