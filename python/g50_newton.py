"""G=50 Newton-Krylov directly. No trim, no picard. Warm from G=17 mp200.

Uses scipy newton_krylov in float64 (mp300 at G=50 is too slow: ~21hr/iter).
2500-unknown system. Jacobian-vector via FD inside Krylov. Target 1e-12.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
sys.path.insert(0, "python")
from posterior_method_v3 import (
    init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64, measure_R2,
)
from gap_reparam import pava_p_only, pava_u_only

RESULTS_DIR = "results/full_ree"
G = 50; UMAX = 4.0
TAU = 2.0; GAMMA = 0.5


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def F_phi(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS_F64, 1 - EPS_F64)
    cand, active, _ = phi_step_f64(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def interp_to_grid(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new); G_old = len(u_old)
    out = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u = u_new[i_new]; u_c = np.clip(u, u_old[0], u_old[-1])
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


print(f"Loading G=17 mp200 warm...", flush=True)
with open(f"{RESULTS_DIR}/posterior_v3_G17_mp200.json") as f:
    state17 = json.load(f)
G17 = 17
mu_17 = np.array([[float(state17["mu_strings"][i][j]) for j in range(G17)]
                      for i in range(G17)])
u_17 = np.array([float(s) for s in state17["u_grid"]])
p_17 = np.array([[float(s) for s in row] for row in state17["p_grid"]])

print(f"Building G={G} no-trim grid (TRIM=0)...", flush=True)
u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid_f64(u_grid, TAU, GAMMA, G, trim=0.0)
mu_warm = interp_to_grid(mu_17, u_17, p_17, u_grid, p_grid)
mu_warm = pava_2d(mu_warm)
cand0, active0, _ = phi_step_f64(mu_warm, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F_init = float(np.max(np.abs(cand0 - mu_warm)[active0]))
print(f"Warm: max={F_init:.3e}", flush=True)

# Newton-Krylov directly
print(f"\nStarting scipy newton_krylov at G={G}, no trim...", flush=True)
t0 = time.time()
n_calls = [0]
def F_wrapped(x):
    n_calls[0] += 1
    if n_calls[0] % 25 == 0:
        elapsed = time.time() - t0
        F_curr = F_phi(x, mu_warm.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
        r_max = float(np.max(np.abs(F_curr)))
        print(f"  call {n_calls[0]}, F_max={r_max:.3e}, t={elapsed:.0f}s",
              flush=True)
    return F_phi(x, mu_warm.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)

try:
    sol = newton_krylov(F_wrapped, mu_warm.ravel(),
                          f_tol=1e-12, maxiter=80,
                          method="lgmres", verbose=False)
    mu_final = np.clip(sol.reshape(mu_warm.shape), EPS_F64, 1 - EPS_F64)
    nk_status = "ok"
    print(f"NK converged after {n_calls[0]} calls", flush=True)
except NoConvergence as e:
    mu_final = np.clip(e.args[0].reshape(mu_warm.shape), EPS_F64, 1 - EPS_F64)
    nk_status = "noconv"
    print(f"NK no-conv after {n_calls[0]} calls", flush=True)

elapsed = time.time() - t0
cand, active, _ = phi_step_f64(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
F = np.abs(cand - mu_final)
res_max = float(F[active].max())
res_med = float(np.median(F[active]))
n_u = int((np.diff(mu_final, axis=0) < 0).sum())
n_p = int((np.diff(mu_final, axis=1) < 0).sum())
print(f"\n=== G=50 Newton FINAL ===")
print(f"  max={res_max:.3e}, med={res_med:.3e}, u/p={n_u}/{n_p}, "
      f"NK={nk_status}, t={elapsed:.0f}s", flush=True)

print(f"\nMeasuring 1-R²...", flush=True)
r2, slope, n = measure_R2(mu_final, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  1-R²={r2:.6e}, slope={slope:.6f}", flush=True)

np.savez(f"{RESULTS_DIR}/posterior_v3_G50_newton.npz",
         mu=mu_final, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
with open(f"{RESULTS_DIR}/posterior_v3_G50_newton_summary.json", "w") as f:
    json.dump({"G": G, "tau": TAU, "gamma": GAMMA, "trim": 0.0,
                "max_residual": res_max, "med_residual": res_med,
                "u_viol": n_u, "p_viol": n_p,
                "nk_status": nk_status,
                "1-R^2": r2, "slope": slope, "elapsed_s": elapsed}, f, indent=2)
print(f"Saved", flush=True)
