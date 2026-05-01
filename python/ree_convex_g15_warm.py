"""REE convex-contour solver, warm-started from G=15 posterior_v3 strict.

Convert μ tensor to P tensor (market clearing at each (u_1,u_2,u_3)),
then run convex Picard with very small damping (0.03) to avoid basin jumps.
Reports max|ΔP| each iter.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import brentq
from posterior_method_v3 import Lam, crra_demand_vec, EPS
from convex_contour import compute_posterior_convex, signal_density
from ree_convex_solver import phi_step_convex, measure_R2_from_P

warnings.filterwarnings("ignore", category=RuntimeWarning)

G = 15
TAU = 2.0; GAMMA = 0.5
UMAX = 4.0
DAMP = 0.03
MAX_ITER = 60

# Load G=15 strict μ tensor
ck = np.load(f"results/full_ree/posterior_v3_strict_G15.npz")
mu_v3 = ck["mu"]; u_grid_v3 = ck["u_grid"]; p_grid_v3 = ck["p_grid"]


def mu_at(u, p):
    if u <= u_grid_v3[0]: idx = 0; w = 0.0
    elif u >= u_grid_v3[-1]: idx = len(u_grid_v3) - 1; w = 1.0
    else:
        ia = np.searchsorted(u_grid_v3, u); ib = ia - 1
        w = (u - u_grid_v3[ib]) / (u_grid_v3[ia] - u_grid_v3[ib])
        p_b = np.clip(p, p_grid_v3[ib, 0], p_grid_v3[ib, -1])
        m_b = np.interp(p_b, p_grid_v3[ib, :], mu_v3[ib, :])
        p_a = np.clip(p, p_grid_v3[ia, 0], p_grid_v3[ia, -1])
        m_a = np.interp(p_a, p_grid_v3[ia, :], mu_v3[ia, :])
        return (1 - w) * m_b + w * m_a
    p_c = np.clip(p, p_grid_v3[idx, 0], p_grid_v3[idx, -1])
    return float(np.interp(p_c, p_grid_v3[idx, :], mu_v3[idx, :]))


u_grid = np.linspace(-UMAX, UMAX, G)
print(f"REE convex from G=15 strict warm, damping={DAMP}", flush=True)

# Compute initial P from μ tensor: market-clear at each (u_i, u_j, u_l)
print("Computing initial P from μ tensor (market clearing at each triple)...",
      flush=True)
t0 = time.time()
P = np.zeros((G, G, G))
for i in range(G):
    for j in range(G):
        for l in range(G):
            u1, u2, u3 = u_grid[i], u_grid[j], u_grid[l]
            def Z(p):
                return (
                    crra_demand_vec(np.array([mu_at(u1, p)]),
                                      np.array([p]), GAMMA)[0]
                    + crra_demand_vec(np.array([mu_at(u2, p)]),
                                        np.array([p]), GAMMA)[0]
                    + crra_demand_vec(np.array([mu_at(u3, p)]),
                                        np.array([p]), GAMMA)[0])
            try:
                P[i, j, l] = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
            except ValueError:
                P[i, j, l] = 0.5
print(f"  done in {time.time()-t0:.1f}s", flush=True)

r2_init, slope_init = measure_R2_from_P(P, u_grid, TAU)
print(f"\nSeed P (from posterior_v3 μ): 1-R²={r2_init:.4e}, "
      f"slope={slope_init:.4f}", flush=True)

history = [{"iter": 0, "1-R2": float(r2_init), "slope": float(slope_init),
            "max_change": 0.0, "norm_change": 0.0}]
for it in range(1, MAX_ITER + 1):
    t_iter = time.time()
    P_new = phi_step_convex(P, u_grid, TAU, GAMMA)
    diff = P_new - P
    max_change = float(np.max(np.abs(diff)))
    l2_change = float(np.sqrt(np.mean(diff**2)))
    P = (1 - DAMP) * P + DAMP * P_new
    r2, slope = measure_R2_from_P(P, u_grid, TAU)
    elapsed = time.time() - t_iter
    history.append({"iter": it, "1-R2": float(r2),
                      "slope": float(slope),
                      "max_change": max_change,
                      "norm_change": l2_change})
    print(f"  iter {it:>2}: max|ΔP|={max_change:.3e}, "
          f"||ΔP||₂={l2_change:.3e}, "
          f"1-R²={r2:.4e}, slope={slope:.4f}, t={elapsed:.0f}s",
          flush=True)
    if max_change < 1e-10:
        print("Converged.", flush=True)
        break

out = f"results/full_ree/ree_convex_G{G}_from_v3.npz"
np.savez(out, P=P, u_grid=u_grid, tau=TAU, gamma=GAMMA)
print(f"\nSaved {out}", flush=True)
with open(f"results/full_ree/ree_convex_G{G}_from_v3_history.json", "w") as f:
    json.dump({"params": {"G": G, "tau": TAU, "gamma": GAMMA,
                            "damping": DAMP, "seed": "posterior_v3_G15_strict"},
                "history": history}, f, indent=2)
