"""REE solver using convex_contour for the contour integration step.

State variable: P[G,G,G] price tensor.
Each iteration:
  For each (i,j,l) and each agent k=1,2,3:
    - Take slice where agent k's signal is u_i (or u_j or u_l)
    - Use compute_posterior_convex to get A_0, A_1 → μ_k
  Market-clear: solve sum_k x_k(μ_k, p_new) = 0 → P_new[i,j,l]
Iterate to fixed point.
"""
import json, time, warnings
import numpy as np
from scipy.optimize import brentq

from posterior_method_v3 import Lam, crra_demand_vec, EPS
from convex_contour import compute_posterior_convex, signal_density

warnings.filterwarnings("ignore", category=RuntimeWarning)


def initial_no_learning_prices(u_grid, tau, gamma):
    """P[i,j,l] = market-clearing price under no-learning beliefs μ_k = Λ(τu_k)."""
    G = len(u_grid)
    P = np.zeros((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                mus = [Lam(tau * u) for u in (u_grid[i], u_grid[j], u_grid[l])]
                def Z(p):
                    return sum(crra_demand_vec(np.array([m]),
                                                  np.array([p]), gamma)[0]
                               for m in mus)
                try:
                    P[i, j, l] = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-12)
                except ValueError:
                    P[i, j, l] = 0.5
    return P


def phi_step_convex(P, u_grid, tau, gamma):
    """One Φ-step on the price tensor.

    For each (i,j,l):
      - For agent k=1 (signal u_i): slice P[i, :, :] is P(u_i fixed, u_j, u_l)
      - For agent k=2 (signal u_j): slice P[:, j, :] is P(u_i, u_j fixed, u_l)
      - For agent k=3 (signal u_l): slice P[:, :, l] is P(u_i, u_j, u_l fixed)
      - Each agent uses convex_contour at their own slice + observed price
      - Market-clear with the three posteriors
    """
    G = len(u_grid)
    P_new = np.zeros_like(P)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p_obs = P[i, j, l]
                # Agent 1: own signal u_i, sees price p_obs
                slice_1 = P[i, :, :]   # shape (G, G), price as function of (u_j, u_l)
                A0_1, A1_1 = compute_posterior_convex(
                    slice_1, u_grid, u_grid[i], p_obs, tau)
                f1_1 = signal_density(u_grid[i], 1, tau)
                f0_1 = signal_density(u_grid[i], 0, tau)
                denom1 = f0_1 * A0_1 + f1_1 * A1_1
                mu1 = f1_1 * A1_1 / denom1 if denom1 > 0 else 0.5

                # Agent 2: own signal u_j
                slice_2 = P[:, j, :]
                A0_2, A1_2 = compute_posterior_convex(
                    slice_2, u_grid, u_grid[j], p_obs, tau)
                f1_2 = signal_density(u_grid[j], 1, tau)
                f0_2 = signal_density(u_grid[j], 0, tau)
                denom2 = f0_2 * A0_2 + f1_2 * A1_2
                mu2 = f1_2 * A1_2 / denom2 if denom2 > 0 else 0.5

                # Agent 3: own signal u_l
                slice_3 = P[:, :, l]
                A0_3, A1_3 = compute_posterior_convex(
                    slice_3, u_grid, u_grid[l], p_obs, tau)
                f1_3 = signal_density(u_grid[l], 1, tau)
                f0_3 = signal_density(u_grid[l], 0, tau)
                denom3 = f0_3 * A0_3 + f1_3 * A1_3
                mu3 = f1_3 * A1_3 / denom3 if denom3 > 0 else 0.5

                # Market-clear with these three μ
                mus = [np.clip(mu1, EPS, 1-EPS), np.clip(mu2, EPS, 1-EPS),
                        np.clip(mu3, EPS, 1-EPS)]
                def Z(p):
                    return sum(crra_demand_vec(np.array([m]),
                                                  np.array([p]), gamma)[0]
                               for m in mus)
                try:
                    P_new[i, j, l] = brentq(Z, 1e-6, 1 - 1e-6, xtol=1e-10)
                except ValueError:
                    P_new[i, j, l] = p_obs   # keep old
    return P_new


def measure_R2_from_P(P, u_grid, tau):
    """Compute 1-R² from the price tensor (no μ needed, just regression)."""
    G = len(u_grid)
    Y, X, W = [], [], []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = P[i, j, l]
                T = tau * (u_grid[i] + u_grid[j] + u_grid[l])
                f1_prod = (signal_density(u_grid[i], 1, tau)
                            * signal_density(u_grid[j], 1, tau)
                            * signal_density(u_grid[l], 1, tau))
                f0_prod = (signal_density(u_grid[i], 0, tau)
                            * signal_density(u_grid[j], 0, tau)
                            * signal_density(u_grid[l], 0, tau))
                w = 0.5 * (f1_prod + f0_prod)
                Y.append(np.log(p / (1 - p))); X.append(T); W.append(float(w))
    Y = np.array(Y); X = np.array(X); W = np.array(W)
    W = W / W.sum()
    Yb = (W * Y).sum(); Xb = (W * X).sum()
    cov = (W * (Y - Yb) * (X - Xb)).sum()
    vy = (W * (Y - Yb)**2).sum()
    vx = (W * (X - Xb)**2).sum()
    R2 = cov**2 / (vy * vx) if vy * vx > 0 else 0.0
    return 1.0 - R2, cov / vx if vx > 0 else 0.0


if __name__ == "__main__":
    import sys
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    TAU = 2.0; GAMMA = 0.5
    UMAX = 4.0
    DAMP = 0.3
    MAX_ITER = 30

    u_grid = np.linspace(-UMAX, UMAX, G)
    print(f"REE convex-contour solver: G={G}, τ={TAU}, γ={GAMMA}", flush=True)

    print("Initializing P from no-learning prices...", flush=True)
    t0 = time.time()
    P = initial_no_learning_prices(u_grid, TAU, GAMMA)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    r2_init, slope_init = measure_R2_from_P(P, u_grid, TAU)
    print(f"  No-learning: 1-R²={r2_init:.4e}, slope={slope_init:.4f}",
          flush=True)

    history = [{"iter": 0, "1-R2": float(r2_init), "slope": float(slope_init),
                "max_change": 0.0}]
    for it in range(1, MAX_ITER + 1):
        t_iter = time.time()
        P_new = phi_step_convex(P, u_grid, TAU, GAMMA)
        change = float(np.max(np.abs(P_new - P)))
        # Damped update
        P = (1 - DAMP) * P + DAMP * P_new
        r2, slope = measure_R2_from_P(P, u_grid, TAU)
        elapsed = time.time() - t_iter
        history.append({"iter": it, "1-R2": float(r2),
                          "slope": float(slope),
                          "max_change": change})
        print(f"  iter {it:>2}: max|ΔP|={change:.3e}, "
              f"1-R²={r2:.4e}, slope={slope:.4f}, t={elapsed:.0f}s",
              flush=True)
        if change < 1e-10:
            print("Converged.", flush=True)
            break

    out = f"results/full_ree/ree_convex_G{G}.npz"
    np.savez(out, P=P, u_grid=u_grid, tau=TAU, gamma=GAMMA)
    print(f"\nSaved {out}", flush=True)
    with open(f"results/full_ree/ree_convex_G{G}_history.json", "w") as f:
        json.dump({"params": {"G": G, "tau": TAU, "gamma": GAMMA,
                               "damping": DAMP},
                    "history": history}, f, indent=2)
