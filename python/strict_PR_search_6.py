"""Strict PR search batch 6: maximum-pertubation seed scan.

Combines every perturbation strategy I have for finding non-FR fixed
points. For each (γ, τ) config:

   • 80 random Gaussian seeds at σ ∈ {0.05, 0.1, 0.2, 0.3, 0.5}
   • 12 ansatz seeds (logit P = α·T*) for α ∈ {0.02..1.5}
   • 6 logit-compression seeds (compress FR by factor κ ∈ {0.1..1.5})
   • 6 reflection seeds (P → 1−P with perturbation)
   • 3 slice-permuted seeds (swap axes of FR P)

Total: ~107 seeds per config × 4 configs = ~430 strict-Newton solves.

Each solve uses lighter Newton (maxiters=8, lgmres_maxiter=50) to keep
total runtime tractable. Strict acceptance: Finf < 1e-12 AND
1-R² > 0.01 → save tensor.
"""
import time
import pickle
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


CONFIGS = [
    (11, 3.4, 3.0, "γ=3 τ=3.4"),
    (11, 2.0, 0.5, "γ=0.5 τ=2"),
    (11, 5.0, 1.0, "γ=1 τ=5"),
    (11, 5.0, 0.5, "γ=0.5 τ=5"),
]
UMAX = 2.0
TARGET = 1e-12

NEWTON_ITERS = 8
LGMRES_MAXITER = 50


def ansatz(u, tau, alpha):
    G = u.shape[0]
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                P[i, j, l] = 1.0 / (1.0 + np.exp(
                    -alpha * tau * (u[i] + u[j] + u[l])))
    return np.clip(P, 1e-9, 1 - 1e-9)


def random_seed(rng, G, sigma, base=0.5):
    return np.clip(base + sigma * rng.standard_normal((G, G, G)),
                     1e-9, 1 - 1e-9)


def logit_compress(P_FR, kappa):
    eps = 1e-9
    L = np.log(np.clip(P_FR, eps, 1-eps) / (1 - np.clip(P_FR, eps, 1-eps)))
    L_mean = L.mean()
    L_new = L_mean + kappa * (L - L_mean)
    return np.clip(1.0 / (1.0 + np.exp(-L_new)), eps, 1 - eps)


def reflect(P, sigma, rng):
    """Replace P with 1 - P plus noise."""
    Pn = 1.0 - P
    Pn += sigma * rng.standard_normal(Pn.shape)
    return np.clip(Pn, 1e-9, 1 - 1e-9)


def axis_permute(P, perm):
    """Permute axes (e.g. swap (0,1,2) → (1,0,2))."""
    return np.transpose(P, perm)


for G, tau, gamma, label in CONFIGS:
    print(f"\n=== {label} (G={G}) ===", flush=True)
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([tau, tau, tau])
    gammas = np.array([gamma, gamma, gamma])
    Ws = np.array([1.0, 1.0, 1.0])
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))

    # Get FR baseline
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=200, abstol=TARGET, alpha=0.3)
    P_FR = res_p["P_star"]
    print(f"  FR baseline Finf={float(np.abs(res_p['residual']).max()):.2e}",
          flush=True)

    # Build seed bank
    rng = np.random.default_rng(2026)
    seeds = []
    # 80 random
    for sigma in (0.05, 0.1, 0.2, 0.3, 0.5):
        for k in range(16):
            seeds.append((f"rand σ={sigma} #{k}", random_seed(rng, G, sigma)))
    # 12 ansatz
    for alpha in (0.02, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.70,
                    0.85, 1.0, 1.2, 1.5):
        seeds.append((f"ansatz α={alpha}", ansatz(u, tau, alpha)))
    # 6 logit-compression of FR
    for kappa in (0.1, 0.3, 0.5, 0.7, 0.9, 1.5):
        seeds.append((f"compress κ={kappa}", logit_compress(P_FR, kappa)))
    # 6 reflection
    rng2 = np.random.default_rng(7)
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        seeds.append((f"reflect σ={sigma}", reflect(P_FR, sigma, rng2)))
    # 3 axis permutations of FR
    for perm in [(1, 0, 2), (2, 1, 0), (0, 2, 1)]:
        seeds.append((f"perm {perm}", axis_permute(P_FR, perm)))

    print(f"  seed bank: {len(seeds)} seeds", flush=True)
    found = 0
    t_total = time.time()
    for s_name, P_init in seeds:
        t0 = time.time()
        try:
            res = pj.solve_newton(
                G, taus, gammas, umax=UMAX,
                P_init=P_init, maxiters=NEWTON_ITERS, abstol=TARGET,
                lgmres_tol=1e-12, lgmres_maxiter=LGMRES_MAXITER)
            P = res["P_star"]; Finf = res["best_Finf"]
        except Exception:
            continue
        if Finf > TARGET:
            continue
        one_r2 = rh.one_minus_R2(P, u, taus)
        p = float(P[ir, jr, lr])
        mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
        if one_r2 > 0.01:
            print(f"  PR FOUND! seed='{s_name}' p={p:.4f} "
                  f"1-R²={one_r2:.4e} PR-gap={mu[0]-mu[1]:+.4f} "
                  f"Finf={Finf:.2e} ({time.time()-t0:.1f}s)",
                  flush=True)
            fn = (f"/home/user/REZN/python/PR_seed_g{gamma}_t{tau}_b6_"
                   f"{found}.pkl")
            with open(fn, "wb") as f:
                pickle.dump({"P": P, "taus": taus, "gammas": gammas,
                              "G": G, "umax": UMAX,
                              "Finf": Finf, "1-R²": one_r2,
                              "seed": s_name}, f)
            print(f"    saved to {fn}", flush=True)
            found += 1
    print(f"  {label}: {found} PR fixed points found  "
          f"({time.time()-t_total:.0f}s, {len(seeds)} seeds)",
          flush=True)
