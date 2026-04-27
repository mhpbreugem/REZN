"""Strict PR search batch 3: perturbation seeds.

For each (γ, τ) that gave FR, try MANY non-FR seeds (random, scaled,
ansatz-α with various values). If any seed lands in a different
basin, we have PR.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


GAMMA_TAU = [
    (3.0, 3.4, 11),
    (0.5, 2.0, 11),
    (1.0, 5.0, 11),
    (3.0, 3.0, 11),
    (0.5, 5.0, 11),
]
UMAX = 2.0
TARGET = 1e-12
N_RAND = 30                                # random seeds per config


def random_seed(rng, G, sigma):
    return np.clip(0.5 + sigma * rng.standard_normal((G, G, G)),
                     1e-9, 1 - 1e-9)


def ansatz_seed(u, tau, alpha):
    G = u.shape[0]
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                P[i, j, l] = 1.0 / (1.0 + np.exp(
                    -alpha * tau * (u[i] + u[j] + u[l])))
    return np.clip(P, 1e-9, 1 - 1e-9)


for gamma_v, tau_v, G in GAMMA_TAU:
    print(f"\n=== γ={gamma_v}, τ={tau_v}, G={G} ===", flush=True)
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([tau_v, tau_v, tau_v])
    gammas = np.array([gamma_v, gamma_v, gamma_v])

    rng = np.random.default_rng(42)
    seeds = {"NL":           rh._nolearning_price(u, taus, gammas,
                                                      np.array([1.0,1.0,1.0]))}
    # ansatz seeds
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80):
        seeds[f"ansatz α={alpha}"] = ansatz_seed(u, tau_v, alpha)
    # random seeds
    for k in range(N_RAND):
        seeds[f"rand σ=0.2 #{k}"] = random_seed(rng, G, 0.2)

    found_pr = False
    for seed_name, P_init in seeds.items():
        t0 = time.time()
        try:
            res_n = pj.solve_newton(
                G, taus, gammas, umax=UMAX,
                P_init=P_init, maxiters=15, abstol=TARGET,
                lgmres_tol=1e-12, lgmres_maxiter=200)
            P = res_n["P_star"]; Finf = res_n["best_Finf"]
        except Exception:
            continue
        if Finf > TARGET:
            continue
        one_r2 = rh.one_minus_R2(P, u, taus)
        ir = int(np.argmin(np.abs(u - 1.0)))
        jr = int(np.argmin(np.abs(u + 1.0)))
        lr = int(np.argmin(np.abs(u - 1.0)))
        p = float(P[ir, jr, lr])
        if one_r2 > 0.01:
            print(f"  PR FOUND! seed='{seed_name}' p={p:.4f} "
                   f"1-R²={one_r2:.4e} Finf={Finf:.2e} "
                   f"({time.time()-t0:.1f}s)", flush=True)
            found_pr = True
            # save the tensor
            import pickle
            fn = f"/home/user/REZN/python/PR_seed_g{gamma_v}_t{tau_v}.pkl"
            with open(fn, "wb") as f:
                pickle.dump({"P": P, "taus": taus, "gammas": gammas,
                              "Finf": Finf, "1-R²": one_r2, "G": G,
                              "umax": UMAX}, f)
            print(f"    saved to {fn}", flush=True)
    if not found_pr:
        print(f"  no seed reached PR — only FR fixed point at this config",
              flush=True)
