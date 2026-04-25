"""Verify analytic Newton works at G=11 from the γ=500 CARA seed."""
import time
import numpy as np
import rezn_pchip as rp
import pchip_jacobian as pj
import sys


def main():
    G = 11; UMAX = 2.5
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([3.0, 3.0, 3.0])
    Ws = np.array([1.0, 1.0, 1.0])

    print("[1/3] G=11 γ=500 seed via Picard...", flush=True)
    t0 = time.time()
    res = rp.solve_picard_pchip(G, taus, np.array([500.0]*3), umax=UMAX,
                                  maxiters=300, abstol=1e-9, alpha=1.0)
    P500 = res["P_star"]
    print(f"  Picard γ=500: {time.time()-t0:.1f}s, "
          f"Finf={float(np.abs(res['residual']).max()):.3e}",
          f"converged={res['converged']}", flush=True)

    print("[2/3] G=11 γ=50 via analytic Newton from γ=500 warm start...", flush=True)
    t0 = time.time()
    res = pj.solve_newton_analytic(G, taus, np.array([50.0]*3), umax=UMAX,
                                     P_init=P500, maxiters=15, abstol=1e-10,
                                     lgmres_tol=1e-9, lgmres_maxiter=120,
                                     verbose=True)
    P50 = res["P_star"]
    print(f"  Newton γ=50: {time.time()-t0:.1f}s, best Finf={res['best_Finf']:.3e}, "
          f"iters={len(res['history'])}, converged={res['converged']}",
          flush=True)

    print("[3/3] G=11 γ=3 via analytic Newton from γ=50 warm start...", flush=True)
    t0 = time.time()
    res = pj.solve_newton_analytic(G, taus, np.array([3.0]*3), umax=UMAX,
                                     P_init=P50, maxiters=20, abstol=1e-10,
                                     lgmres_tol=1e-9, lgmres_maxiter=120,
                                     verbose=True)
    print(f"  Newton γ=3: {time.time()-t0:.1f}s, best Finf={res['best_Finf']:.3e}, "
          f"iters={len(res['history'])}, converged={res['converged']}",
          flush=True)


if __name__ == "__main__":
    main()
