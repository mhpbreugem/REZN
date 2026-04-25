"""Quick G=11 diagnostic.

Why isn't Picard converging at γ=500 G=11 cold? And is Newton actually
fast enough at the workhorse parameters?
"""
import time
import numpy as np
import rezn_pchip as rp
import pchip_jacobian as pj


def main():
    G = 11; UMAX = 2.5
    u = np.linspace(-UMAX, UMAX, G)
    Ws = np.array([1.0, 1.0, 1.0])

    print("=== Picard cold at γ=500 with α=1.0, longer ===", flush=True)
    t0 = time.time()
    res = rp.solve_picard_pchip(G, np.array([3.0]*3), np.array([500.0]*3),
                                  umax=UMAX, maxiters=2000, abstol=1e-10, alpha=1.0)
    print(f"  Picard 2000: {time.time()-t0:.1f}s, "
          f"Finf={float(np.abs(res['residual']).max()):.3e}, "
          f"history last 5: {[f'{h:.3e}' for h in res['history'][-5:]]}", flush=True)
    print(f"  history first 10: {[f'{h:.3e}' for h in res['history'][:10]]}", flush=True)
    P500 = res["P_star"]

    print("\n=== Single Newton step from γ=500 Picard, lgmres maxiter=30 ===", flush=True)
    t0 = time.time()
    res = pj.solve_newton_analytic(G, np.array([3.0]*3), np.array([500.0]*3),
                                     umax=UMAX, P_init=P500, maxiters=3,
                                     abstol=1e-10, lgmres_tol=1e-6,
                                     lgmres_maxiter=30, verbose=True)
    print(f"  Newton: {time.time()-t0:.1f}s, best Finf={res['best_Finf']:.3e}",
          flush=True)


if __name__ == "__main__":
    main()
