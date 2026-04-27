"""Test Picard at γ=3, τ=3, G=11 (the published PR branch regime)."""
import time, numpy as np
import rezn_pchip as rp
import rezn_het as rh

G, UMAX = 11, 2.0
configs = [(3.0, 3.0), (3.0, 2.0), (3.0, 4.0),
            (1.0, 3.0), (0.5, 3.0)]

for tau, gamma in configs:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([tau, tau, tau])
    gammas = np.array([gamma, gamma, gamma])

    t0 = time.time()
    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                  maxiters=2000, abstol=1e-7, alpha=0.3)
    Finf = float(np.abs(res["residual"]).max())
    P = res["P_star"]
    one_r2 = rh.one_minus_R2(P, u, taus)
    i = int(np.argmin(np.abs(u - 1.0)))
    j = int(np.argmin(np.abs(u + 1.0)))
    l = int(np.argmin(np.abs(u - 1.0)))
    mu = rh.posteriors_at(i, j, l, float(P[i, j, l]), P, u, taus)
    print(f"τ={tau:.1f} γ={gamma:.1f}  iters={len(res['history']):4d}  "
          f"Finf={Finf:.2e}  1-R²={one_r2:.3e}  "
          f"PR-gap={mu[0]-mu[1]:+.4f}  ({time.time()-t0:.0f}s)",
          flush=True)
