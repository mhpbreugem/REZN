"""Search for any fixed point with 1-R² > 0.01 across a wide parameter
sweep. If we find ONE, we have a starting point for the figures.

Each config: cold Picard (no-learning seed), then Anderson polish.
Report (1-R², PR-gap, p, Finf). Convergence threshold: Finf < 1e-4.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

CONFIGS = [
    # G,  τ_vec,                 γ_vec,                    label
    (11, (2.0, 2.0, 2.0),       (0.1, 0.1, 0.1),         "homog γ=0.1, τ=2"),
    (11, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "homog γ=0.5, τ=2"),
    (11, (5.0, 5.0, 5.0),       (0.5, 0.5, 0.5),         "homog γ=0.5, τ=5"),
    (11, (10.0, 10.0, 10.0),    (3.0, 3.0, 3.0),         "homog γ=3, τ=10"),
    (11, (2.0, 2.0, 2.0),       (5.0, 3.0, 1.0),         "het γ=(5,3,1), τ=2"),
    (11, (2.0, 2.0, 2.0),       (1.0, 3.0, 5.0),         "het γ=(1,3,5), τ=2"),
    (11, (1.0, 3.0, 5.0),       (3.0, 3.0, 3.0),         "homog γ=3, het τ=(1,3,5)"),
    (11, (1.0, 3.0, 10.0),      (1.0, 3.0, 10.0),        "opposed (1,3,10)"),
    (11, (10.0, 3.0, 1.0),      (1.0, 3.0, 10.0),        "aligned"),
    ( 7, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=7 homog"),
    (15, (2.0, 2.0, 2.0),       (0.5, 0.5, 0.5),         "G=15 homog γ=0.5"),
    (15, (3.0, 3.0, 3.0),       (3.0, 3.0, 3.0),         "G=15 γ=3 τ=3"),
]

UMAX = 2.0
ABSTOL_PICARD = 1e-7
ABSTOL_ANDERSON = 1e-7

print(f"{'cfg':30s}  G   {'p_(1,-1,1)':>10s}  {'1-R²':>10s}  "
      f"{'PR-gap':>9s}  {'Finf':>10s}  {'time':>6s}  status")

for G, taus_t, gammas_t, label in CONFIGS:
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array(taus_t)
    gammas = np.array(gammas_t)

    t0 = time.time()
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=800, abstol=ABSTOL_PICARD,
                                    alpha=0.3)
    res_a = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                      maxiters=200, abstol=ABSTOL_ANDERSON,
                                      m_window=8, P_init=res_p["P_star"])
    Finf_p = float(np.abs(res_p["residual"]).max())
    Finf_a = float(np.abs(res_a["residual"]).max())
    if Finf_a < Finf_p:
        P = res_a["P_star"]; Finf = Finf_a
    else:
        P = res_p["P_star"]; Finf = Finf_p
    one_r2 = rh.one_minus_R2(P, u, taus)
    ir = int(np.argmin(np.abs(u - 1.0)))
    jr = int(np.argmin(np.abs(u + 1.0)))
    lr = int(np.argmin(np.abs(u - 1.0)))
    p = float(P[ir, jr, lr])
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr_gap = mu[0] - mu[1]
    converged = Finf < 1e-4
    status = "PR!" if (one_r2 > 0.01 and converged) else \
             ("FR" if converged else "diverged")
    print(f"{label:30s}  {G:2d}  {p:10.4f}  {one_r2:10.3e}  "
          f"{pr_gap:+9.4f}  {Finf:10.2e}  {time.time()-t0:6.1f}s  {status}",
          flush=True)
