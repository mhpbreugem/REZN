"""Quick test: does Picard at G=11 stay on PR branch from NL seed?"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

G = 11
UMAX = 2.0
TAU = 2.0
GAMMA = 0.5

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])

t0 = time.time()
res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                              maxiters=2000, abstol=1e-7, alpha=0.3)
P = res["P_star"]
Finf = float(np.abs(res["residual"]).max())
oneR2 = rh.one_minus_R2(P, u, taus)

i = int(np.argmin(np.abs(u - 1.0)))
j = int(np.argmin(np.abs(u + 1.0)))
l = int(np.argmin(np.abs(u - 1.0)))
mu = rh.posteriors_at(i, j, l, float(P[i, j, l]), P, u, taus)
print(f"Picard α=0.3 G={G} γ={GAMMA} τ={TAU}")
print(f"  iters={len(res['history'])}  Finf={Finf:.3e}  "
      f"1-R²={oneR2:.4e}")
print(f"  μ at (1,-1,1) = {tuple(round(m, 4) for m in mu)}")
print(f"  p_(1,-1,1) = {float(P[i, j, l]):.4f}")
print(f"  PR-gap μ_1 - μ_2 = {mu[0] - mu[1]:+.4f}")
print(f"  ({time.time()-t0:.0f}s)")
