"""Single-config precision test at G=15, homogeneous τ=γ=(3,3,3).

At G=9 PCHIP with Anderson plateaus at ≈9e-9 for CRRA homo. G=15 should
cut that floor substantially if it's piecewise-linear-on-rows residual
(we're now using PCHIP, so the floor should really be driven by the
parametric error of PCHIP given finite-node data — smaller with more
nodes). Also measures how much slower G=15 is per Picard step.
"""
import sys
import time
import numpy as np
import rezn_pchip as rp

G       = 15
UMAX    = 2.0
TAU     = 3.0
GAMMA   = 3.0
ABSTOL  = 1e-13

print(f"\n=== G={G} PCHIP Picard  τ=({TAU},{TAU},{TAU}) γ=({GAMMA},{GAMMA},{GAMMA}) ===")
sys.stdout.flush()

print("numba JIT warmup at small G…")
sys.stdout.flush()
_ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

# Cold-start Picard α=1.0
print(f"\nPicard α=1.0, abstol={ABSTOL:.0e}, maxiters=3000")
sys.stdout.flush()
t0 = time.time()
res_picard = rp.solve_picard_pchip(
    G, (TAU, TAU, TAU), (GAMMA, GAMMA, GAMMA),
    umax=UMAX, maxiters=3000, abstol=ABSTOL, alpha=1.0)
dt_picard = time.time() - t0
PhiI_p = res_picard["history"][-1] if res_picard["history"] else float("inf")
Finf_p = float(np.abs(res_picard["residual"]).max())
print(f"  iters={len(res_picard['history'])}  time={dt_picard:.1f}s  "
      f"PhiI={PhiI_p:.3e}  Finf={Finf_p:.3e}  conv={res_picard['converged']}")
print(f"  last 8 PhiI: {[f'{x:.2e}' for x in res_picard['history'][-8:]]}")
sys.stdout.flush()

# Anderson window m=8
print(f"\nAnderson m=8, abstol={ABSTOL:.0e}, maxiters=1500")
sys.stdout.flush()
t0 = time.time()
res_and = rp.solve_anderson_pchip(
    G, (TAU, TAU, TAU), (GAMMA, GAMMA, GAMMA),
    umax=UMAX, maxiters=1500, abstol=ABSTOL, m_window=8, damping=1.0,
    P_init=res_picard["P_star"])   # warm-start from Picard result
dt_and = time.time() - t0
PhiI_a = res_and["history"][-1] if res_and["history"] else float("inf")
Finf_a = float(np.abs(res_and["residual"]).max())
print(f"  iters={len(res_and['history'])}  time={dt_and:.1f}s  "
      f"PhiI={PhiI_a:.3e}  Finf={Finf_a:.3e}  conv={res_and['converged']}")
print(f"  last 8 PhiI: {[f'{x:.2e}' for x in res_and['history'][-8:]]}")
sys.stdout.flush()

# Report at (1, -1, 1)
u = np.linspace(-UMAX, UMAX, G)
idx = lambda x: int(np.argmin(np.abs(u - x)))
i, j, l = idx(1.0), idx(-1.0), idx(1.0)
P = res_and["P_star"] if Finf_a <= Finf_p else res_picard["P_star"]
import rezn_het as rh
mu = rh.posteriors_at(i, j, l, float(P[i,j,l]), P, u, np.asarray((TAU,TAU,TAU)))
print(f"\nEquilibrium at cell (u1,u2,u3) ≈ ({u[i]:.3f}, {u[j]:.3f}, {u[l]:.3f}):")
print(f"  p*       = {float(P[i,j,l]):.12f}")
print(f"  μ1       = {mu[0]:.12f}")
print(f"  μ2       = {mu[1]:.12f}")
print(f"  μ3       = {mu[2]:.12f}")
print(f"  μ1 - μ2  = {mu[0]-mu[1]:.6e}   (PR gap)")

# 1-R² comparisons
y = np.log(P / (1 - P)).reshape(-1)
T = np.array([TAU*(u[i]+u[j]+u[l]) for i in range(G) for j in range(G) for l in range(G)])
yc = y - y.mean(); Tc = T - T.mean()
R2 = (yc*Tc).sum()**2 / ((yc*yc).sum() * (Tc*Tc).sum())
print(f"\n1-R² (vs T = τ·Σu, equal-weight CARA-FR reference) = {1-R2:.6e}")
