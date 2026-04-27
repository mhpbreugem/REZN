"""Walk the Picard trajectory from no-learning toward FR at γ=3, τ=3.4
and report (1-R², p_(1,-1,1), Finf) at every iteration.

Hypothesis: HANDOFF's claimed strong-PR fixed point is just a transient
at iter k where Finf temporarily ≈ 3.8e-4 while p_(1,-1,1) is still
≈ 0.632 — i.e., a Picard waypoint, not a true fixed point.
"""
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

G, UMAX, TAU, GAMMA = 11, 2.0, 3.4, 3.0
u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])
Ws = np.array([1.0, 1.0, 1.0])
ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))

# Walk Picard from NL, snapshot at decreasing diffs
P = rh._nolearning_price(u, taus, gammas, Ws)
Phi = rp._phi_map_pchip(P, u, taus, gammas, Ws)
diff0 = float(np.abs(P - Phi).max())
p0 = float(P[ir, jr, lr])
mu0 = rh.posteriors_at(ir, jr, lr, p0, P, u, taus)
oneR2_0 = rh.one_minus_R2(P, u, taus)
print(f"iter   p_(1,-1,1)   1-R²       PR-gap   Finf       diff")
print(f"{'NL':>4s}  {p0:.4f}      {oneR2_0:.4e}  {mu0[0]-mu0[1]:+.4f}  ---       {diff0:.3e}")

ALPHA = 0.3
last_diff = diff0
N = 1500
SNAP_LEVELS = [1e-1, 1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7]
shown = set()
for it in range(N):
    Pn = rp._phi_map_pchip(P, u, taus, gammas, Ws)
    diff = float(np.abs(Pn - P).max())
    P = (1 - ALPHA) * P + ALPHA * Pn
    P = np.clip(P, 1e-9, 1 - 1e-9)
    # Find the first SNAP_LEVEL that's just below current diff
    for snap in SNAP_LEVELS:
        if last_diff > snap >= diff and snap not in shown:
            shown.add(snap)
            p = float(P[ir, jr, lr])
            mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
            one_r2 = rh.one_minus_R2(P, u, taus)
            Phi = rp._phi_map_pchip(P, u, taus, gammas, Ws)
            finf = float(np.abs(P - Phi).max())
            print(f"{it+1:>4d}  {p:.4f}      {one_r2:.4e}  {mu[0]-mu[1]:+.4f}  "
                  f"{finf:.3e}  {diff:.3e}")
            break
    last_diff = diff

# final
p = float(P[ir, jr, lr])
mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
one_r2 = rh.one_minus_R2(P, u, taus)
Phi = rp._phi_map_pchip(P, u, taus, gammas, Ws)
finf = float(np.abs(P - Phi).max())
print(f"\nfinal {N:>3d}  {p:.4f}      {one_r2:.4e}  {mu[0]-mu[1]:+.4f}  {finf:.3e}")
print()
print("HANDOFF PR target:    p_(1,-1,1)=0.6322  1-R²=7.33e-2  μ_1-μ_2=0.012  Finf=3.79e-4")
