"""Verify whether the documented backward-snap PR values are just
the no-learning equilibrium, or a genuinely-different fixed point.

Compare:
  reference (from pchip_G11_backward_snap.csv at τ=3.4, γ=3):
    1-R² = 0.073, p_(1,-1,1) = 0.632, μ = (0.645, 0.633, 0.645),
    Finf = 3.785e-4  (this was "conv=1" under whatever F_TOL was used)

  no-learning at the same (γ, τ): just market clearing with private
  priors, no contour iteration at all — should give the same metrics
  if the "PR branch" is really just no-learning.
"""
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

G, UMAX, TAU, GAMMA = 11, 2.0, 3.4, 3.0

u = np.linspace(-UMAX, UMAX, G)
taus   = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])
Ws     = np.array([1.0, 1.0, 1.0])

# 1) No-learning
P0 = rh._nolearning_price(u, taus, gammas, Ws)

ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))

p_nl = float(P0[ir, jr, lr])
mu_nl = rh.posteriors_at(ir, jr, lr, p_nl, P0, u, taus)
oneR2_nl = rh.one_minus_R2(P0, u, taus)

# Compute the no-learning Φ-residual: Finf = ||P0 - Φ(P0)||∞
Phi0 = rp._phi_map_pchip(P0, u, taus, gammas, Ws)
finf_nl = float(np.abs(P0 - Phi0).max())

print(f"=== γ={GAMMA}, τ={TAU}, G={G} ===")
print(f"\nReference (backward_snap CSV):")
print(f"  1-R² = 0.073, p = 0.6320, μ = (0.645, 0.633, 0.645), Finf = 3.785e-4")
print(f"\nNo-learning (no contour iteration):")
print(f"  1-R² = {oneR2_nl:.4f}")
print(f"  p_(1,-1,1) = {p_nl:.4f}")
print(f"  μ at (1,-1,1) = {tuple(round(m,4) for m in mu_nl)}")
print(f"  Finf = ||P_0 - Φ(P_0)||_∞ = {finf_nl:.3e}")
print(f"\n=> If these match → 'PR branch' was just no-learning all along.")
print(f"   If they don't match → real different fixed point exists.")
