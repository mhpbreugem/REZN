"""Single-step φ diagnostic for posterior_method_v3.

Inspect what one Φ_μ application does to the no-learning seed at a small grid.
Print at a central cell:
  - the μ-column extracted at the cell's price
  - the demand array d(u; p₀) and its monotonicity
  - the contour roots u₃*(u₂)
  - the resulting A₁, A₀ and posterior μ_new
Compare to mu^0 = Λ(τu) (no-learning).
"""
import numpy as np
import math
from posterior_method_v3 import (
    Lam, logit, f_v, crra_demand_vec,
    init_p_grid, extract_mu_col, phi_step,
)

Gu = Gp = 8
umax = 3.0
tau = 2.0
gamma = 0.5

u_grid = np.linspace(-umax, umax, Gu)
p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, Gp)
print("u_grid =", u_grid)
print("p_grid (per-row min/max):")
for i in range(Gu):
    print(f"  i={i}, u={u_grid[i]:+.3f}: p in [{p_grid[i,0]:.4f}, {p_grid[i,-1]:.4f}]")

# Seed mu^0 = Lam(tau u) (no learning, constant in p)
mu0 = np.zeros((Gu, Gp))
for i, u in enumerate(u_grid):
    mu0[i, :] = Lam(tau * u)

print("\nmu0 (constant in p):")
for i in range(Gu):
    print(f"  i={i} u={u_grid[i]:+.3f} mu={mu0[i,0]:.4f}")

# Pick the central cell
i_mid = Gu // 2
j_mid = Gp // 2
p0 = p_grid[i_mid, j_mid]
u_i = u_grid[i_mid]
print(f"\nCentral cell i={i_mid} j={j_mid}: u_i={u_i:+.3f}, p0={p0:.4f}")

mu_col = extract_mu_col(mu0, p_grid, p0, u_grid, tau, p_lo, p_hi)
print("mu_col(u; p0):", mu_col)

d = crra_demand_vec(mu_col, np.full_like(mu_col, p0), gamma)
print("d(u; p0):", d)
print("d monotone increasing?", np.all(np.diff(d) > 0))

D_i = -d[i_mid]
targets = D_i - d
print(f"D_i = -d[i_mid] = {D_i:.4f}")
print("targets:", targets)

if d[-1] > d[0]:
    u3_star = np.interp(targets, d, u_grid,
                        left=u_grid[0] - 1, right=u_grid[-1] + 1)
else:
    u3_star = np.interp(targets, d[::-1], u_grid[::-1],
                        left=u_grid[-1] + 1, right=u_grid[0] - 1)
valid = (u3_star >= u_grid[0]) & (u3_star <= u_grid[-1])
print("u3_star:", u3_star)
print("valid:", valid)
print(f"n_valid = {valid.sum()}")

f1_grid = f_v(u_grid, 1, tau)
f0_grid = f_v(u_grid, 0, tau)
f1_root = f_v(u3_star[valid], 1, tau)
f0_root = f_v(u3_star[valid], 0, tau)
A1 = float(np.sum(f1_grid[valid] * f1_root))
A0 = float(np.sum(f0_grid[valid] * f0_root))
print(f"A0 = {A0:.6f}")
print(f"A1 = {A1:.6f}")
mu_new_cell = f1_grid[i_mid] * A1 / (f0_grid[i_mid] * A0 + f1_grid[i_mid] * A1)
print(f"mu_new[i_mid, j_mid] = {mu_new_cell:.6f}")
print(f"mu0[i_mid, j_mid]    = {mu0[i_mid, j_mid]:.6f}")
print(f"Lam(tau*u_i) [no-learning ref] = {Lam(tau*u_i):.6f}")

# Full phi step
mu_new, active, ncross = phi_step(mu0, u_grid, p_grid, p_lo, p_hi, tau, gamma)
print(f"\nFull phi: active = {active.sum()}/{Gu*Gp}")
print(f"max|mu_new - mu0| over active = {np.max(np.abs(mu_new - mu0)[active]):.4f}")

# Inspect mu_new at i_mid for all p
print(f"\nmu_new[i_mid, :] (post-phi) at u_i={u_i:+.3f}:")
for j in range(Gp):
    a = "*" if active[i_mid, j] else " "
    print(f"  j={j} p={p_grid[i_mid,j]:.4f} mu_new={mu_new[i_mid,j]:.4f} {a}")
