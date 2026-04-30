"""Find which cells oscillate at G=12, gamma=0.5."""
import numpy as np
from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, picard_anderson,
)

Gu = Gp = 12
tau = 2.0; gamma = 0.5
u_grid = np.linspace(-4, 4, Gu)
p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, Gp)

# Seed
mu = np.zeros((Gu, Gp))
for i, u in enumerate(u_grid):
    mu[i, :] = Lam(tau * u)

# Run 200 iters then dump 4 consecutive iterates
mu, _, _ = picard_anderson(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma,
                            damping=0.05, anderson=0, max_iter=200, tol=1e-12,
                            progress=False)
print("After warmup, residual=...")

# Now do 4 manual phi steps to inspect
mus = [mu.copy()]
for k in range(4):
    cand, active, ncross = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = cand - mu
    mu = 0.95*mu + 0.05*cand
    mu = np.clip(mu, 1e-10, 1-1e-10)
    mus.append(mu.copy())
    # Top 5 oscillating cells
    flat_idx = np.argsort(np.abs(F.ravel()))[::-1][:5]
    print(f"--- step {k+1} ---")
    for fi in flat_idx:
        i, j = fi // Gp, fi % Gp
        print(f"  cell ({i:2d},{j:2d}) u={u_grid[i]:+.2f} p={p_grid[i,j]:.3f} "
              f"mu_old={mu.ravel()[fi]:.4f} cand={cand.ravel()[fi]:.4f} "
              f"F={F.ravel()[fi]:+.4f} ncross={ncross[i,j]}")

# Check 2-cycle: |mu_n+2 - mu_n| should be small at FP, large in cycle
diff_2step = np.abs(mus[3] - mus[1])
print(f"\nmax |mu_n+2 - mu_n| = {diff_2step.max():.4f} (small=>FP, large=>cycle)")
print(f"max |mu_n+1 - mu_n| = {np.abs(mus[2]-mus[1]).max():.4f}")
