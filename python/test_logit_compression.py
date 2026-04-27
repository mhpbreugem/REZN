"""Try to land in the PR basin by compressing logit(P) at the
FR-converged seed.

Idea: PR fixed point has p_(1,-1,1)=0.632 (less extreme than FR's
~0.797). Logit-compression scales logit(P) by α ∈ (0, 1), which
shrinks all prices toward 0.5. Picard from this seed may land in
the PR basin instead of snapping back to FR.

Test at γ=3, τ=3.4, G=11 with several compression factors.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

G, UMAX, TAU, GAMMA = 11, 2.0, 3.4, 3.0

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])
Ws = np.array([1.0, 1.0, 1.0])

# Step 1: get FR-converged P
print("Solving FR converge from no-learning ...")
res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                              maxiters=600, abstol=1e-7, alpha=0.3)
P_FR = res["P_star"]
finf_fr = float(np.abs(res["residual"]).max())
print(f"  FR Finf={finf_fr:.2e}", flush=True)

ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))

def report(name, P):
    p = float(P[ir, jr, lr])
    finf = float(np.abs(P - rp._phi_map_pchip(P, u, taus, gammas, Ws)).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    print(f"  [{name:25s}]  p={p:.4f}  1-R²={one_r2:.4e}  "
          f"PR-gap={mu[0]-mu[1]:+.4f}  Finf={finf:.2e}",
          flush=True)
    return one_r2, p, mu

# Step 2: try various compressed seeds
print("\nLogit-compression seeds:")
eps = 1e-9
L_FR = np.log(np.clip(P_FR, eps, 1-eps) / (1 - np.clip(P_FR, eps, 1-eps)))
L_FR_mean = L_FR.mean()
report("FR (baseline)", P_FR)

for alpha in (0.9, 0.7, 0.5, 0.3, 0.1):
    L_seed = L_FR_mean + alpha * (L_FR - L_FR_mean)
    P_seed = np.clip(1.0 / (1.0 + np.exp(-L_seed)), eps, 1-eps)
    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                  maxiters=600, abstol=1e-7, alpha=0.3,
                                  P_init=P_seed)
    P = res["P_star"]
    report(f"compress α={alpha}", P)

# Step 3: also try mixtures with no-learning
print("\nNo-learning + small kick:")
P_NL = rh._nolearning_price(u, taus, gammas, Ws)
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    P_seed = w * P_NL + (1 - w) * P_FR
    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                  maxiters=600, abstol=1e-7, alpha=0.3,
                                  P_init=np.clip(P_seed, eps, 1-eps))
    P = res["P_star"]
    report(f"NL·{w} + FR·{1-w:.1f}", P)
