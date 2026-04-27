"""Analytic-Jacobian Newton from compressed FR seeds at γ=3, τ=3.4.

Earlier the Picard solver collapsed every seed to FR. NK behaves
differently — per HANDOFF, NK is what found the PR branch
historically. Try it with longer iter count, tighter lgmres, and
both compressed-FR and randomized-perturbation seeds.
"""
import time
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj

G, UMAX, TAU, GAMMA = 11, 2.0, 3.4, 3.0
MAXITERS = 40
LGMRES_TOL = 1e-10
LGMRES_MAXITER = 200
ABSTOL = 1e-9

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])
Ws = np.array([1.0, 1.0, 1.0])

ir = int(np.argmin(np.abs(u - 1.0)))
jr = int(np.argmin(np.abs(u + 1.0)))
lr = int(np.argmin(np.abs(u - 1.0)))


def report(name, P, t):
    p = float(P[ir, jr, lr])
    finf = float(np.abs(P - rp._phi_map_pchip(P, u, taus, gammas, Ws)).max())
    one_r2 = rh.one_minus_R2(P, u, taus)
    mu = rh.posteriors_at(ir, jr, lr, p, P, u, taus)
    pr = "PR!" if (one_r2 > 0.02 and finf < 1e-3) else "FR/?"
    print(f"  [{name:30s}]  p={p:.4f}  1-R²={one_r2:.4e}  "
          f"PR-gap={mu[0]-mu[1]:+.4f}  Finf={finf:.2e}  [{pr}]  "
          f"({t:.1f}s)", flush=True)


def run_seed(name, P_init):
    t0 = time.time()
    try:
        res = pj.solve_newton(
            G, taus, gammas, umax=UMAX,
            P_init=P_init, maxiters=MAXITERS, abstol=ABSTOL,
            lgmres_tol=LGMRES_TOL, lgmres_maxiter=LGMRES_MAXITER)
        report(name, res["P_star"], time.time() - t0)
    except Exception as ex:
        print(f"  [{name:30s}]  FAIL: {ex}", flush=True)


# Build seeds
print(f"=== analytic Newton at γ={GAMMA}, τ={TAU}, G={G} ===\n")

# Seed 1: FR-converged baseline
print("Solving FR baseline ...", flush=True)
res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                              maxiters=600, abstol=1e-7, alpha=0.3)
P_FR = res["P_star"]
print(f"  FR Finf={float(np.abs(res['residual']).max()):.2e}\n", flush=True)

eps = 1e-9
L_FR = np.log(np.clip(P_FR, eps, 1-eps) / (1 - np.clip(P_FR, eps, 1-eps)))
L_mean = L_FR.mean()

P_NL = rh._nolearning_price(u, taus, gammas, Ws)

print("Newton seeds:")
run_seed("FR baseline",            P_FR)
run_seed("compress α=0.5",
            np.clip(1/(1+np.exp(-(L_mean + 0.5*(L_FR-L_mean)))), eps, 1-eps))
run_seed("compress α=0.3",
            np.clip(1/(1+np.exp(-(L_mean + 0.3*(L_FR-L_mean)))), eps, 1-eps))
run_seed("no-learning",            P_NL)
# Direct seed using documented PR cell value
P_seed = P_FR.copy()
# Replace the (1,-1,1) cell area with PR-target prices
target = 0.632
# Smear toward target across the whole tensor by interpolating prices
P_seed = 0.5 * P_FR + 0.5 * np.full_like(P_FR, target)
run_seed("FR/target blend",        np.clip(P_seed, eps, 1-eps))

# Random perturbations
for sigma, sd in ((0.3, 7), (0.5, 11), (0.7, 13), (1.0, 17)):
    rng = np.random.default_rng(sd)
    P_seed = np.clip(P_NL + sigma * rng.standard_normal(P_NL.shape) * 0.2,
                       eps, 1-eps)
    run_seed(f"NL + σ={sigma:.1f}·rand", P_seed)
