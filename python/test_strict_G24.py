"""Test strict convergence at G=24, γ=0.5, τ=2.

Same pipeline as strict_ladder_emin13.py: slow Picard + NK polish.
Strict criterion: max < 1e-14, monotone in both directions.

Warm-start from G=14 strict, interpolate up.
"""
import time, json, warnings
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0
G = 24
TAU = 2.0
GAMMA = 0.5
TOL_MAX = 1e-14


def pava_2d(mu):
    return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    mu_new = np.empty((G_new, p_new.shape[1]))
    for i_new in range(G_new):
        u_target = u_new[i_new]
        u_clamped = np.clip(u_target, u_old[0], u_old[-1])
        r_above = np.searchsorted(u_old, u_clamped)
        r_below = max(r_above - 1, 0)
        r_above = min(r_above, len(u_old) - 1)
        w = ((u_clamped - u_old[r_below]) / (u_old[r_above] - u_old[r_below])
             if r_above != r_below else 1.0)
        for j_new in range(p_new.shape[1]):
            p_target = p_new[i_new, j_new]
            p_b = np.clip(p_target, p_old[r_below, 0], p_old[r_below, -1])
            mu_b = np.interp(p_b, p_old[r_below, :], mu_old[r_below, :])
            p_a = np.clip(p_target, p_old[r_above, 0], p_old[r_above, -1])
            mu_a = np.interp(p_a, p_old[r_above, :], mu_old[r_above, :])
            mu_new[i_new, j_new] = (1 - w) * mu_b + w * mu_a
    return np.clip(mu_new, EPS, 1 - EPS)


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma, n, na, alpha):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi(x_flat, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x_flat.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Test strict convergence at G={G}, γ={GAMMA}, τ={TAU} ===\n",
      flush=True)
ck = np.load(f"results/full_ree/posterior_v3_strict_G14_gamma{GAMMA:g}.npz")
mu_warm = ck["mu"]; u_warm = ck["u_grid"]; p_warm = ck["p_grid"]
print(f"Seeded from G=14 strict γ={GAMMA}\n", flush=True)

u_grid = np.linspace(-UMAX, UMAX, G)
p_lo, p_hi, p_grid = init_p_grid(u_grid, TAU, GAMMA, G)
mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
mu = pava_2d(mu)

t0 = time.time()
print("Slow Picard rounds...", flush=True)
mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.05)
d1 = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  r1 (α=0.05): max={d1['max']:.3e} med={d1['med']:.3e} u/p={d1['u_viol']}/{d1['p_viol']} t={time.time()-t0:.0f}s", flush=True)

mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 5000, 2500, 0.01)
d2 = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  r2 (α=0.01): max={d2['max']:.3e} med={d2['med']:.3e} u/p={d2['u_viol']}/{d2['p_viol']} t={time.time()-t0:.0f}s", flush=True)

mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA, 10000, 5000, 0.003)
d3 = measure(mu, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  r3 (α=0.003): max={d3['max']:.3e} med={d3['med']:.3e} u/p={d3['u_viol']}/{d3['p_viol']} t={time.time()-t0:.0f}s", flush=True)
mu_picard = mu

print("\nNK polish...", flush=True)
try:
    sol = newton_krylov(
        lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA),
        mu.ravel(), f_tol=TOL_MAX, maxiter=300,
        method="lgmres", verbose=False)
    mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "ok"
except NoConvergence as e:
    mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
    nk_status = "noconv_kept"
except (ValueError, RuntimeError) as exc:
    mu_nk = mu
    nk_status = f"err:{type(exc).__name__}"
d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"  post-NK: max={d_nk['max']:.3e} med={d_nk['med']:.3e} "
      f"u/p={d_nk['u_viol']}/{d_nk['p_viol']} NK={nk_status} t={time.time()-t0:.0f}s",
      flush=True)

strict_conv_nk = (d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0
                   and d_nk["p_viol"] == 0)
strict_conv_picard = (d3["max"] < TOL_MAX and d3["u_viol"] == 0
                       and d3["p_viol"] == 0)

if strict_conv_nk:
    print("\n*** STRICT CONV via NK ***")
    mu_use = mu_nk
elif strict_conv_picard:
    print("\n*** STRICT CONV via Picard alone ***")
    mu_use = mu_picard
elif d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
    print("\n*** NK monotone but max not strict; falling back to picard ***")
    mu_use = mu_nk if d_nk["max"] < d3["max"] else mu_picard
else:
    print(f"\n*** STRICT FAILED at G={G} (NK violations: u={d_nk['u_viol']}, p={d_nk['p_viol']}) ***")
    mu_use = mu_picard

r2, slope, _ = measure_R2(mu_use, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
d = measure(mu_use, u_grid, p_grid, p_lo, p_hi, TAU, GAMMA)
print(f"\nFinal: max={d['max']:.3e}, med={d['med']:.3e}, "
      f"u/p={d['u_viol']}/{d['p_viol']}")
print(f"  1-R² = {r2:.4e}, slope = {slope:.4f}")
print(f"  total time = {time.time()-t0:.0f}s")

# Save if strict
if strict_conv_nk or strict_conv_picard:
    np.savez(f"results/full_ree/posterior_v3_strict_G{G}.npz",
             mu=mu_use, u_grid=u_grid, p_grid=p_grid, p_lo=p_lo, p_hi=p_hi)
    print(f"\nSaved: results/full_ree/posterior_v3_strict_G{G}.npz")

# Always save summary
result = {"G": G, "tau": TAU, "gamma": GAMMA,
          "max": d["max"], "med": d["med"],
          "u_viol": d["u_viol"], "p_viol": d["p_viol"],
          "1-R^2": float(r2), "slope": float(slope),
          "strict_conv_nk": strict_conv_nk,
          "strict_conv_picard": strict_conv_picard,
          "nk_status": nk_status}
with open(f"results/full_ree/test_strict_G{G}.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: results/full_ree/test_strict_G{G}.json")
