"""Task 4a: Fig R1 — no-learning weighted 1-R² vs K (number of agents).

For K ∈ {3, 5, 7, 10, 15, 20}, γ ∈ {0.5, 1.0, 4.0}, τ=2.
Method: Monte Carlo sample (u_1, ..., u_K) from joint signal density
(mixture of N(±0.5, 1/τ)), market-clear, weighted regression of
logit(p) on T* with weights w = 0.5*(Π f_0 + Π f_1).

Saves: results/full_ree/figR1_G20_pgfplots.tex
       results/full_ree/figR1_G20.json
"""
import json
import numpy as np
from scipy.optimize import brentq

RESULTS_DIR = "results/full_ree"
TAU = 2.0
N_MC = 50000   # Monte Carlo samples per (K, γ)
KS = [3, 5, 7, 10, 15, 20]
GAMMAS = [0.5, 1.0, 4.0]


def Lam(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0/(1.0+np.exp(-z[pos]))
    e = np.exp(z[~pos]); out[~pos] = e/(1.0+e)
    return out


def Lam_scalar(z):
    z = float(z)
    if z >= 0: return 1.0/(1.0+np.exp(-z))
    e = np.exp(z); return e/(1.0+e)


def logit(p):
    p = float(np.clip(p, 1e-15, 1-1e-15))
    return float(np.log(p/(1-p)))


def crra_d(mu, p, gamma):
    z = (logit(mu) - logit(p))/gamma
    R = float(np.exp(z))
    return (R-1.0)/((1.0-p) + R*p)


def signal_density(u, v, tau):
    mean = float(v) - 0.5
    return float(np.sqrt(tau/(2*np.pi)) * np.exp(-tau/2 * (u - mean)**2))


def market_clear_NL_K(u_arr, gamma, tau):
    """K-agent no-learning market clearing. u_arr: (K,) array."""
    priors = [Lam_scalar(tau*uk) for uk in u_arr]
    def F(p):
        return sum(crra_d(mk, p, gamma) for mk in priors)
    f_lo, f_hi = F(1e-4), F(1-1e-4)
    if f_lo*f_hi > 0:
        return 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
    return brentq(F, 1e-4, 1-1e-4, xtol=1e-12)


def weighted_R2_K(K, gamma, tau, n_mc, rng):
    """MC sample K-tuples from joint signal mixture (= weight density) and
    use UNWEIGHTED regression — the sampling distribution IS the prior weight,
    so weighted regression = unweighted regression on these samples.
    """
    sigma = 1.0/np.sqrt(tau)
    v_samples = rng.integers(0, 2, size=n_mc)
    Ts = np.empty(n_mc); lp = np.empty(n_mc)
    valid = np.zeros(n_mc, dtype=bool)
    for n in range(n_mc):
        v = v_samples[n]
        u_arr = rng.normal(v - 0.5, sigma, size=K)
        p = market_clear_NL_K(u_arr, gamma, tau)
        if not (1e-10 < p < 1-1e-10):
            continue
        Ts[n] = tau * np.sum(u_arr)
        lp[n] = logit(p)
        valid[n] = True
    Ts = Ts[valid]; lp = lp[valid]
    slope, intercept = np.polyfit(Ts, lp, 1)
    pred = slope*Ts + intercept
    var_tot = np.var(lp)
    var_res = np.mean((lp - pred)**2)
    return float(var_res/var_tot), float(slope), int(len(Ts))


def main():
    rng = np.random.default_rng(42)
    results = {gamma: [] for gamma in GAMMAS}

    for gamma in GAMMAS:
        for K in KS:
            r2, slope, n = weighted_R2_K(K, gamma, TAU, N_MC, rng)
            results[gamma].append({"K": K, "1-R2": r2, "slope": slope,
                                          "n_valid": n})
            print(f"  K={K:2d}, γ={gamma}: 1-R²={r2:.6e}, "
                  f"slope={slope:.4f}, n={n}")

    # Save JSON
    out = f"{RESULTS_DIR}/figR1_G20.json"
    summary = {
        "figure": "figR1",
        "params": {"tau": TAU, "method": "no-learning weighted 1-R²",
                       "n_mc": N_MC,
                       "weighting": "ex-ante 0.5*(f0^K + f1^K)"},
        "curves": [{"gamma": g, "points": results[g]} for g in GAMMAS],
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")

    # pgfplots
    out_tex = f"{RESULTS_DIR}/figR1_G20_pgfplots.tex"
    lines = [f"% Fig R1 weighted 1-R² vs K (no learning, τ={TAU})", ""]
    for g in GAMMAS:
        pts = "".join(f"({d['K']},{d['1-R2']:.6e})" for d in results[g])
        lines += [f"% γ={g}",
                   f"\\addplot coordinates {{{pts}}};", ""]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_tex}")


if __name__ == "__main__":
    main()
