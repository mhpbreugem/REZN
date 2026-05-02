"""Task 4c: Fig 11 — mechanisms bar chart (no-learning 1-R²).

6 configurations of 3 agents, no-learning weighted 1-R² at each.
1. CRRA symmetric (γ=0.5, τ=2 each)
2. Het γ = (0.5, 3, 10), equal τ=2
3. Het τ = (1, 3, 10), equal γ=0.5
4. Het γ + het τ aligned (low γ = high τ)
5. Het γ + het τ opposed (low γ = low τ)
6. CRRA γ=2 (weak effect)

Saves: results/full_ree/fig11_G20_pgfplots.tex
       results/full_ree/fig11_G20.json
"""
import json
import numpy as np
from scipy.optimize import brentq

RESULTS_DIR = "results/full_ree"
N_MC = 50000

CONFIGS = [
    ("CRRA symm γ=0.5",   [0.5, 0.5, 0.5], [2.0, 2.0, 2.0]),
    ("Het γ=(0.5,3,10)", [0.5, 3.0, 10.0], [2.0, 2.0, 2.0]),
    ("Het τ=(1,3,10)",   [0.5, 0.5, 0.5], [1.0, 3.0, 10.0]),
    ("Het γ+τ aligned",  [10.0, 3.0, 0.5], [10.0, 3.0, 1.0]),  # low γ = high τ
    ("Het γ+τ opposed",  [0.5, 3.0, 10.0], [1.0, 3.0, 10.0]),  # low γ = low τ
    ("CRRA symm γ=2",    [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]),
]


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


def market_clear_het(u, gammas, taus):
    """3-agent het γ/τ no-learning market clear."""
    priors = [Lam_scalar(taus[k]*u[k]) for k in range(3)]
    def F(p):
        return sum(crra_d(priors[k], p, gammas[k]) for k in range(3))
    f_lo, f_hi = F(1e-4), F(1-1e-4)
    if f_lo*f_hi > 0:
        return 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
    return brentq(F, 1e-4, 1-1e-4, xtol=1e-12)


def signal_density_log(u, v, tau):
    mean = float(v) - 0.5
    return 0.5*np.log(tau/(2*np.pi)) - 0.5*tau*(u - mean)**2


def weighted_R2(gammas, taus, n_mc, rng):
    """MC sample u_k from het signal density (mixture conditional on v).
    Sampling from prior density → use UNWEIGHTED regression.
    """
    Ts = np.empty(n_mc); lp = np.empty(n_mc)
    valid = np.zeros(n_mc, dtype=bool)
    for n in range(n_mc):
        v = rng.integers(0, 2)
        u = np.array([rng.normal(v - 0.5, 1.0/np.sqrt(taus[k]))
                            for k in range(3)])
        p = market_clear_het(u, gammas, taus)
        if not (1e-10 < p < 1-1e-10):
            continue
        Ts[n] = float(np.sum([taus[k]*u[k] for k in range(3)]))
        lp[n] = logit(p)
        valid[n] = True
    Ts = Ts[valid]; lp = lp[valid]
    slope, intercept = np.polyfit(Ts, lp, 1)
    pred = slope*Ts + intercept
    var_tot = np.var(lp)
    var_res = np.mean((lp - pred)**2)
    return float(var_res/var_tot), float(slope), int(len(Ts))


def main():
    rng = np.random.default_rng(123)
    results = []
    for name, gammas, taus in CONFIGS:
        r2, slope, n = weighted_R2(gammas, taus, N_MC, rng)
        results.append({"name": name, "gamma": gammas, "tau": taus,
                              "1-R2": r2, "slope": slope, "n": n})
        print(f"  {name:25s}: 1-R²={r2:.6e}, slope={slope:.4f}")

    out = f"{RESULTS_DIR}/fig11_G20.json"
    with open(out, "w") as f:
        json.dump({"figure": "fig11",
                          "params": {"K": 3, "n_mc": N_MC,
                                          "method": "no-learning weighted 1-R²"},
                          "configs": results}, f, indent=2)
    print(f"\nSaved {out}")

    out_tex = f"{RESULTS_DIR}/fig11_G20_pgfplots.tex"
    lines = ["% Fig 11 mechanisms bar chart (no-learning weighted 1-R²)", ""]
    pts = "".join(f"({i+1},{r['1-R2']:.6e})" for i, r in enumerate(results))
    lines.append(f"\\addplot coordinates {{{pts}}};")
    for i, r in enumerate(results):
        lines.append(f"% bar {i+1}: {r['name']} → 1-R²={r['1-R2']:.4e}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_tex}")


if __name__ == "__main__":
    main()
