"""Post-processor: compute weighted 1-R² for each γ ckpt and emit fig4B.

Loads converged ckpts at G=20 UMAX=5 trim=0.05 (γ ∈ {0.1, 0.25, 0.5, 1.0,
2.0, 4.0}), computes weighted 1-R² and slope per FIGURES_TODO.md spec,
emits fig4B_G20_gamma_sweep.json + fig4B_G20_pgfplots.tex.

Also computes no-learning weighted 1-R² at each γ.
"""
import json
import numpy as np
from scipy.optimize import brentq

RESULTS_DIR = "results/full_ree"
G = 20
UMAX = 5.0
TAU = 2.0

# γ → ckpt path
CKPT = {
    0.5: f"{RESULTS_DIR}/posterior_v3_G20_umax5_trim05_mp300.json",  # seed
    1.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g100_mp50.json",
    2.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g200_mp50.json",
    4.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g400_mp50.json",
    0.25: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g025_mp50.json",
    0.1: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g010_mp50.json",
}
PAPER_GAMMAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def Lam(z):
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


def load_ckpt(path):
    with open(path) as f:
        d = json.load(f)
    u_grid = np.array([float(s) for s in d["u_grid"]])
    p_grid = np.array([[float(s) for s in row] for row in d["p_grid"]])
    mu = np.array([[float(s) for s in row] for row in d["mu_strings"]])
    return u_grid, p_grid, mu


def mu_at(u, p, u_grid, p_grid, mu):
    G = len(u_grid)
    if u <= u_grid[0]: ia = ib = 0; w = 0.0
    elif u >= u_grid[-1]: ia = ib = G - 1; w = 0.0
    else:
        ib = int(np.searchsorted(u_grid, u))
        ia = ib - 1
        w = (u - u_grid[ia])/(u_grid[ib] - u_grid[ia])

    def row(i):
        pr = p_grid[i]
        if p <= pr[0]: return mu[i, 0]
        if p >= pr[-1]: return mu[i, -1]
        return float(np.interp(p, pr, mu[i]))

    if ia == ib: return row(ia)
    return (1-w)*row(ia) + w*row(ib)


def market_clear(u3, gamma, u_grid, p_grid, mu):
    def F(p):
        return sum(crra_d(mu_at(uk, p, u_grid, p_grid, mu), p, gamma)
                      for uk in u3)
    f_lo, f_hi = F(1e-4), F(1-1e-4)
    if f_lo*f_hi > 0:
        return 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
    return brentq(F, 1e-4, 1-1e-4, xtol=1e-12)


def market_clear_NL(u3, gamma, tau):
    priors = [Lam(tau*uk) for uk in u3]
    def F(p):
        return sum(crra_d(mk, p, gamma) for mk in priors)
    f_lo, f_hi = F(1e-4), F(1-1e-4)
    if f_lo*f_hi > 0:
        return 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
    return brentq(F, 1e-4, 1-1e-4, xtol=1e-12)


def weighted_R2(u_grid, gamma, tau, mu_func):
    """mu_func: (u3) -> price.  weighted regression of logit(p) on T*."""
    f0 = np.array([signal_density(u, 0, tau) for u in u_grid])
    f1 = np.array([signal_density(u, 1, tau) for u in u_grid])
    Ts = []; lp = []; ws = []
    G = len(u_grid)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u3 = (u_grid[i], u_grid[j], u_grid[l])
                p = mu_func(u3)
                if not (1e-10 < p < 1-1e-10): continue
                w_ijl = 0.5*(f0[i]*f0[j]*f0[l] + f1[i]*f1[j]*f1[l])
                Ts.append(tau*(u3[0]+u3[1]+u3[2]))
                lp.append(logit(p))
                ws.append(w_ijl)
    Ts = np.array(Ts); lp = np.array(lp); ws = np.array(ws)
    slope, intercept = np.polyfit(Ts, lp, 1, w=np.sqrt(ws))
    pred = slope*Ts + intercept
    mean_lp = np.average(lp, weights=ws)
    var_tot = np.average((lp - mean_lp)**2, weights=ws)
    var_res = np.average((lp - pred)**2, weights=ws)
    return float(var_res/var_tot), float(slope), int(len(Ts))


def main():
    # Build canonical u_grid (all ckpts share this)
    u_grid_canonical = np.linspace(-UMAX, UMAX, G)

    ree_results = {}
    nl_results = {}
    print("=== REE weighted 1-R² ===")
    for gamma in PAPER_GAMMAS:
        path = CKPT.get(gamma)
        if path is None:
            print(f"  γ={gamma}: NO CKPT — skipping")
            continue
        try:
            u_grid, p_grid, mu = load_ckpt(path)
        except FileNotFoundError:
            print(f"  γ={gamma}: ckpt {path} not found yet — skipping")
            continue
        def fn(u3, gg=gamma, ug=u_grid, pg=p_grid, m=mu):
            return market_clear(u3, gg, ug, pg, m)
        one_R2, slope, n = weighted_R2(u_grid, gamma, TAU, fn)
        ree_results[gamma] = {"1-R2": one_R2, "slope": slope, "n_triples": n}
        print(f"  γ={gamma}: 1-R²={one_R2:.6e}, slope={slope:.6f}")

    print("\n=== No-learning weighted 1-R² ===")
    for gamma in PAPER_GAMMAS:
        def fn(u3, gg=gamma):
            return market_clear_NL(u3, gg, TAU)
        one_R2, slope, n = weighted_R2(u_grid_canonical, gamma, TAU, fn)
        nl_results[gamma] = {"1-R2": one_R2, "slope": slope}
        print(f"  γ={gamma}: NL 1-R²={one_R2:.6e}, slope={slope:.6f}")

    # Save summary JSON
    summary = {
        "figure": "fig4B",
        "params": {"G": G, "tau": TAU, "umax": UMAX, "trim": 0.05,
                       "weighting": "ex-ante 0.5*(f0³+f1³)"},
        "REE": [{"gamma": g, **ree_results[g]} for g in PAPER_GAMMAS
                  if g in ree_results],
        "no_learning": [{"gamma": g, **nl_results[g]}
                              for g in PAPER_GAMMAS],
    }
    out = f"{RESULTS_DIR}/fig4B_G20_gamma_sweep.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")

    # Pgfplots
    out_tex = f"{RESULTS_DIR}/fig4B_G20_pgfplots.tex"
    lines = [
        f"% Fig 4B weighted 1-R² vs γ (G={G}, τ={TAU}, UMAX={UMAX}, "
        f"trim=0.05)",
        "% REE",
    ]
    if ree_results:
        ree_pts = "".join(f"({g},{ree_results[g]['1-R2']:.6e})"
                              for g in PAPER_GAMMAS if g in ree_results)
        lines.append(f"\\addplot coordinates {{{ree_pts}}};")
    lines += ["", "% No-learning"]
    nl_pts = "".join(f"({g},{nl_results[g]['1-R2']:.6e})"
                          for g in PAPER_GAMMAS)
    lines.append(f"\\addplot coordinates {{{nl_pts}}};")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_tex}")


if __name__ == "__main__":
    main()
