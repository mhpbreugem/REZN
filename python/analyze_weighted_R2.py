"""Post-processor: re-solve every converged config from search_het_results.csv
and compute BOTH unweighted and prior-weighted 1-R² over the G^3 cells.

Weighting:  w[i,j,l] = ½·Π_k f₁(u_k, τ_k) + ½·Π_k f₀(u_k, τ_k)
then normalised Σ w = 1. This corresponds to the ex-ante joint signal density
(marginal over v). Weighted OLS R² computes the fraction of logit(p) variance
that the linear CARA-FR predictor fails to explain UNDER THE ACTUAL SIGNAL
DISTRIBUTION, rather than treating the grid as uniform.

Expected directional effect: weighted 1-R² is usually SMALLER than the
grid-uniform version because corners (u=±2 with tiny Gaussian density)
contribute a lot of curvature but carry little weight under the prior.
"""
from __future__ import annotations
import csv
import sys
import time
import numpy as np
import rezn_het as rh

CSV_IN  = "/home/user/REZN/python/search_het_results.csv"
CSV_OUT = "/home/user/REZN/python/search_het_results_weighted.csv"
G      = 9
UMAX   = 2.0

# --- Loading converged rows --------------------------------------------------

def load_converged():
    with open(CSV_IN, "r") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("converged") == "1"]


# --- Signal density prior ----------------------------------------------------

def prior_weights(u, taus):
    G = u.shape[0]
    taus = np.asarray(taus, dtype=float)
    W = np.zeros((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                f1 = (
                    rh._f1(u[i], taus[0])
                    * rh._f1(u[j], taus[1])
                    * rh._f1(u[l], taus[2])
                )
                f0 = (
                    rh._f0(u[i], taus[0])
                    * rh._f0(u[j], taus[1])
                    * rh._f0(u[l], taus[2])
                )
                W[i, j, l] = 0.5 * f1 + 0.5 * f0
    s = W.sum()
    return W / s if s > 0 else W


# --- R² computations ---------------------------------------------------------

def _cara_fr_predictor(u, taus, gammas):
    taus = np.asarray(taus, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus   # weights for T_CARA = Σ (τ_k/γ_k)·u_k / Σ(1/γ_k)
    G = u.shape[0]
    T = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T[i, j, l] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
    return T


def one_minus_R2(y, T, w=None):
    """(Possibly weighted) 1 − R² of OLS y on T, sample-mass weighting."""
    y = y.reshape(-1)
    T = T.reshape(-1)
    if w is None:
        w = np.ones_like(y) / y.size
    else:
        w = np.asarray(w).reshape(-1)
        w = w / w.sum()
    ymean = (w * y).sum()
    Tmean = (w * T).sum()
    yc = y - ymean
    Tc = T - Tmean
    Syy = (w * yc * yc).sum()
    STT = (w * Tc * Tc).sum()
    SyT = (w * yc * Tc).sum()
    if Syy == 0 or STT == 0:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


# --- Main --------------------------------------------------------------------

def main():
    converged = load_converged()
    print(f"re-solving {len(converged)} converged configs to compute weighted 1-R²")

    _ = rh.solve_picard(5, 2.0, 0.5, maxiters=3)  # warmup

    out = []
    for idx, row in enumerate(converged, start=1):
        taus   = tuple(float(row[f"tau_{k+1}"]) for k in range(3))
        gammas = tuple(float(row[f"gamma_{k+1}"]) for k in range(3))
        alpha  = float(row["alpha"])
        t0 = time.time()
        res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                              maxiters=5000, abstol=1e-10, alpha=alpha)
        dt = time.time() - t0
        u = res["u"]
        P = res["P_star"]
        PhiI = res["history"][-1] if res["history"] else float("nan")

        y = np.log(P / (1.0 - P))
        T = _cara_fr_predictor(u, taus, gammas)
        W = prior_weights(u, taus)

        r2_unw = one_minus_R2(y, T, None)
        r2_w   = one_minus_R2(y, T, W)

        print(f"  [{idx:2d}/{len(converged)}] τ={taus} γ={gammas} "
              f"  t={dt:5.1f}s  PhiI={PhiI:.2e}  "
              f"1-R²_grid={r2_unw:.4e}  1-R²_weighted={r2_w:.4e}")
        sys.stdout.flush()

        out.append({
            **row,
            "oneR2_weighted": f"{r2_w:.6e}",
            "oneR2_grid": f"{r2_unw:.6e}",
            "resolved_PhiI": f"{PhiI:.3e}",
            "resolved_time_s": f"{dt:.2f}",
        })

    fieldnames = list(out[0].keys())
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"\nwrote {CSV_OUT} with {len(out)} rows")

    # Ranked output
    print("\n=== Top configs ranked by PRIOR-WEIGHTED 1-R² ===")
    out.sort(key=lambda r: float(r["oneR2_weighted"]), reverse=True)
    print(f"{'rank':>4}  {'1-R² grid':>12}  {'1-R² weighted':>15}  "
          f"{'τ':<20}  {'γ':<20}")
    for i, r in enumerate(out, start=1):
        τ = (r["tau_1"], r["tau_2"], r["tau_3"])
        γ = (r["gamma_1"], r["gamma_2"], r["gamma_3"])
        print(f"{i:>4}  {float(r['oneR2_grid']):>12.4e}  "
              f"{float(r['oneR2_weighted']):>15.4e}  "
              f"{str(τ):<20}  {str(γ):<20}")


if __name__ == "__main__":
    main()
