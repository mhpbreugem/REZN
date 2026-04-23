"""Post-processor for every converged config from search_het_results.csv:

  • re-solve via Picard
  • compute grid-uniform and prior-weighted 1-R²
  • compute per-agent value functions:
        V_k^full  = E[ U_k(W + x_k·(v - p*)) ]   using μ_k(u_k, p*)  (private+public)
        V_k^pub   = E[ U_k(W + x^pub_k·(v - p*)) ] using μ^pub(p*)    (public only)
    and the value of agent k's private information  ΔV_k = V_k^full − V_k^pub.
  • welfare totals Σ_k V_k^full and Σ_k V_k^pub.

All expectations are taken under the ex-ante joint signal density:
      Pr(u_1,u_2,u_3 | v)   with   Pr(v=1)=Pr(v=0)=½.

Signal-triple probabilities on the G×G×G grid use trapezoidal quadrature
(Δu = 2·UMAX/(G-1) spacing) so the reported values are consistent with a
continuous Gaussian signal integral.

The public-only posterior μ^pub(p) is an inverse-Bayes estimate: a
Gaussian kernel in logit-price space aggregates prior f_v weights across
cells whose equilibrium price is near p. Choice of bandwidth is the only
free knob — default 0.2 in logit-p units.
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


# --- Per-cell prior densities & public-only posterior ----------------------

def per_cell_f_weights(u, taus):
    """Returns F1[i,j,l] = Π_k f₁(u_k, τ_k)  and  F0[i,j,l] = Π_k f₀(u_k, τ_k)
    — the conditional signal densities per state at each grid cell."""
    G = u.shape[0]
    taus = np.asarray(taus, dtype=float)
    F1 = np.empty((G, G, G)); F0 = np.empty_like(F1)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                F1[i,j,l] = rh._f1(u[i], taus[0]) * rh._f1(u[j], taus[1]) * rh._f1(u[l], taus[2])
                F0[i,j,l] = rh._f0(u[i], taus[0]) * rh._f0(u[j], taus[1]) * rh._f0(u[l], taus[2])
    return F1, F0


def public_posterior(Pg, F1, F0, h=0.2):
    """μ^pub(p) at each cell: posterior on v=1 given ONLY the price observation.

    Kernel estimate in logit-price space:
       π(p | v)  ∝  Σ_cells  K_h(logit(p) - logit(p*[cell]))  · F_v[cell]
    with normal kernel K_h.  Bandwidth h (in logit units) controls smoothing.
    """
    p = Pg.reshape(-1)
    lp = np.log(p / (1.0 - p))
    F1f = F1.reshape(-1); F0f = F0.reshape(-1)
    n = p.size
    mu = np.empty(n)
    # O(n^2) at n=G^3=729 -> ~0.5M ops, instant.
    for k in range(n):
        d = (lp - lp[k]) / h
        K = np.exp(-0.5 * d * d)
        A1 = (F1f * K).sum()
        A0 = (F0f * K).sum()
        mu[k] = A1 / (A0 + A1) if (A0 + A1) > 0 else 0.5
    return mu.reshape(Pg.shape)


# --- Utility & value functions --------------------------------------------

def utility(W, gamma):
    # CRRA;  γ=1 → log utility
    out = np.empty_like(W, dtype=float)
    pos = W > 0
    out[~pos] = -1e30  # heavy penalty for bankruptcy (shouldn't happen at FP)
    if abs(gamma - 1.0) < 1e-12:
        out[pos] = np.log(W[pos])
    else:
        out[pos] = W[pos]**(1.0 - gamma) / (1.0 - gamma)
    return out


def value_function(Pg, mu_k, gamma_k, W_k, F1, F0):
    """V_k = E_{v,u}[ U(W + x_k·(v - p*)) ] with signal-triple prior weights.

    x_k at each cell uses the given posterior μ_k[i,j,l] (which differs
    between the full-info and public-only computations). The price is the
    equilibrium p*[i,j,l] in both cases.
    """
    # optimal demand per cell given the supplied posterior tensor
    p   = Pg
    mu  = np.clip(mu_k, 1e-12, 1 - 1e-12)
    pc  = np.clip(p,    1e-12, 1 - 1e-12)
    R   = np.exp((np.log(mu/(1-mu)) - np.log(pc/(1-pc))) / gamma_k)
    x   = W_k * (R - 1.0) / ((1.0 - pc) + R * pc)

    # wealth outcomes under each state
    W_if_v1 = W_k + x * (1.0 - p)
    W_if_v0 = W_k - x * p

    U_v1 = utility(W_if_v1, gamma_k)
    U_v0 = utility(W_if_v0, gamma_k)

    # ex-ante expectation:  ½·F1(cell)·U_v1 + ½·F0(cell)·U_v0,  normalised by Σ
    num = 0.5 * (F1 * U_v1 + F0 * U_v0)
    den = 0.5 * (F1       + F0)
    return float(num.sum() / den.sum())


def value_bundle(Pg, u, taus, gammas, W=1.0, h_pub=0.2):
    """Compute V_k^full, V_k^pub, ΔV_k for each agent and welfare totals."""
    G = u.shape[0]
    F1, F0 = per_cell_f_weights(u, taus)

    # posteriors per agent under FULL info (private + public) — reuse contour
    mus_full = np.empty((3, G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                m = rh._posteriors_at(i, j, l, float(Pg[i,j,l]), Pg, u,
                                      np.asarray(taus))
                mus_full[0,i,j,l] = m[0]
                mus_full[1,i,j,l] = m[1]
                mus_full[2,i,j,l] = m[2]

    mu_pub = public_posterior(Pg, F1, F0, h=h_pub)

    V_full = np.zeros(3); V_pub = np.zeros(3)
    for k in range(3):
        V_full[k] = value_function(Pg, mus_full[k], gammas[k], W, F1, F0)
        V_pub[k]  = value_function(Pg, mu_pub,       gammas[k], W, F1, F0)
    dV = V_full - V_pub
    return dict(V_full=V_full, V_pub=V_pub, dV=dV,
                W_full=float(V_full.sum()), W_pub=float(V_pub.sum()),
                W_delta=float(dV.sum()))


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

        vb = value_bundle(P, u, taus, gammas, W=1.0, h_pub=0.2)

        print(f"  [{idx:2d}/{len(converged)}] τ={taus} γ={gammas} "
              f"t={dt:5.1f}s  1-R²_grid={r2_unw:.3e}  1-R²_w={r2_w:.3e}  "
              f"ΔV=({vb['dV'][0]:+.3e},{vb['dV'][1]:+.3e},{vb['dV'][2]:+.3e})  "
              f"ΣV_full={vb['W_full']:.4f}  ΣV_pub={vb['W_pub']:.4f}")
        sys.stdout.flush()

        out.append({
            **row,
            "oneR2_weighted": f"{r2_w:.6e}",
            "oneR2_grid": f"{r2_unw:.6e}",
            "V_full_1": f"{vb['V_full'][0]:.6e}",
            "V_full_2": f"{vb['V_full'][1]:.6e}",
            "V_full_3": f"{vb['V_full'][2]:.6e}",
            "V_pub_1":  f"{vb['V_pub'][0]:.6e}",
            "V_pub_2":  f"{vb['V_pub'][1]:.6e}",
            "V_pub_3":  f"{vb['V_pub'][2]:.6e}",
            "dV_1":     f"{vb['dV'][0]:.6e}",
            "dV_2":     f"{vb['dV'][1]:.6e}",
            "dV_3":     f"{vb['dV'][2]:.6e}",
            "welfare_full": f"{vb['W_full']:.6e}",
            "welfare_pub":  f"{vb['W_pub']:.6e}",
            "welfare_gain": f"{vb['W_delta']:.6e}",
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

    # Ranked output — by prior-weighted 1-R²
    print("\n=== Top configs ranked by PRIOR-WEIGHTED 1-R² ===")
    out.sort(key=lambda r: float(r["oneR2_weighted"]), reverse=True)
    print(f"{'rk':>3}  {'1-R²_grid':>10}  {'1-R²_w':>10}  "
          f"{'τ':<24}  {'γ':<24}  "
          f"{'ΔV_1':>9}  {'ΔV_2':>9}  {'ΔV_3':>9}  "
          f"{'W_full':>10}  {'W_pub':>10}")
    for i, r in enumerate(out, start=1):
        τ = (r["tau_1"], r["tau_2"], r["tau_3"])
        γ = (r["gamma_1"], r["gamma_2"], r["gamma_3"])
        print(f"{i:>3}  {float(r['oneR2_grid']):>10.3e}  "
              f"{float(r['oneR2_weighted']):>10.3e}  "
              f"{str(τ):<24}  {str(γ):<24}  "
              f"{float(r['dV_1']):>+9.2e}  {float(r['dV_2']):>+9.2e}  {float(r['dV_3']):>+9.2e}  "
              f"{float(r['welfare_full']):>10.3e}  {float(r['welfare_pub']):>10.3e}")

    # Ranked by aggregate welfare gain
    print("\n=== Top configs ranked by AGGREGATE VALUE OF PRIVATE INFO (ΣΔV_k) ===")
    out.sort(key=lambda r: float(r["welfare_gain"]), reverse=True)
    print(f"{'rk':>3}  {'welfare gain':>13}  {'τ':<24}  {'γ':<24}  {'1-R²_w':>10}")
    for i, r in enumerate(out, start=1):
        τ = (r["tau_1"], r["tau_2"], r["tau_3"])
        γ = (r["gamma_1"], r["gamma_2"], r["gamma_3"])
        print(f"{i:>3}  {float(r['welfare_gain']):>13.4e}  "
              f"{str(τ):<24}  {str(γ):<24}  {float(r['oneR2_weighted']):>10.3e}")


if __name__ == "__main__":
    main()
