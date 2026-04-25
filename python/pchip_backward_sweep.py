"""Backward τ sweep along the POST-JUMP branch.

The forward sweep (pchip_continuation.py) found that at γ=(3,3,3) the CRRA
equilibrium Φ map has (at least) two branches:

  • Pre-jump branch: present for τ ∈ [3.00, 3.45], 1-R² ≈ 1e-3, PR gap ≈ -6e-4.
  • Post-jump branch: found by NK at τ ≥ 3.39, 1-R² ≈ 1e-1, PR gap ≈ +1e-2.

This script starts from a high-1-R² post-jump solution (picked from the
pickled CACHE) and walks τ DOWNWARD in -0.01 steps, warm-starting each
step only from the previous BACKWARD solution (never from _nearest on
the global CACHE, which would snap us back to the pre-jump branch).

We want to see whether the post-jump branch persists below τ=3.39 — i.e.
does CRRA admit a high-PR equilibrium also in the τ<3.39 regime, hidden
by Picard's basin of attraction?

Output: pchip_backward_results.csv (separate from the forward CSV).
"""
from __future__ import annotations
import csv
import pickle
import sys
import time

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

import rezn_het as rh
import rezn_pchip as rp


G         = 11
UMAX      = 2.0
GAMMA     = 3.0
ABSTOL    = 1e-4
F_TOL     = 3e-3
TAU_STOP  = 2.00          # walk τ down to this value (or until tracking fails)
TAU_STEP  = 0.05
MAX_FAIL  = 3             # stop after this many consecutive failures

CACHE_PKL    = "/home/user/REZN/python/pchip_cache.pkl"
FORWARD_CSV  = "/home/user/REZN/python/pchip_continuation_results.csv"
OUT_CSV      = "/home/user/REZN/python/pchip_backward_results.csv"
STATUS_PATH  = "/home/user/REZN/python/sweep_status.txt"


FIELDS = ["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
          "alpha","iters","time_s","PhiI","Finf","oneR2_het",
          "p_star","mu_1","mu_2","mu_3","pr_gap","converged","init"]


def _one_minus_R2_het(Pg, u, taus, gammas):
    taus = np.asarray(taus, float); gammas = np.asarray(gammas, float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1 - Pg)).reshape(-1)
    G_ = u.shape[0]
    T = np.empty(G_*G_*G_)
    it = 0
    for i in range(G_):
        for j in range(G_):
            for l in range(G_):
                T[it] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
                it += 1
    yc = y - y.mean(); Tc = T - T.mean()
    num = (yc*Tc).sum()**2
    den = (yc*yc).sum() * (Tc*Tc).sum()
    if den == 0 or not np.isfinite(den):
        return float("nan")
    R2 = num / den
    return float(1.0 - R2)


def _find_post_jump_seed():
    """Find a high-1-R² (post-jump) row in the forward CSV. Return its
    (τ, P_star) by re-solving with NK from the cached pre-jump P at the
    SAME τ."""
    with open(FORWARD_CSV, "r") as f:
        rows = list(csv.DictReader(f))
    # Post-jump rows: γ=3 homo, τ≥3.39, 1-R² > 0.01
    candidates = []
    for r in rows:
        try:
            if not (abs(float(r["gamma_1"])-3.0)<1e-9
                    and abs(float(r["gamma_2"])-3.0)<1e-9
                    and abs(float(r["gamma_3"])-3.0)<1e-9):
                continue
            if int(r["converged"]) != 1:
                continue
            oneR2 = float(r["oneR2_het"])
            tau = float(r["tau_1"])
            if oneR2 > 0.01 and tau > 3.38:
                candidates.append((tau, oneR2, r))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No post-jump seed found in forward CSV.")
    candidates.sort(key=lambda x: x[0])  # lowest τ first — closest to jump
    tau_seed = candidates[0][0]
    # Load CACHE and pick the entry at this τ with the HIGHEST 1-R² (post-
    # jump). CACHE pickle may have multiple entries at same τ if both
    # branches were visited; take the one matching the forward CSV record.
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    matches = [e for e in cache
               if abs(e["taus"][0]-tau_seed) < 1e-9
               and abs(e["gammas"][0]-3.0) < 1e-9]
    if not matches:
        raise RuntimeError(f"No CACHE entry for τ={tau_seed}")
    # If multiple matches, pick the one with highest 1-R²
    def one_r2(e):
        u = np.linspace(-UMAX, UMAX, G)
        return _one_minus_R2_het(e["P_star"], u,
                                 e["taus"], e["gammas"])
    matches.sort(key=one_r2, reverse=True)
    P0 = matches[0]["P_star"]
    r2 = one_r2(matches[0])
    print(f"[seed] τ={tau_seed:.3f}  1-R²={r2:.4e}")
    sys.stdout.flush()
    return tau_seed, P0


def _solve_nk(taus, gammas, P_init):
    u = np.linspace(-UMAX, UMAX, G)
    taus_v = rh._as_vec3(taus[0])
    gammas_v = rh._as_vec3(gammas[0])
    Ws = rh._as_vec3(1.0)

    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus_v, gammas_v, Ws)
        return x - Pn.reshape(-1)

    x0 = np.clip(P_init, 1e-9, 1-1e-9).reshape(-1)
    try:
        sol = newton_krylov(F, x0, f_tol=max(ABSTOL, 1e-8),
                            rdiff=1e-7, method="lgmres",
                            maxiter=30, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
    if not np.all(np.isfinite(sol)):
        sol = x0
    P_star = np.clip(sol.reshape(G, G, G), 1e-9, 1 - 1e-9)
    Pn = rp._phi_map_pchip(P_star, u, taus_v, gammas_v, Ws)
    return P_star, float(np.abs(P_star - Pn).max())


def _solve_anderson(taus, gammas, P_init, m=10, iters=500):
    res = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                   maxiters=iters, abstol=ABSTOL,
                                   m_window=m, damping=1.0,
                                   P_init=P_init,
                                   status_path=STATUS_PATH, status_every=25)
    Finf = float(np.abs(res["residual"]).max())
    return res["P_star"], Finf


def main():
    print("numba warmup…"); sys.stdout.flush()
    _ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

    tau_seed, P_post = _find_post_jump_seed()

    # Walk τ downward from just below tau_seed
    taus_bw = np.arange(tau_seed - TAU_STEP, TAU_STOP - 1e-9, -TAU_STEP)

    writer_rows = []
    def flush():
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for rr in writer_rows: w.writerow(rr)

    # Record the seed itself
    u = np.linspace(-UMAX, UMAX, G)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    ir, jr, lr = idx(1.0), idx(-1.0), idx(1.0)
    t0 = (tau_seed,)*3; g0 = (GAMMA,)*3
    Pn = rp._phi_map_pchip(P_post, u, rh._as_vec3(tau_seed),
                            rh._as_vec3(GAMMA), rh._as_vec3(1.0))
    Finf_seed = float(np.abs(P_post - Pn).max())
    one_het = _one_minus_R2_het(P_post, u, t0, g0)
    mu = rh.posteriors_at(ir, jr, lr, float(P_post[ir,jr,lr]), P_post,
                          u, np.asarray(t0))
    writer_rows.append({
        "tau_1": t0[0], "tau_2": t0[1], "tau_3": t0[2],
        "gamma_1": g0[0], "gamma_2": g0[1], "gamma_3": g0[2],
        "alpha": "SEED", "iters": 0, "time_s": "0.00",
        "PhiI": f"{Finf_seed:.3e}", "Finf": f"{Finf_seed:.3e}",
        "oneR2_het": f"{one_het:.6e}",
        "p_star": f"{float(P_post[ir,jr,lr]):.10f}",
        "mu_1": f"{mu[0]:.8f}", "mu_2": f"{mu[1]:.8f}", "mu_3": f"{mu[2]:.8f}",
        "pr_gap": f"{mu[0]-mu[1]:.6f}",
        "converged": 1, "init": "post_jump_seed",
    })
    flush()

    P_cur = P_post.copy()
    fail_count = 0
    for tau in taus_bw:
        t = (float(tau),)*3; g = (GAMMA,)*3
        # Try NK first (fast, usually works on post-jump branch)
        t_start = time.time()
        P_star, Finf = _solve_nk(t, g, P_cur)
        alpha = "NK"
        iters = 1
        if Finf > F_TOL:
            # Fall back to Anderson
            P_star2, Finf2 = _solve_anderson(t, g, P_cur, m=10, iters=400)
            if Finf2 < Finf:
                P_star, Finf, alpha, iters = P_star2, Finf2, "A10", 400
        dt = time.time() - t_start

        # Compute diagnostics
        one_het = _one_minus_R2_het(P_star, u, t, g)
        mu = rh.posteriors_at(ir, jr, lr, float(P_star[ir,jr,lr]),
                              P_star, u, np.asarray(t))

        converged = (Finf < F_TOL) and np.all(np.isfinite(P_star))
        writer_rows.append({
            "tau_1": t[0], "tau_2": t[1], "tau_3": t[2],
            "gamma_1": g[0], "gamma_2": g[1], "gamma_3": g[2],
            "alpha": alpha, "iters": iters, "time_s": f"{dt:.2f}",
            "PhiI": f"{Finf:.3e}", "Finf": f"{Finf:.3e}",
            "oneR2_het": f"{one_het:.6e}",
            "p_star": f"{float(P_star[ir,jr,lr]):.10f}",
            "mu_1": f"{mu[0]:.8f}", "mu_2": f"{mu[1]:.8f}", "mu_3": f"{mu[2]:.8f}",
            "pr_gap": f"{mu[0]-mu[1]:.6f}",
            "converged": int(converged), "init": "backward",
        })
        flush()
        print(f"  τ={tau:.3f}  α={alpha}  Finf={Finf:.2e}  "
              f"1-R²={one_het:.3e}  conv={int(converged)}  time={dt:.1f}s")
        sys.stdout.flush()

        if converged:
            fail_count = 0
            P_cur = P_star          # chain for next step
        else:
            fail_count += 1
            # Keep P_cur as the last-good warm start; try a tiny perturb next
            if fail_count >= MAX_FAIL:
                print(f"[stop] {fail_count} consecutive failures — likely the "
                      f"post-jump branch has terminated at τ≈{tau:.3f}.")
                break

    print(f"\n[done] {len(writer_rows)} backward rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
