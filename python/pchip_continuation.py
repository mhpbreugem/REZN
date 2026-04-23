"""PCHIP-based parameter continuation grid.

Start from the known (3,3,3)/(50,50,50) PCHIP solution and walk outward
in γ-space, warm-starting each new config from the previous converged
P tensor (with a small logit-space perturbation to break symmetry).
Each step tries an α-ladder α ∈ {1.0, 0.3, 0.1, 0.03} so oscillation
(seen at γ=(3,3,3) under α=1) falls back to damping.

Currently sweeps γ homogeneous along a log-grid, then walks along
heterogeneous axes. Grows the cache of solved configs as we go.

Output: pchip_continuation_results.csv with one row per attempted
config: τ, γ, α, iters, PhiI, Finf, 1-R²_het, 1-R²_eq, posteriors,
PR gap at (1,-1,1), converged flag, init method.
"""
from __future__ import annotations
import csv
import os
import sys
import time
import numpy as np

import rezn_het as rh
import rezn_pchip as rp


G        = 11
UMAX     = 2.0
TAU      = 3.0           # default τ for the γ sweep
GAMMA    = 3.0           # default γ for the τ sweep
ABSTOL   = 1e-6          # PCHIP-Picard floor at G=11 is ~1e-6 (spectral
                         # radius near 1); at G=7 it was ~1e-8. Accept 1e-6
                         # here to capture the PR signal across the sweep.
F_TOL    = 1.0
CSV_OUT  = "/home/user/REZN/python/pchip_continuation_results.csv"
# Anderson windows to try. Anderson with window m≈6 usually works very
# well; larger windows bring more memory-of-past iterates (better for
# slow modes) but can be unstable.
ANDERSON_WINDOWS = [6, 10, 15]
ANDERSON_MAXITER = 800    # Anderson rarely needs more than this if it works
PERTURB_SIGMA = 0.0      # DEBUG: disabled to isolate hang cause


# ---------------- Cache of converged (τ, γ, P*) -----------------------
CACHE = []               # dicts: {log_tg, P_star, taus, gammas}


def _log_tg(t, g):
    return np.log(np.concatenate([np.asarray(t, float), np.asarray(g, float)]))


def _nearest(t, g):
    if not CACHE: return None, None
    q = _log_tg(t, g)
    d = [float(np.linalg.norm(e["log_tg"] - q)) for e in CACHE]
    i = int(np.argmin(d))
    return CACHE[i]["P_star"], CACHE[i]["taus"], CACHE[i]["gammas"]


_PERTURB_RNG = np.random.default_rng()  # unseeded → different noise each call

def _perturb(P):
    if PERTURB_SIGMA == 0.0:
        return P
    logit_P = np.log(P / (1.0 - P))
    noise = _PERTURB_RNG.normal(0.0, PERTURB_SIGMA, P.shape)
    return np.clip(1.0 / (1.0 + np.exp(-(logit_P + noise))), 1e-9, 1 - 1e-9)


# ---------------- Solve one config with α-ladder + warm start ---------

def _linear_extrapolate(taus, gammas):
    """DEBUG: disabled. Returns None → solve_one falls back to _nearest."""
    return None
    if len(CACHE) < 2:
        return None
    q = _log_tg(taus, gammas)
    # distances
    dists = [(float(np.linalg.norm(e["log_tg"] - q)), i) for i, e in enumerate(CACHE)]
    dists.sort()
    # top 2 nearest entries — assume they are on the continuation path
    i1 = dists[0][1]; i2 = dists[1][1]
    e1 = CACHE[i1]; e2 = CACHE[i2]
    # direction from e2 → e1, extrapolate to query
    v21 = e1["log_tg"] - e2["log_tg"]          # last step taken
    v1q = q - e1["log_tg"]                      # step to query
    norm21 = np.linalg.norm(v21)
    if norm21 < 1e-9:
        return e1["P_star"]
    # cos similarity: if query continues the e2→e1 direction, extrapolate.
    cos = float(np.dot(v21, v1q)) / (norm21 * max(np.linalg.norm(v1q), 1e-12))
    if cos < 0.8:
        return e1["P_star"]                     # not aligned; just use nearest
    # first-order Taylor in logit space: logit P_q ≈ logit P_e1 + λ (logit P_e1 - logit P_e2)
    lam = float(np.dot(v21, v1q)) / (norm21 ** 2)
    LP1 = np.log(e1["P_star"] / (1 - e1["P_star"]))
    LP2 = np.log(e2["P_star"] / (1 - e2["P_star"]))
    LP_q = LP1 + lam * (LP1 - LP2)
    P_q = 1.0 / (1.0 + np.exp(-LP_q))
    return np.clip(P_q, 1e-9, 1 - 1e-9)


def solve_one(taus, gammas):
    """Warm-start from linearly-extrapolated (or nearest) cached P;
    apply tiny perturbation; try α-ladder."""
    t_near = g_near = None
    P_extrap = _linear_extrapolate(taus, gammas)
    if P_extrap is not None:
        P_warm = _perturb(P_extrap)
        # record which point we extrapolated "near"
        nr = _nearest(taus, gammas)
        if nr[0] is not None:
            t_near, g_near = nr[1], nr[2]
    else:
        nr = _nearest(taus, gammas)
        if nr[0] is not None:
            P_warm = _perturb(nr[0])
            t_near, g_near = nr[1], nr[2]
        else:
            P_warm = None

    best = None
    attempts = []
    # Attempt 1: plain Picard α=1 (works for seed and nearby configs)
    attempts.append(("P1.0", dict(solver="picard", alpha=1.0, maxiters=2000)))
    # Attempts 2-4: Anderson with growing windows
    for m in ANDERSON_WINDOWS:
        attempts.append((f"A{m}", dict(solver="anderson", m_window=m,
                                        maxiters=ANDERSON_MAXITER)))
    # Attempt 5: damped Picard as last resort
    attempts.append(("P0.3", dict(solver="picard", alpha=0.3, maxiters=5000)))

    for tag, opts in attempts:
        t0 = time.time()
        try:
            if opts["solver"] == "picard":
                res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                            maxiters=opts["maxiters"],
                                            abstol=ABSTOL, alpha=opts["alpha"],
                                            P_init=P_warm)
            else:
                res = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                              maxiters=opts["maxiters"],
                                              abstol=ABSTOL,
                                              m_window=opts["m_window"],
                                              damping=1.0, P_init=P_warm)
        except Exception as e:
            print(f"    {tag} error: {e}")
            continue
        dt = time.time() - t0
        PhiI = res["history"][-1] if res["history"] else float("inf")
        Finf = float(np.abs(res["residual"]).max())
        converged = (PhiI < ABSTOL) and (Finf < F_TOL)
        cand = dict(alpha=tag, iters=len(res["history"]), time=dt,
                    PhiI=PhiI, Finf=Finf, P_star=res["P_star"],
                    converged=converged,
                    init=("warm" if P_warm is not None else "cold"),
                    warm_from=(t_near, g_near))
        if best is None or (cand["converged"] and not best["converged"]) \
           or (not best["converged"] and cand["PhiI"] < best["PhiI"]):
            best = cand
        if converged:
            break
    return best


def one_minus_R2_het(Pg, u, taus, gammas):
    taus = np.asarray(taus, float); gammas = np.asarray(gammas, float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1 - Pg)).reshape(-1)
    G_ = u.shape[0]
    T = np.empty(G_ ** 3); k = 0
    for i in range(G_):
        for j in range(G_):
            for l in range(G_):
                T[k] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]; k += 1
    yc = y - y.mean(); Tc = T - T.mean()
    Syy = (yc * yc).sum(); STT = (Tc * Tc).sum(); SyT = (yc * Tc).sum()
    if Syy == 0 or STT == 0: return 0.0
    return 1 - (SyT * SyT) / (Syy * STT)


# ---------------- Build γ grid ----------------------------------------

def gamma_sweep():
    """Homogeneous γ grid at fixed τ=(TAU,TAU,TAU), unit-size steps."""
    vals = (
        list(np.arange(50.0, 1.0 - 1e-9, -1.0))          # 50, 49, ..., 2, 1
        + [0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    )
    for g in vals:
        yield (TAU, TAU, TAU), (float(g), float(g), float(g))


def tau_sweep():
    """Very fine homogeneous τ grid, push up from τ=3 (γ-sweep anchor):
      pass A: 3.00 → 5.00 in steps of +0.01 (201 points).
    Warm-start chain from each previous solution."""
    vals = np.arange(3.0, 5.0 + 1e-9, 0.01)
    for t in vals:
        yield (float(t), float(t), float(t)), (GAMMA, GAMMA, GAMMA)


def het_sweep(start_g=50.0):
    """Walk one gamma down while holding others at start_g. Three axes."""
    axes = [(0, [start_g, 30, 20, 10, 5, 3, 1, 0.3]),
            (1, [start_g, 30, 20, 10, 5, 3, 1, 0.3]),
            (2, [start_g, 30, 20, 10, 5, 3, 1, 0.3])]
    for axis, vals in axes:
        for v in vals:
            g = [start_g, start_g, start_g]
            g[axis] = v
            yield (TAU, TAU, TAU), tuple(g)


# ---------------- Main ------------------------------------------------

def main():
    # Warmup JIT
    print("numba JIT warmup…")
    sys.stdout.flush()
    _ = rp.solve_picard_pchip(5, 2.0, 0.5, maxiters=3)

    # Solve the SEED first: τ=(3,3,3) γ=(50,50,50) — easy near-CARA anchor.
    # Chain-warm through γ sweep down to γ=3, then launch τ sweep from there.
    print(f"[seed] solving τ=({TAU},{TAU},{TAU}) γ=(50,50,50)")
    sys.stdout.flush()
    seed = solve_one((TAU, TAU, TAU), (50.0, 50.0, 50.0))
    if seed["converged"]:
        CACHE.append({"log_tg": _log_tg((TAU,TAU,TAU), (50,50,50)),
                      "P_star": seed["P_star"].copy(),
                      "taus": (TAU,TAU,TAU), "gammas": (50,50,50)})
        print(f"  seed converged: iters={seed['iters']} "
              f"PhiI={seed['PhiI']:.2e} Finf={seed['Finf']:.2e}")
    else:
        print(f"  SEED FAILED: PhiI={seed['PhiI']:.2e}")
    sys.stdout.flush()

    u = np.linspace(-UMAX, UMAX, G)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(1.0), idx(-1.0), idx(1.0)

    # CSV writer
    fieldnames = ["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
                  "alpha","iters","time_s","PhiI","Finf","oneR2_het",
                  "p_star","mu_1","mu_2","mu_3","pr_gap","converged","init"]
    rows = []

    def flush():
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rr in rows: w.writerow(rr)

    def record(t, g, best):
        P = best["P_star"]
        if P is None:
            return
        try:
            mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r,j_r,l_r]), P, u,
                                    np.asarray(t))
            one_het = one_minus_R2_het(P, u, t, g)
        except Exception:
            mu = (float("nan"),)*3; one_het = float("nan")
        rows.append({
            "tau_1": t[0], "tau_2": t[1], "tau_3": t[2],
            "gamma_1": g[0], "gamma_2": g[1], "gamma_3": g[2],
            "alpha": f"{best['alpha']}",
            "iters": best["iters"],
            "time_s": f"{best['time']:.2f}",
            "PhiI": f"{best['PhiI']:.3e}",
            "Finf": f"{best['Finf']:.3e}",
            "oneR2_het": f"{one_het:.6e}",
            "p_star": f"{float(P[i_r,j_r,l_r]):.10f}",
            "mu_1": f"{mu[0]:.8f}", "mu_2": f"{mu[1]:.8f}",
            "mu_3": f"{mu[2]:.8f}",
            "pr_gap": f"{mu[0]-mu[1]:.6f}",
            "converged": int(best["converged"]),
            "init": best["init"],
        })

    # Record seed
    if seed is not None:
        record((TAU,TAU,TAU), (50,50,50), seed)
        flush()

    # Walk homogeneous γ downward (warm-start chain 50 → 49 → … → 3)
    print(f"\n=== homogeneous γ sweep (τ={TAU} fixed) ===")
    sys.stdout.flush()
    for (t, g) in gamma_sweep():
        if g == (50.0, 50.0, 50.0) and rows:
            continue
        # stop at γ=3, we'll pivot into the τ sweep
        if g[0] < GAMMA - 1e-9:
            break
        best = solve_one(t, g)
        record(t, g, best)
        flush()
        if best["converged"]:
            CACHE.append({"log_tg": _log_tg(t, g),
                          "P_star": best["P_star"].copy(),
                          "taus": t, "gammas": g})
        print(f"  γ={g[0]:>5.1f}  α={best['alpha']:<5}  "
              f"iters={best['iters']:>5}  PhiI={best['PhiI']:.2e}  "
              f"conv={best['converged']}  time={best['time']:.1f}s")
        sys.stdout.flush()

    # --- τ sweep at fixed γ=(GAMMA, GAMMA, GAMMA) ---
    # Seed the τ sweep: we need γ=(3,3,3) in the cache at τ=(3,3,3). That came
    # from the end of the γ sweep above.
    print(f"\n=== homogeneous τ sweep (γ={GAMMA} fixed) ===")
    sys.stdout.flush()
    for (t, g) in tau_sweep():
        # skip τ=(3,3,3) γ=(3,3,3) if already solved by γ sweep (it was)
        already = any((abs(r["tau_1"] - t[0]) < 1e-9
                       and abs(r["gamma_1"] - g[0]) < 1e-9) for r in rows)
        if already:
            continue
        best = solve_one(t, g)
        record(t, g, best)
        flush()
        if best["converged"]:
            CACHE.append({"log_tg": _log_tg(t, g),
                          "P_star": best["P_star"].copy(),
                          "taus": t, "gammas": g})
        print(f"  τ={t[0]:>5.1f}  α={best['alpha']:<5}  "
              f"iters={best['iters']:>5}  PhiI={best['PhiI']:.2e}  "
              f"Finf={best['Finf']:.2e}  "
              f"conv={best['converged']}  time={best['time']:.1f}s")
        sys.stdout.flush()

    # Summary
    n_conv = sum(1 for r in rows if r["converged"] == 1)
    print(f"\n[done] {n_conv}/{len(rows)} converged")
    print(f"CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
