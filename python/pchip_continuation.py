"""PCHIP-based parameter continuation grid.

Start from the known (3,3,3)/(500,500,500) PCHIP solution and walk outward
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
import pickle
import sys
import time
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

import rezn_het as rh
import rezn_pchip as rp


G        = 11
UMAX     = 2.0           # density 2.5 points per SD
TAU      = 3.0           # default τ for the γ sweep
GAMMA    = 3.0           # default γ for the τ sweep
ABSTOL   = 1e-11         # Picard-step tolerance. FD-Jacobian noise floor in
                         # our stack is ~1e-10; 1e-11 is an ambitious target
                         # that may only be reachable via continuation warm
                         # start (not cold).
F_TOL    = 1e-9          # Practical floor at G=11 with logit-PCHIP + FD-NK.
                         # Seed typically hits 7e-11 via Anderson polishing;
                         # warm-started configs can sometimes go tighter.
CSV_OUT  = "/home/user/REZN/python/pchip_G11logit_forward.csv"
CACHE_PKL = "/home/user/REZN/python/pchip_G11logit_cache.pkl"
STATUS_PATH = "/home/user/REZN/python/sweep_status.txt"
# Anderson windows to try. Anderson with window m≈6 usually works very
# well; larger windows bring more memory-of-past iterates (better for
# slow modes) but can be unstable.
ANDERSON_WINDOWS = [6, 10, 15]
ANDERSON_MAXITER = 800    # Anderson rarely needs more than this if it works
PERTURB_SIGMA = 1e-7     # tiny random logit noise on warm start


# ---------------- Cache of converged (τ, γ, P*) -----------------------
CACHE = []               # dicts: {log_tg, P_star, taus, gammas}


def _save_cache():
    """Pickle CACHE to disk so subsequent runs skip re-solving."""
    try:
        with open(CACHE_PKL, "wb") as f:
            pickle.dump(CACHE, f)
    except Exception as e:
        print(f"  [cache save failed: {e}]")


def _preload_from_csv(csv_path, fieldnames):
    """Load previously-converged rows (Finf ≤ F_TOL) from CSV. If a
    pickled CACHE exists and matches, use it directly (fast path).
    Otherwise re-solve each config once to rebuild P tensors (slow path).
    Returns the list of rows (dicts, for re-writing)."""
    if not os.path.exists(csv_path):
        return []

    # Try the pickle fast-path
    if os.path.exists(CACHE_PKL):
        try:
            with open(CACHE_PKL, "rb") as f:
                cached = pickle.load(f)
            # Expect list of dicts matching CACHE schema; sanity-check
            if isinstance(cached, list) and cached \
               and all(isinstance(e, dict) and "P_star" in e and "log_tg" in e
                       for e in cached):
                CACHE.extend(cached)
                print(f"[cache] loaded {len(CACHE)} entries from {CACHE_PKL}")
                sys.stdout.flush()
        except Exception as e:
            print(f"  [cache load failed: {e}; will re-solve]")
            CACHE.clear()

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # Keep only Finf ≤ F_TOL AND converged=1 (reject fake convergences
    # and previously-failed rows — those should be re-attempted).
    good = []
    bad = 0
    for r in rows:
        try:
            Finf = float(r["Finf"])
        except Exception:
            continue
        if int(r.get("converged", 0)) == 1 and Finf <= F_TOL:
            # Normalize τ, γ to floats (csv.DictReader gives strings)
            for k in ("tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3"):
                r[k] = float(r[k])
            good.append(r)
        else:
            bad += 1
    print(f"[preload] {len(good)} clean rows from CSV ({bad} rejected)")
    sys.stdout.flush()

    # If CACHE has any entries from the pickle, use them and skip re-solve.
    # The predictor will handle missing neighbors via linear extrapolation.
    if len(CACHE) > 0:
        print(f"  [cache hit] skipping re-solve, CACHE has {len(CACHE)} entries")
        sys.stdout.flush()
        return good

    # Slow path: re-solve each to repopulate CACHE with P_star tensors.
    # Use shorter maxiters and only Picard (no Anderson) to keep preload fast.
    for i, r in enumerate(good):
        t = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
        g = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
        nr = _nearest(t, g)
        P_init = nr[0] if nr[0] is not None else None
        res = rp.solve_picard_pchip(G, t, g, umax=UMAX, maxiters=400,
                                    abstol=ABSTOL, alpha=1.0, P_init=P_init)
        Finf = float(np.abs(res["residual"]).max())
        # Accept into CACHE if Finf ≤ F_TOL (true fixed-point residual OK).
        if Finf <= F_TOL:
            CACHE.append({"log_tg": _log_tg(t, g),
                          "P_star": res["P_star"].copy(),
                          "taus": t, "gammas": g})
        if (i+1) % 20 == 0 or (i+1) == len(good):
            print(f"  preload {i+1}/{len(good)}  CACHE={len(CACHE)}")
            sys.stdout.flush()
    _save_cache()
    print(f"  [cache saved] {CACHE_PKL}")
    sys.stdout.flush()
    return good


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
    """Predictor step for continuation: extrapolate P* linearly in logit
    space along the SAME continuation axis as the query.

    The cache may contain entries from both a γ-chain (varying γ, fixed τ)
    and a τ-sweep (varying τ, fixed γ). Picking the global 2-nearest can
    mix these and extrapolate along the wrong axis (which caused earlier
    hangs). So:

      - Identify which coordinate of (taus, gammas) differs from most
        cached entries → that's the "axis" we're sweeping.
      - Filter cache to entries whose OTHER coordinates match the query.
      - Use the 2 nearest along the swept axis.
      - Require the query to extend in the same direction as e2 → e1,
        i.e. λ > 0 (extrapolation, not interpolation between them).

    Returns warm-start P or None if constraints not satisfied.
    """
    if len(CACHE) < 2:
        return None
    t = np.asarray(taus, float); g = np.asarray(gammas, float)
    # homogeneous case only: we use τ_1 and γ_1 as scalar axes
    tq, gq = float(t[0]), float(g[0])

    # Filter to same-γ (τ sweep) or same-τ (γ sweep) entries.
    same_g = [e for e in CACHE if np.allclose(e["gammas"], g, atol=1e-12)
              and not np.isclose(e["taus"][0], tq, atol=1e-12)]
    same_t = [e for e in CACHE if np.allclose(e["taus"], t, atol=1e-12)
              and not np.isclose(e["gammas"][0], gq, atol=1e-12)]

    # prefer whichever axis has ≥2 same-axis neighbors
    if len(same_g) >= 2:
        candidates = same_g
        axis_val = lambda e: float(e["taus"][0])
        target = tq
    elif len(same_t) >= 2:
        candidates = same_t
        axis_val = lambda e: float(e["gammas"][0])
        target = gq
    else:
        return None

    # Sort by distance along the swept axis
    candidates.sort(key=lambda e: abs(axis_val(e) - target))
    e1 = candidates[0]; e2 = candidates[1]
    x1 = axis_val(e1); x2 = axis_val(e2)
    if abs(x1 - x2) < 1e-12:
        return None
    # Standard linear interpolation/extrapolation in logit(P):
    #   w = (target - x1) / (x2 - x1)   (0 = at e1, 1 = at e2)
    # Allow w in [-1, 2]: modest extrapolation in either direction +
    # any interpolation. Cap to avoid overshoot when the step is wide.
    w = (target - x1) / (x2 - x1)
    if w < -1.0 or w > 2.0:
        return None

    LP1 = np.log(e1["P_star"] / (1 - e1["P_star"]))
    LP2 = np.log(e2["P_star"] / (1 - e2["P_star"]))
    LP_q = LP1 + w * (LP2 - LP1)
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
    # Seed (empty CACHE): Picard first to land inside the fixed-point basin,
    # then NK to polish to machine precision. Picard is robust from cold at
    # γ=500 (spectral radius tiny), reaching ~1e-12 in a handful of iters.
    # Subsequent configs: NK first (warm-started → quadratic), Anderson as
    # backup.
    if len(CACHE) == 0:
        # Short Picard (land in the basin) → NK (polish to machine precision)
        attempts.append(("P1.0", dict(solver="picard", alpha=1.0, maxiters=200)))
        attempts.append(("NK",   dict(solver="nk")))
        for m in ANDERSON_WINDOWS:
            attempts.append((f"A{m}", dict(solver="anderson", m_window=m,
                                            maxiters=2000)))
    else:
        # Warm-started configs: long Picard (tracks best iterate), Anderson
        # backup. 20000 iters lets Picard ride the linear-convergence curve
        # down to the noise floor even when ρ is close to 1.
        attempts.append(("P1.0", dict(solver="picard", alpha=1.0, maxiters=20000)))
        for m in ANDERSON_WINDOWS:
            attempts.append((f"A{m}", dict(solver="anderson", m_window=m,
                                            maxiters=2000)))

    prefix = (f"τ=({taus[0]:.3f},{taus[1]:.3f},{taus[2]:.3f}) "
              f"γ=({gammas[0]:.3f},{gammas[1]:.3f},{gammas[2]:.3f})")
    for tag, opts in attempts:
        t0 = time.time()
        try:
            if opts["solver"] == "picard":
                res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                            maxiters=opts["maxiters"],
                                            abstol=ABSTOL, alpha=opts["alpha"],
                                            P_init=P_warm,
                                            status_path=STATUS_PATH,
                                            status_every=25,
                                            status_prefix=f"{prefix} | {tag}")
            elif opts["solver"] == "anderson":
                res = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                              maxiters=opts["maxiters"],
                                              abstol=ABSTOL,
                                              m_window=opts["m_window"],
                                              damping=1.0, P_init=P_warm,
                                              status_path=STATUS_PATH,
                                              status_every=25,
                                              status_prefix=f"{prefix} | {tag}")
            else:  # newton-krylov
                res = _solve_nk(taus, gammas, P_warm,
                                status_prefix=f"{prefix} | {tag}")
        except Exception as e:
            print(f"    {tag} error: {e}")
            continue
        dt = time.time() - t0
        PhiI = res["history"][-1] if res["history"] else float("inf")
        Finf = float(np.abs(res["residual"]).max())
        # Non-finite results shouldn't replace a real candidate
        if not (np.isfinite(PhiI) and np.isfinite(Finf)):
            PhiI = float("inf"); Finf = float("inf")
        # Acceptance: Finf (true fixed-point residual ||Φ(P)-P||∞) ≤ F_TOL.
        # PhiI oscillation in stiff regimes doesn't preclude acceptance.
        converged = (Finf < F_TOL)
        cand = dict(alpha=tag, iters=len(res["history"]), time=dt,
                    PhiI=PhiI, Finf=Finf, P_star=res["P_star"],
                    converged=converged,
                    init=("warm" if P_warm is not None else "cold"),
                    warm_from=(t_near, g_near))
        if best is None or (cand["converged"] and not best["converged"]) \
           or (not best["converged"] and cand["Finf"] < best["Finf"]):
            best = cand
        if converged:
            break
        # Chain subsequent attempts: if this attempt reduced Finf below what
        # the warm start would have given, use its P_star as the next warm
        # start. This lets Anderson polish NK's output (or vice versa).
        if np.isfinite(Finf) and Finf < 1.0 and np.all(np.isfinite(res["P_star"])):
            P_warm = np.clip(res["P_star"].copy(), 1e-9, 1 - 1e-9)
    return best


def _solve_nk(taus, gammas, P_init, status_prefix=""):
    """Newton-Krylov on F(P)=P-Φ(P). Uses no-learning seed when no warm start.
    Writes live status (outer iteration, current ||F||∞) to STATUS_PATH."""
    u = np.linspace(-UMAX, UMAX, G)
    taus_v = rh._as_vec3(taus[0]) if len(set(taus))==1 else np.asarray(taus, float)
    gammas_v = rh._as_vec3(gammas[0]) if len(set(gammas))==1 else np.asarray(gammas, float)
    Ws = rh._as_vec3(1.0)
    if P_init is None:
        P_init = rh._nolearning_price(u, taus_v, gammas_v, Ws)

    def F(x):
        P = x.reshape(G, G, G)
        Pn = rp._phi_map_pchip(P, u, taus_v, gammas_v, Ws)
        return x - Pn.reshape(-1)

    t_start = time.time()
    nk_iter = [0]
    def cb(xk, fk):
        nk_iter[0] += 1
        try:
            Finf_k = float(np.abs(fk).max())
            with open(STATUS_PATH, "w") as _sf:
                _sf.write(f"{status_prefix} NK iter={nk_iter[0]} "
                          f"Finf={Finf_k:.3e} elapsed={time.time()-t_start:.1f}s\n")
        except Exception:
            pass

    x0 = np.clip(P_init, 1e-9, 1-1e-9).reshape(-1)

    def _run_nk(start_vec):
        try:
            s = newton_krylov(F, start_vec, f_tol=ABSTOL, rdiff=1e-8,
                               method="lgmres", maxiter=80, verbose=False,
                               callback=cb)
        except NoConvergence as e:
            s = np.asarray(e.args[0])
        if not np.all(np.isfinite(s)):
            s = start_vec
        return s

    sol = _run_nk(x0)
    P_star = np.clip(sol.reshape(G, G, G), 1e-9, 1 - 1e-9)
    Pn = rp._phi_map_pchip(P_star, u, taus_v, gammas_v, Ws)
    Finf = float(np.abs(P_star - Pn).max())
    PhiI = Finf
    best = dict(P=P_star.copy(), Finf=Finf)

    # If NK didn't reach ABSTOL, perturb in LOGIT space by σ=1e-10 and retry.
    # The noise dislodges the FD-Jacobian from a spurious noise floor without
    # changing the actual solution meaningfully.
    rng = np.random.default_rng(12345)
    for attempt in range(3):
        if best["Finf"] < ABSTOL:
            break
        L = np.log(best["P"] / (1.0 - best["P"]))
        sigma = 1e-10 * (10 ** attempt)  # 1e-10, 1e-9, 1e-8
        L_perturbed = L + rng.standard_normal(L.shape) * sigma
        P_perturbed = 1.0 / (1.0 + np.exp(-L_perturbed))
        nk_iter[0] = 0  # reset the status counter for callback reporting
        sol = _run_nk(P_perturbed.reshape(-1))
        P_try = np.clip(sol.reshape(G, G, G), 1e-9, 1 - 1e-9)
        Pn_try = rp._phi_map_pchip(P_try, u, taus_v, gammas_v, Ws)
        F_try = float(np.abs(P_try - Pn_try).max())
        if F_try < best["Finf"]:
            best = dict(P=P_try, Finf=F_try)

    return {"history": [best["Finf"]], "residual": (best["P"] -
            rp._phi_map_pchip(best["P"], u, taus_v, gammas_v, Ws)),
            "P_star": best["P"]}


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
        # Dense near pure-CARA so each γ step is small (good warm start)
        [500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 120.0, 100.0, 80.0, 60.0]
        + list(np.arange(50.0, 1.0 - 1e-9, -1.0))        # 50, 49, ..., 2, 1
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

    u = np.linspace(-UMAX, UMAX, G)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(1.0), idx(-1.0), idx(1.0)

    fieldnames = ["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
                  "alpha","iters","time_s","PhiI","Finf","oneR2_het",
                  "p_star","mu_1","mu_2","mu_3","pr_gap","converged","init"]

    # Preload CACHE + rows from the snapshot of the previous run.
    rows = _preload_from_csv(CSV_OUT, fieldnames)

    # If the seed is absent, solve it now (cold start).
    seed_present = any(
        abs(float(r["tau_1"]) - TAU) < 1e-9 and abs(float(r["gamma_1"]) - 500.0) < 1e-9
        for r in rows)
    if not seed_present:
        print(f"[seed] solving τ=({TAU},{TAU},{TAU}) γ=(500,500,500)")
        sys.stdout.flush()
        seed = solve_one((TAU, TAU, TAU), (500.0, 500.0, 500.0))
        if seed.get("Finf") is not None and np.isfinite(seed["Finf"]) \
           and seed["Finf"] < 1e-6:
            CACHE.append({"log_tg": _log_tg((TAU,TAU,TAU), (500,500,500)),
                          "P_star": seed["P_star"].copy(),
                          "taus": (TAU,TAU,TAU), "gammas": (500,500,500)})
            print(f"  seed added to CACHE: iters={seed['iters']} "
                  f"PhiI={seed['PhiI']:.2e} Finf={seed['Finf']:.2e} "
                  f"conv={seed['converged']}")
        else:
            print(f"  SEED FAILED: PhiI={seed['PhiI']:.2e} Finf={seed['Finf']:.2e}")
        sys.stdout.flush()
    else:
        seed = None
        print(f"[seed] already in preload")
        sys.stdout.flush()

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
        record((TAU,TAU,TAU), (500,500,500), seed)
        flush()

    # Walk homogeneous γ downward (warm-start chain 50 → 49 → … → 3)
    print(f"\n=== homogeneous γ sweep (τ={TAU} fixed) ===")
    sys.stdout.flush()
    for (t, g) in gamma_sweep():
        if g == (500.0, 500.0, 500.0) and rows:
            continue
        # stop at γ=3, we'll pivot into the τ sweep
        if g[0] < GAMMA - 1e-9:
            break
        already = any((abs(float(r["tau_1"]) - t[0]) < 1e-9
                       and abs(float(r["gamma_1"]) - g[0]) < 1e-9) for r in rows)
        if already:
            continue
        best = solve_one(t, g)
        record(t, g, best)
        flush()
        # Add to CACHE when close enough to be a useful warm start,
        # even if F_TOL check fails. Chain continuity matters more than
        # strict convergence for the sweep.
        if (best.get("Finf") is not None and np.isfinite(best["Finf"])
            and best["Finf"] < 1e-6):
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
        # skip configs already present in rows (preload + prior progress)
        already = any((abs(float(r["tau_1"]) - t[0]) < 1e-9
                       and abs(float(r["gamma_1"]) - g[0]) < 1e-9) for r in rows)
        if already:
            continue
        best = solve_one(t, g)
        record(t, g, best)
        flush()
        # Add to CACHE when close enough to be a useful warm start,
        # even if F_TOL check fails. Chain continuity matters more than
        # strict convergence for the sweep.
        if (best.get("Finf") is not None and np.isfinite(best["Finf"])
            and best["Finf"] < 1e-6):
            CACHE.append({"log_tg": _log_tg(t, g),
                          "P_star": best["P_star"].copy(),
                          "taus": t, "gammas": g})
            _save_cache()
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
