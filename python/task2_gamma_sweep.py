"""Task 2: γ-sweep at G=20 UMAX=5 τ=2 mp50.

Per origin/main:FIGURES_TODO.md (UPDATED 2026-05-02), Task 2.

Warm-start chain:
  γ=0.5 seed → γ=1.0 → γ=2.0 → γ=4.0          (walk up)
  γ=0.5 seed → γ=0.25 → γ=0.1                  (walk down)

Tolerance: ||F||∞ < 1e-25 at mp50 (50 decimal digits).

Outputs:
- results/full_ree/posterior_v3_G20_umax5_g{NNN}_mp50.json  (one per γ)
- results/full_ree/fig4B_G20_gamma_sweep.json              (summary)
- results/full_ree/fig4B_G20_pgfplots.tex                  (plot data)

Encoding: γ=0.5 → g050, γ=1.0 → g100, γ=0.25 → g025, etc.
(int(round(gamma*100, 0)).
"""
import time, json, warnings
import numpy as np
import mpmath
from mpmath import mp, mpf
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 50

import sys
sys.path.insert(0, "python")
from posterior_method_v3 import init_p_grid as init_p_grid_f64

RESULTS_DIR = "results/full_ree"
SEED = f"{RESULTS_DIR}/posterior_v3_G20_umax5_notrim_mp300.json"
G = 20
UMAX = 5.0
TAU = mpf("2")
TRIM = 0.0          # match seed
H_FD = mpf("1e-30")
TARGET = mpf("1e-25")
MAX_ITERS = 12

# Warm-start chain: (gamma, source_gamma_or_None)
CHAIN = [
    (mpf("0.5"),  None),         # seed
    (mpf("1"),    mpf("0.5")),
    (mpf("2"),    mpf("1")),
    (mpf("4"),    mpf("2")),
    (mpf("0.25"), mpf("0.5")),
    (mpf("0.1"),  mpf("0.25")),
]

# All γ values for the figure (in plot order)
PAPER_GAMMAS = [mpf("0.1"), mpf("0.25"), mpf("0.5"), mpf("1"), mpf("2"), mpf("4")]


def gamma_tag(gamma):
    return f"g{int(round(float(gamma)*100)):03d}"


# -------- mp50 utilities --------

def Lam_mp(z):
    if z >= 0: return mpf(1) / (mpf(1) + mpmath.exp(-z))
    e = mpmath.exp(z); return e / (mpf(1) + e)


def logit_mp(p):
    return mpmath.log(p / (mpf(1) - p))


def crra_demand_mp(mu, p, gamma):
    z = (logit_mp(mu) - logit_mp(p)) / gamma
    R = mpmath.exp(z)
    return (R - mpf(1)) / ((mpf(1) - p) + R * p)


def f_v_mp(u, v, tau):
    mean = mpf(v) - mpf("0.5")
    return mpmath.sqrt(tau / (mpf(2) * mpmath.pi)) * mpmath.exp(
        -tau / mpf(2) * (u - mean) ** 2)


def interp_mp(x_target, x_arr, y_arr):
    n = len(x_arr)
    if x_target <= x_arr[0]: return y_arr[0]
    if x_target >= x_arr[-1]: return y_arr[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_arr[mid] > x_target: hi = mid
        else: lo = mid
    x0, x1 = x_arr[lo], x_arr[lo + 1]
    y0, y1 = y_arr[lo], y_arr[lo + 1]
    if x1 == x0: return y0
    w = (x_target - x0) / (x1 - x0)
    return (mpf(1) - w) * y0 + w * y1


def phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    """One Φ step at mp precision. Returns mu_new (G x G nested list)."""
    EPS_MP = mpf("1e-25")
    mu_new = [[mu[i][j] for j in range(G)] for i in range(G)]
    f1_grid = [f_v_mp(u_grid[i], 1, tau) for i in range(G)]
    f0_grid = [f_v_mp(u_grid[i], 0, tau) for i in range(G)]
    for i in range(G):
        for j in range(G):
            p0 = p_grid[i][j]
            mu_col = []
            for ii in range(G):
                if p0 < p_grid[ii][0]: val = mu[ii][0]
                elif p0 > p_grid[ii][-1]: val = mu[ii][-1]
                else: val = interp_mp(p0, p_grid[ii], mu[ii])
                if val < EPS_MP: val = EPS_MP
                if val > mpf(1) - EPS_MP: val = mpf(1) - EPS_MP
                mu_col.append(val)
            d = [crra_demand_mp(mu_col[ii], p0, gamma) for ii in range(G)]
            if abs(d[-1] - d[0]) < mpf("1e-25"): continue
            D_i = -d[i]
            targets = [D_i - d[ii] for ii in range(G)]
            d_inc = d[-1] > d[0]
            d_arr = d if d_inc else list(reversed(d))
            u_arr = u_grid if d_inc else list(reversed(u_grid))
            u3_star = []; valid_mask = []
            for ii in range(G):
                if targets[ii] < d_arr[0] or targets[ii] > d_arr[-1]:
                    u3_star.append(None); valid_mask.append(False)
                else:
                    u3 = interp_mp(targets[ii], d_arr, u_arr)
                    u3_star.append(u3)
                    valid_mask.append(u_grid[0] <= u3 <= u_grid[-1])
            valid = [k for k in range(G) if valid_mask[k]]
            if len(valid) < 2: continue
            f1_root = [f_v_mp(u3_star[ii], 1, tau) for ii in valid]
            f0_root = [f_v_mp(u3_star[ii], 0, tau) for ii in valid]
            f1_sweep = [f1_grid[ii] for ii in valid]
            f0_sweep = [f0_grid[ii] for ii in valid]
            A1 = sum(f1_sweep[k] * f1_root[k] for k in range(len(valid)))
            A0 = sum(f0_sweep[k] * f0_root[k] for k in range(len(valid)))
            f1_own = f1_grid[i]; f0_own = f0_grid[i]
            denom = f0_own * A0 + f1_own * A1
            if denom <= 0: continue
            new_val = f1_own * A1 / denom
            if new_val < EPS_MP: new_val = EPS_MP
            if new_val > mpf(1) - EPS_MP: new_val = mpf(1) - EPS_MP
            mu_new[i][j] = new_val
    return mu_new


def F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    return [[cand[i][j] - mu[i][j] for j in range(G)] for i in range(G)]


def F_max(F):
    return max(abs(F[i][j]) for i in range(G) for j in range(G))


def F_med(F):
    vals = sorted(abs(F[i][j]) for i in range(G) for j in range(G))
    return vals[len(vals) // 2]


def lm_step(J, F_flat, n, lam):
    JT = J.T
    JTJ = JT * J
    for k in range(n):
        JTJ[k, k] = JTJ[k, k] + lam
    JTF = JT * mpmath.matrix([-F_flat[k] for k in range(n)])
    return mpmath.lu_solve(JTJ, JTF)


def nk_solve(mu, u_grid, p_grid, p_lo, p_hi, gamma, tag):
    """LM NK with Picard fallback. Returns (mu, history)."""
    history = []
    lam = mpf("1e-15")  # smaller starting λ for mp50
    for nk_iter in range(1, MAX_ITERS + 1):
        t_iter = time.time()
        F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
        F_max_c = F_max(F_curr)
        F_med_c = F_med(F_curr)
        F_norm_c = sum(abs(F_curr[i][j])**2 for i in range(G)
                         for j in range(G))
        print(f"  [{tag}] iter {nk_iter}: max={mpmath.nstr(F_max_c, 4)}, "
              f"med={mpmath.nstr(F_med_c, 4)}", flush=True)
        if F_max_c < TARGET:
            print(f"  [{tag}] target reached", flush=True)
            history.append({"iter": nk_iter, "F_max": mpmath.nstr(F_max_c, 30),
                              "F_med": mpmath.nstr(F_med_c, 30)})
            return mu, F_max_c, F_med_c, history
        n = G * G
        F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
        # Build Jacobian
        t_jac = time.time()
        J = mpmath.zeros(n, n)
        for col in range(n):
            i, j = col // G, col % G
            mu_pert = [row[:] for row in mu]
            mu_pert[i][j] = mu_pert[i][j] + H_FD
            F_pert = F_mu(mu_pert, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
            for row_idx in range(n):
                ii, jj = row_idx // G, row_idx % G
                J[row_idx, col] = (F_pert[ii][jj] - F_curr[ii][jj]) / H_FD
        t_jac_done = time.time()
        # LM with up to 8 attempts
        accepted = False
        for attempt in range(8):
            try:
                delta = lm_step(J, F_flat, n, lam)
            except (ZeroDivisionError, mpmath.libmp.libelefun.NoConvergence):
                lam = lam * mpf(100); continue
            mu_trial = [row[:] for row in mu]
            for k in range(n):
                i, j = k // G, k % G
                mu_trial[i][j] = mu_trial[i][j] + delta[k]
                if mu_trial[i][j] < mpf("1e-50"): mu_trial[i][j] = mpf("1e-50")
                if mu_trial[i][j] > mpf(1) - mpf("1e-50"):
                    mu_trial[i][j] = mpf(1) - mpf("1e-50")
            F_try = F_mu(mu_trial, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
            F_norm_try = sum(abs(F_try[i][j])**2 for i in range(G)
                                for j in range(G))
            if F_norm_try < F_norm_c:
                mu = mu_trial
                lam = lam / mpf(10)
                accepted = True
                break
            else:
                lam = lam * mpf(10)
        if not accepted:
            # Picard fallback
            print(f"  [{tag}] LM stuck — applying damped Picard α=0.1",
                  flush=True)
            cand_p = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
            mu = [[mu[i][j] + mpf("0.1") * (cand_p[i][j] - mu[i][j])
                     for j in range(G)] for i in range(G)]
            lam = mpf("1e-15")
        F_after = F_mu(mu, u_grid, p_grid, p_lo, p_hi, TAU, gamma)
        F_max_a = F_max(F_after)
        F_med_a = F_med(F_after)
        elapsed = time.time() - t_iter
        print(f"  [{tag}] after iter {nk_iter}: max={mpmath.nstr(F_max_a, 4)}, "
              f"med={mpmath.nstr(F_med_a, 4)}, t={elapsed:.0f}s "
              f"(jac {(t_jac_done-t_jac):.0f}s)", flush=True)
        history.append({"iter": nk_iter,
                          "F_max": mpmath.nstr(F_max_a, 30),
                          "F_med": mpmath.nstr(F_med_a, 30),
                          "elapsed_s": elapsed})
    return mu, F_max_a, F_med_a, history


# -------- Interp from one γ result onto another's p-grid --------

def interp_mu(mu_old, u_old, p_old, p_new):
    """Both grids share the same u_grid. p_old and p_new are per-row.
    Interp mu along each row's p-grid.
    """
    G = len(u_old)
    out = [[mpf(0)] * G for _ in range(G)]
    for i in range(G):
        for j in range(G):
            p = p_new[i][j]
            if p <= p_old[i][0]: out[i][j] = mu_old[i][0]
            elif p >= p_old[i][-1]: out[i][j] = mu_old[i][-1]
            else: out[i][j] = interp_mp(p, p_old[i], mu_old[i])
    return out


# -------- 1-R² measurement (float64) --------

def measure_R2_float(mu_f, u_grid_f, p_grid_f, gamma_f, tau_f):
    """Measure 1-R² over all G³ triples. mu/grids in float64.

    p_REE solved via brentq market clearing.
    """
    def Lam(z):
        z = float(z)
        if z >= 0: return 1.0/(1.0 + np.exp(-z))
        e = np.exp(z); return e/(1.0+e)

    def logit_f(p):
        p = float(np.clip(p, 1e-15, 1-1e-15))
        return float(np.log(p/(1-p)))

    def crra_d_f(mu, p, gamma):
        z = (logit_f(mu) - logit_f(p))/gamma
        R = float(np.exp(z))
        return (R-1.0)/((1.0-p) + R*p)

    def mu_at(u, p):
        if u <= u_grid_f[0]: ia = ib = 0; w = 0.0
        elif u >= u_grid_f[-1]: ia = ib = G-1; w = 0.0
        else:
            ib = int(np.searchsorted(u_grid_f, u))
            ia = ib-1
            w = (u - u_grid_f[ia])/(u_grid_f[ib] - u_grid_f[ia])

        def row(i):
            pr = p_grid_f[i]
            if p <= pr[0]: return mu_f[i,0]
            if p >= pr[-1]: return mu_f[i,-1]
            return float(np.interp(p, pr, mu_f[i]))

        if ia == ib: return row(ia)
        return (1-w)*row(ia) + w*row(ib)

    Tstar = []; lp = []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u = (u_grid_f[i], u_grid_f[j], u_grid_f[l])
                def F(p):
                    return sum(crra_d_f(mu_at(uk, p), p, gamma_f)
                                  for uk in u)
                p_lo, p_hi = 1e-4, 1-1e-4
                f_lo, f_hi = F(p_lo), F(p_hi)
                if f_lo*f_hi > 0:
                    p_e = p_lo if abs(f_lo) < abs(f_hi) else p_hi
                else:
                    p_e = brentq(F, p_lo, p_hi, xtol=1e-12)
                if 1e-10 < p_e < 1-1e-10:
                    Tstar.append(tau_f*(u[0]+u[1]+u[2]))
                    lp.append(logit_f(p_e))
    Tstar = np.array(Tstar); lp = np.array(lp)
    b, a = np.polyfit(Tstar, lp, 1)
    pred = a + b*Tstar
    ss_res = float(np.sum((lp - pred)**2))
    ss_tot = float(np.sum((lp - lp.mean())**2))
    R2 = 1 - ss_res/ss_tot
    return 1-R2, float(b), int(len(Tstar))


def measure_no_learning(u_grid_f, gamma_f, tau_f):
    """No-learning 1-R²: μ_k = Λ(τ u_k), market-clear, regress."""
    def Lam(z):
        z = float(z)
        if z >= 0: return 1.0/(1.0 + np.exp(-z))
        e = np.exp(z); return e/(1.0+e)

    def logit_f(p):
        p = float(np.clip(p, 1e-15, 1-1e-15))
        return float(np.log(p/(1-p)))

    def crra_d_f(mu, p, gamma):
        z = (logit_f(mu) - logit_f(p))/gamma
        R = float(np.exp(z))
        return (R-1.0)/((1.0-p) + R*p)

    Tstar = []; lp = []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u = (u_grid_f[i], u_grid_f[j], u_grid_f[l])
                priors = [Lam(tau_f*uk) for uk in u]
                def F(p):
                    return sum(crra_d_f(mk, p, gamma_f) for mk in priors)
                f_lo = F(1e-4); f_hi = F(1-1e-4)
                if f_lo*f_hi > 0:
                    p_e = 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
                else:
                    p_e = brentq(F, 1e-4, 1-1e-4, xtol=1e-12)
                if 1e-10 < p_e < 1-1e-10:
                    Tstar.append(tau_f*(u[0]+u[1]+u[2]))
                    lp.append(logit_f(p_e))
    Tstar = np.array(Tstar); lp = np.array(lp)
    b, a = np.polyfit(Tstar, lp, 1)
    pred = a + b*Tstar
    ss_res = float(np.sum((lp - pred)**2))
    ss_tot = float(np.sum((lp - lp.mean())**2))
    return 1-(1-ss_res/ss_tot), float(b)


def to_floats(mu_mp):
    return np.array([[float(mu_mp[i][j]) for j in range(G)]
                          for i in range(G)])


# -------- Main: warm-start chain --------

def load_seed():
    print(f"Loading seed {SEED} (mp300 → mp50)...")
    with open(SEED) as f:
        d = json.load(f)
    u = [mpf(s) for s in d["u_grid"]]
    p = [[mpf(s) for s in row] for row in d["p_grid"]]
    mu_seed = [[mpf(s) for s in row] for row in d["mu_strings"]]
    return u, p, mu_seed


def build_grid(u_grid_np):
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, 2.0,
                                                       0.5, G, trim=TRIM)
    p_grid_mp = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo_mp = [mpf(str(x)) for x in p_lo_np]
    p_hi_mp = [mpf(str(x)) for x in p_hi_np]
    return p_lo_np, p_hi_np, p_grid_np, p_lo_mp, p_hi_mp, p_grid_mp


def save_ckpt(gamma, mu_mp, u_grid_np, p_grid_np, F_max_v, F_med_v, history):
    tag = gamma_tag(gamma)
    out = f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_{tag}_mp50.json"
    mu_strs = [[mpmath.nstr(mu_mp[i][j], 40) for j in range(G)]
                for i in range(G)]
    rec = {
        "G": G, "UMAX": UMAX, "tau": float(TAU), "gamma": float(gamma),
        "trim": TRIM, "dps": 50,
        "F_max": mpmath.nstr(F_max_v, 30),
        "F_med": mpmath.nstr(F_med_v, 30),
        "u_grid": [str(x) for x in u_grid_np],
        "p_grid": [[str(p) for p in row] for row in p_grid_np],
        "mu_strings": mu_strs,
        "history": history,
    }
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"  Saved {out}")


def main():
    t_start = time.time()
    u_seed, p_seed, mu_seed = load_seed()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    p_lo_np, p_hi_np, p_grid_np, p_lo_mp, p_hi_mp, p_grid_mp = \
        build_grid(u_grid_np)

    # Cache: gamma_str -> (mu_mp, p_grid_mp, p_grid_np)
    cache = {}

    # Seed: γ=0.5 — use directly (no convergence run, just cast)
    print(f"\n=== γ=0.5 (seed) ===")
    cache["0.5"] = (mu_seed, p_seed, p_grid_np)
    # Measure 1-R² of seed
    mu_f = to_floats(mu_seed)
    one_R2, slope, n = measure_R2_float(mu_f, u_grid_np,
                                            np.array([[float(x) for x in row]
                                                          for row in p_seed]),
                                            0.5, 2.0)
    print(f"  Seed 1-R² = {one_R2:.6e}, slope = {slope:.6f}, n={n}")
    seed_metrics = {"1-R2": one_R2, "slope": slope, "n_triples": n}

    # Walk through chain
    results = {0.5: {"1-R2": one_R2, "slope": slope, "n_triples": n,
                        "F_max": "0", "F_med": "0", "iters": 0,
                        "elapsed_s": 0.0}}

    for gamma_mp, src_mp in CHAIN:
        if src_mp is None: continue   # seed already cached
        gamma_f = float(gamma_mp); src_f = float(src_mp)
        tag = f"γ={gamma_f}"
        print(f"\n=== {tag} (warm from γ={src_f}) ===")
        t0 = time.time()
        # Warm start: interp source μ onto current p-grid
        mu_src, p_src, _ = cache[str(src_f)]
        mu_init = interp_mu(mu_src, u_seed, p_src, p_grid_mp)

        # Solve
        mu_conv, F_max_v, F_med_v, history = nk_solve(
            mu_init, u_seed, p_grid_mp, p_lo_mp, p_hi_mp, gamma_mp, tag)

        # Measure
        mu_f = to_floats(mu_conv)
        one_R2, slope, n = measure_R2_float(mu_f, u_grid_np, p_grid_np,
                                                 gamma_f, 2.0)
        print(f"  {tag}: 1-R² = {one_R2:.6e}, slope = {slope:.6f}")

        # Save ckpt
        save_ckpt(gamma_mp, mu_conv, u_grid_np, p_grid_np,
                       F_max_v, F_med_v, history)

        # Cache for downstream warm starts
        cache[str(gamma_f)] = (mu_conv, p_grid_mp, p_grid_np)

        elapsed = time.time() - t0
        results[gamma_f] = {
            "1-R2": one_R2, "slope": slope, "n_triples": n,
            "F_max": mpmath.nstr(F_max_v, 20),
            "F_med": mpmath.nstr(F_med_v, 20),
            "iters": len(history),
            "elapsed_s": elapsed,
        }

    # No-learning measurements at the same γ values
    print(f"\n=== No-learning 1-R² ===")
    nl_results = {}
    for gamma_mp in PAPER_GAMMAS:
        gamma_f = float(gamma_mp)
        nl_R2, nl_slope = measure_no_learning(u_grid_np, gamma_f, 2.0)
        nl_results[gamma_f] = {"1-R2": nl_R2, "slope": nl_slope}
        print(f"  γ={gamma_f}: NL 1-R² = {nl_R2:.6e}, slope = {nl_slope:.6f}")

    # Build summary
    summary = {
        "figure": "fig4B",
        "params": {"G": G, "tau": float(TAU), "umax": UMAX},
        "REE": [
            {"gamma": float(g), **results[float(g)]}
            for g in PAPER_GAMMAS
        ],
        "no_learning": [
            {"gamma": float(g), **nl_results[float(g)]}
            for g in PAPER_GAMMAS
        ],
    }
    with open(f"{RESULTS_DIR}/fig4B_G20_gamma_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/fig4B_G20_gamma_sweep.json")

    # pgfplots
    out = f"{RESULTS_DIR}/fig4B_G20_pgfplots.tex"
    lines = [f"% Fig 4B 1-R² vs γ (G={G}, τ={float(TAU)}, UMAX={UMAX})",
              "% REE (full information aggregation)"]
    ree_pts = "".join(f"({float(g)},{results[float(g)]['1-R2']:.6e})"
                            for g in PAPER_GAMMAS)
    lines.append(f"\\addplot coordinates {{{ree_pts}}};")
    lines += ["", "% No-learning"]
    nl_pts = "".join(f"({float(g)},{nl_results[float(g)]['1-R2']:.6e})"
                            for g in PAPER_GAMMAS)
    lines.append(f"\\addplot coordinates {{{nl_pts}}};")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
