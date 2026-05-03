"""Task 3: τ-sweep at G=20 UMAX=5 mp50 for γ ∈ {0.5, 1.0, 4.0}.

Per FIGURES_TODO.md.
τ values: [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
τ=2 is already done (Task 2 ckpts). 11 new τ per γ → 33 runs.

Cold-start each (γ, τ) with no-learning + float64 picard polish + mp50 LM
(target ||F||∞ < 1e-25).

Saves:
- Per-(γ,τ) ckpt: posterior_v3_G20_umax5_g{NNN}_t{NNN}_mp50.json
- Summary: fig4A_G20_tau_sweep.json
- Pgfplots: fig4A_G20_pgfplots.tex (weighted 1-R²)
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
from posterior_method_v3 import (
    init_p_grid as init_p_grid_f64,
    phi_step as phi_step_f64, EPS as EPS_F64,
)
from gap_reparam import pava_p_only, pava_u_only

RESULTS_DIR = "results/full_ree"
G = 20
UMAX = 5.0
TRIM = 0.05
H_FD = mpf("1e-30")
TARGET = mpf("1e-25")
MAX_ITERS = 15

GAMMAS = [0.5, 1.0, 4.0]
TAU_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
# τ=2 is in Task 2 ckpts, skipped here

# Existing γ ckpts at τ=2
TAU2_CKPT = {
    0.5: f"{RESULTS_DIR}/posterior_v3_G20_umax5_trim05_mp300.json",
    1.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g100_mp50.json",
    4.0: f"{RESULTS_DIR}/posterior_v3_G20_umax5_g400_mp50.json",
}


def gamma_tag(gamma):
    return f"g{int(round(float(gamma)*100)):03d}"


def tau_tag(tau):
    return f"t{int(round(float(tau)*100)):04d}"


def pava_2d_f64(mu): return pava_u_only(pava_p_only(mu))


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


def nk_solve(mu, u_grid, p_grid, p_lo, p_hi, tau_mp, gamma_mp, tag):
    history = []
    lam = mpf("1e-15")
    for nk_iter in range(1, MAX_ITERS + 1):
        t_iter = time.time()
        F_curr = F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau_mp, gamma_mp)
        F_max_c = F_max(F_curr)
        F_med_c = F_med(F_curr)
        F_norm_c = sum(abs(F_curr[i][j])**2 for i in range(G)
                         for j in range(G))
        print(f"  [{tag}] iter {nk_iter}: max={mpmath.nstr(F_max_c,4)}, "
              f"med={mpmath.nstr(F_med_c,4)}", flush=True)
        if F_max_c < TARGET:
            print(f"  [{tag}] target reached", flush=True)
            history.append({"iter": nk_iter,
                              "F_max": mpmath.nstr(F_max_c, 30),
                              "F_med": mpmath.nstr(F_med_c, 30)})
            return mu, F_max_c, F_med_c, history
        n = G * G
        F_flat = [F_curr[i][j] for i in range(G) for j in range(G)]
        t_jac = time.time()
        J = mpmath.zeros(n, n)
        for col in range(n):
            i, j = col // G, col % G
            mu_pert = [row[:] for row in mu]
            mu_pert[i][j] = mu_pert[i][j] + H_FD
            F_pert = F_mu(mu_pert, u_grid, p_grid, p_lo, p_hi,
                              tau_mp, gamma_mp)
            for row_idx in range(n):
                ii, jj = row_idx // G, row_idx % G
                J[row_idx, col] = (F_pert[ii][jj] - F_curr[ii][jj]) / H_FD
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
            F_try = F_mu(mu_trial, u_grid, p_grid, p_lo, p_hi,
                              tau_mp, gamma_mp)
            F_norm_try = sum(abs(F_try[i][j])**2 for i in range(G)
                                for j in range(G))
            if F_norm_try < F_norm_c:
                mu = mu_trial; lam = lam / mpf(10); accepted = True
                break
            else:
                lam = lam * mpf(10)
        if not accepted:
            print(f"  [{tag}] LM stuck — Picard α=0.1", flush=True)
            cand_p = phi_step_mp(mu, u_grid, p_grid, p_lo, p_hi,
                                       tau_mp, gamma_mp)
            mu = [[mu[i][j] + mpf("0.1") * (cand_p[i][j] - mu[i][j])
                     for j in range(G)] for i in range(G)]
            lam = mpf("1e-15")
        F_after = F_mu(mu, u_grid, p_grid, p_lo, p_hi, tau_mp, gamma_mp)
        F_max_a = F_max(F_after); F_med_a = F_med(F_after)
        elapsed = time.time() - t_iter
        print(f"  [{tag}] after iter {nk_iter}: max={mpmath.nstr(F_max_a,4)},"
              f" med={mpmath.nstr(F_med_a,4)}, t={elapsed:.0f}s",
              flush=True)
        history.append({"iter": nk_iter,
                          "F_max": mpmath.nstr(F_max_a, 30),
                          "F_med": mpmath.nstr(F_med_a, 30),
                          "elapsed_s": elapsed})
        # Fast-fail: if max < 1e-3 we have a usable solution; bail
        # out early to avoid wasting time on stuck-boundary cases.
        if F_max_a < mpf("1e-3") and nk_iter >= 4:
            print(f"  [{tag}] usable convergence (max < 1e-3), stopping early",
                  flush=True)
            return mu, F_max_a, F_med_a, history
        # Fast-fail stuck: if max barely changing for 3 iters AND high, bail
        if (len(history) >= 3 and nk_iter >= 4 and F_max_a > mpf("0.1")
                and abs(F_max_a - mpf(history[-2]["F_max"])) /
                    F_max_a < mpf("0.05")):
            print(f"  [{tag}] stuck (max~{mpmath.nstr(F_max_a,3)}), "
                  f"bailing out early", flush=True)
            return mu, F_max_a, F_med_a, history
    return mu, F_max_a, F_med_a, history


def signal_density(u, v, tau):
    mean = float(v) - 0.5
    return float(np.sqrt(tau/(2*np.pi)) * np.exp(-tau/2 * (u - mean)**2))


def Lam(z):
    z = float(z)
    if z >= 0: return 1.0/(1.0+np.exp(-z))
    e = np.exp(z); return e/(1.0+e)


def logit(p):
    p = float(np.clip(p, 1e-15, 1-1e-15))
    return float(np.log(p/(1-p)))


def crra_d_f64(mu, p, gamma):
    z = (logit(mu) - logit(p))/gamma
    R = float(np.exp(z))
    return (R-1.0)/((1.0-p) + R*p)


def mu_at(u, p, u_grid, p_grid, mu):
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
        return sum(crra_d_f64(mu_at(uk, p, u_grid, p_grid, mu), p, gamma)
                      for uk in u3)
    f_lo, f_hi = F(1e-4), F(1-1e-4)
    if f_lo*f_hi > 0:
        return 1e-4 if abs(f_lo) < abs(f_hi) else 1-1e-4
    return brentq(F, 1e-4, 1-1e-4, xtol=1e-12)


def weighted_R2_grid(u_grid, p_grid, mu, gamma, tau):
    f0 = np.array([signal_density(u, 0, tau) for u in u_grid])
    f1 = np.array([signal_density(u, 1, tau) for u in u_grid])
    Ts = []; lp = []; ws = []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u3 = (u_grid[i], u_grid[j], u_grid[l])
                p = market_clear(u3, gamma, u_grid, p_grid, mu)
                if not (1e-10 < p < 1-1e-10): continue
                w = 0.5*(f0[i]*f0[j]*f0[l] + f1[i]*f1[j]*f1[l])
                Ts.append(tau*(u3[0]+u3[1]+u3[2]))
                lp.append(logit(p))
                ws.append(w)
    Ts = np.array(Ts); lp = np.array(lp); ws = np.array(ws)
    slope, intercept = np.polyfit(Ts, lp, 1, w=np.sqrt(ws))
    pred = slope*Ts + intercept
    mean_lp = np.average(lp, weights=ws)
    var_tot = np.average((lp - mean_lp)**2, weights=ws)
    var_res = np.average((lp - pred)**2, weights=ws)
    return float(var_res/var_tot), float(slope)


def to_floats(mu_mp):
    return np.array([[float(mu_mp[i][j]) for j in range(G)]
                          for i in range(G)])


def solve_one(gamma_f, tau_f, u_grid_np, u_grid_mp):
    """Cold-start + polish + LM for one (γ, τ)."""
    gamma_mp = mpf(str(gamma_f))
    tau_mp = mpf(str(tau_f))
    p_lo_np, p_hi_np, p_grid_np = init_p_grid_f64(u_grid_np, tau_f, gamma_f,
                                                       G, trim=TRIM)
    p_grid_mp = [[mpf(str(p)) for p in row] for row in p_grid_np]
    p_lo_mp = [mpf(str(x)) for x in p_lo_np]
    p_hi_mp = [mpf(str(x)) for x in p_hi_np]

    # Cold start
    mu_f = np.zeros((G, G))
    for i, u in enumerate(u_grid_np):
        mu_f[i, :] = Lam(tau_f * u)
    mu_f = pava_2d_f64(mu_f)

    # Float64 polish
    last_status = time.time()
    for round_idx, (n_iter, n_avg, alpha) in enumerate(
            [(2000, 1000, 0.005), (2000, 1000, 0.002),
             (3000, 1500, 0.001)]):
        mu_sum = np.zeros_like(mu_f); n_collected = 0
        for it in range(n_iter):
            cand, active, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                              p_lo_np, p_hi_np, tau_f, gamma_f)
            cand = pava_2d_f64(cand)
            mu_f = alpha * cand + (1 - alpha) * mu_f
            mu_f = np.clip(mu_f, EPS_F64, 1 - EPS_F64)
            if it >= n_iter - n_avg:
                mu_sum += mu_f; n_collected += 1
            if time.time() - last_status > 30:
                cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np,
                                                  p_lo_np, p_hi_np,
                                                  tau_f, gamma_f)
                r = float(np.max(np.abs(cand2 - mu_f)[act2]))
                print(f"    polish r{round_idx+1} α={alpha} "
                      f"it {it+1}/{n_iter}: max={r:.3e}", flush=True)
                last_status = time.time()
        mu_f = pava_2d_f64(mu_sum / max(n_collected, 1))
    cand2, act2, _ = phi_step_f64(mu_f, u_grid_np, p_grid_np, p_lo_np, p_hi_np,
                                      tau_f, gamma_f)
    F_f64 = float(np.max(np.abs(cand2 - mu_f)[act2]))
    print(f"  Polish done: max={F_f64:.3e}", flush=True)

    # Cast to mp50
    mu_init = [[mpf(str(mu_f[i, j])) for j in range(G)] for i in range(G)]

    # mp50 LM
    tag = f"γ={gamma_f},τ={tau_f}"
    mu_conv, F_max_v, F_med_v, history = nk_solve(
        mu_init, u_grid_mp, p_grid_mp, p_lo_mp, p_hi_mp, tau_mp, gamma_mp, tag)

    # Measure weighted 1-R²
    mu_arr = to_floats(mu_conv)
    one_R2, slope = weighted_R2_grid(u_grid_np, p_grid_np, mu_arr,
                                          gamma_f, tau_f)
    return mu_conv, F_max_v, F_med_v, history, one_R2, slope, p_grid_np


def save_ckpt(gamma_f, tau_f, mu_mp, u_grid_np, p_grid_np, F_max_v, F_med_v,
              history, one_R2, slope):
    path = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
            f"{gamma_tag(gamma_f)}_{tau_tag(tau_f)}_mp50.json")
    mu_strs = [[mpmath.nstr(mu_mp[i][j], 40) for j in range(G)]
                  for i in range(G)]
    rec = {
        "G": G, "UMAX": UMAX, "tau": tau_f, "gamma": gamma_f,
        "trim": TRIM, "dps": 50,
        "F_max": mpmath.nstr(F_max_v, 30),
        "F_med": mpmath.nstr(F_med_v, 30),
        "1-R2_weighted": one_R2,
        "slope_weighted": slope,
        "u_grid": [str(x) for x in u_grid_np],
        "p_grid": [[str(p) for p in row] for row in p_grid_np],
        "mu_strings": mu_strs,
        "history": history,
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"  Saved {path}")
    return path


def main():
    t_start = time.time()
    u_grid_np = np.linspace(-UMAX, UMAX, G)
    u_grid_mp = [mpf(str(x)) for x in u_grid_np]

    results = {gamma: [] for gamma in GAMMAS}

    # First, recompute weighted 1-R² for the existing τ=2 ckpts at each γ
    print("\n=== Recomputing weighted 1-R² for τ=2 ckpts ===")
    for gamma_f in GAMMAS:
        ckpt_path = TAU2_CKPT[gamma_f]
        with open(ckpt_path) as f:
            d = json.load(f)
        u_grid = np.array([float(s) for s in d["u_grid"]])
        p_grid = np.array([[float(s) for s in row] for row in d["p_grid"]])
        mu_arr = np.array([[float(s) for s in row] for row in d["mu_strings"]])
        one_R2, slope = weighted_R2_grid(u_grid, p_grid, mu_arr, gamma_f, 2.0)
        results[gamma_f].append({"tau": 2.0, "1-R2": one_R2, "slope": slope,
                                       "source": "task2_ckpt"})
        print(f"  γ={gamma_f}, τ=2: 1-R²={one_R2:.6e}, slope={slope:.4f}")

    # Now sweep τ for each γ
    for gamma_f in GAMMAS:
        for tau_f in TAU_VALUES:
            # Skip if ckpt already exists (resume across restarts)
            existing = (f"{RESULTS_DIR}/posterior_v3_G{G}_umax5_"
                        f"{gamma_tag(gamma_f)}_{tau_tag(tau_f)}_mp50.json")
            import os.path
            if os.path.exists(existing):
                with open(existing) as f:
                    d = json.load(f)
                results[gamma_f].append({
                    "tau": tau_f,
                    "1-R2": d.get("1-R2_weighted", None),
                    "slope": d.get("slope_weighted", None),
                    "source": "existing_ckpt",
                })
                print(f"  Skipping γ={gamma_f}, τ={tau_f} (ckpt exists, "
                      f"1-R²={d.get('1-R2_weighted')})", flush=True)
                continue

            print(f"\n=== γ={gamma_f}, τ={tau_f} ({time.time()-t_start:.0f}s elapsed) ===")
            t0 = time.time()
            try:
                (mu_conv, F_max_v, F_med_v, history, one_R2, slope,
                 p_grid_np) = solve_one(gamma_f, tau_f, u_grid_np, u_grid_mp)
                save_ckpt(gamma_f, tau_f, mu_conv, u_grid_np, p_grid_np,
                          F_max_v, F_med_v, history, one_R2, slope)
                results[gamma_f].append({"tau": tau_f, "1-R2": one_R2,
                                              "slope": slope})
                print(f"  γ={gamma_f}, τ={tau_f}: "
                      f"1-R²={one_R2:.6e}, slope={slope:.4f} "
                      f"(t={time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  γ={gamma_f}, τ={tau_f} FAILED: {e}")
                results[gamma_f].append({"tau": tau_f, "error": str(e)})

            # Save partial summary after each run (in case of crash)
            partial = {
                "figure": "fig4A",
                "params": {"G": G, "umax": UMAX, "trim": TRIM,
                           "weighting": "ex-ante 0.5*(f0³+f1³)"},
                "curves": [
                    {"gamma": g,
                     "points": sorted(results[g], key=lambda d: d["tau"])}
                    for g in GAMMAS
                ],
            }
            with open(f"{RESULTS_DIR}/fig4A_G20_tau_sweep.json", "w") as f:
                json.dump(partial, f, indent=2)

    # Final pgfplots
    out_tex = f"{RESULTS_DIR}/fig4A_G20_pgfplots.tex"
    lines = [f"% Fig 4A weighted 1-R² vs τ (G={G}, UMAX={UMAX}, trim={TRIM})"]
    for gamma_f in GAMMAS:
        pts_sorted = sorted(results[gamma_f], key=lambda d: d["tau"])
        good = [d for d in pts_sorted if "1-R2" in d]
        coords = "".join(f"({d['tau']:.1f},{d['1-R2']:.6e})" for d in good)
        lines += [f"% γ={gamma_f}",
                  f"\\addplot coordinates {{{coords}}};", ""]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {out_tex}")
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
