"""Search (tau, gamma) configurations maximizing 1-R² subject to strict convergence.

Convergence criterion:  ‖Φ(P) - P‖∞  <  ABSTOL  (= 1e-10 by default).
Only strictly converged solutions are recorded; the rest are logged but
excluded from the ranked output.

Sampling strategy:
  1. Deterministic coarse grid over {γ} × {τ} with some symmetry culling.
  2. Random log-uniform samples for broader exploration.
  3. Targeted "endogenous noise trader" samples: low γ on one agent, high γ
     on the others, aligned/misaligned with τ.

Each config is tried with a damping ladder α ∈ {1.0, 0.3, 0.1, 0.03}
and max-iter ladder {3k, 10k, 30k}. We accept the first α that meets
the convergence criterion.

All results stream to /home/user/REZN/python/search_het_results.csv.

Usage:
    python3 -u search_het.py --budget-hours 3 --G 9 --seed 42
"""
import argparse
import csv
import itertools
import os
import random
import sys
import time

import numpy as np

import rezn_het as rh

# ---------------------------------------------------------
# Arguments / global config
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--G", type=int, default=9)
parser.add_argument("--umax", type=float, default=2.0)
parser.add_argument("--abstol", type=float, default=1e-10, help="‖Φ-I‖∞ tolerance")
parser.add_argument("--f-tol", type=float, default=1e-6, help="‖F‖∞ (market-clear) tolerance")
parser.add_argument("--budget-hours", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str,
                    default="/home/user/REZN/python/search_het_results.csv")
args = parser.parse_args()

G = args.G
UMAX = args.umax
ABSTOL = args.abstol
F_TOL = args.f_tol
BUDGET_S = args.budget_hours * 3600.0
random.seed(args.seed); np.random.seed(args.seed)

U_REPORT = (1.0, -1.0, 1.0)


# ---------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------

def coarse_grid():
    """Deterministic coarse grid with light symmetry culling."""
    gammas_grid = [0.3, 0.5, 1.0, 2.0, 5.0]
    taus_grid = [0.5, 1.0, 2.0, 4.0]
    # Homogeneous baselines: all gammas equal, all taus equal
    for g in gammas_grid:
        for t in taus_grid:
            yield (t, t, t), (g, g, g)
    # Heterogeneous gammas with equal taus
    for g1, g2, g3 in itertools.combinations_with_replacement(gammas_grid, 3):
        if (g1, g2, g3) == (g1,)*3: continue  # dedup with homo loop
        for t in taus_grid:
            yield (t, t, t), (g1, g2, g3)
    # Equal gammas with het taus
    for t1, t2, t3 in itertools.combinations_with_replacement(taus_grid, 3):
        if (t1, t2, t3) == (t1,)*3: continue
        for g in gammas_grid:
            yield (t1, t2, t3), (g, g, g)


def random_loguniform(n):
    """Random (tau_vec, gamma_vec) with log-uniform components."""
    for _ in range(n):
        tau = tuple(10 ** np.random.uniform(np.log10(0.3), np.log10(5.0), 3))
        gam = tuple(10 ** np.random.uniform(np.log10(0.3), np.log10(10.0), 3))
        yield tau, gam


def extreme_misaligned(n):
    """Low-γ / low-τ on one agent, high on the others (aligned or not)."""
    for _ in range(n):
        agent = np.random.randint(3)
        tau_noisy = 10 ** np.random.uniform(np.log10(0.3), np.log10(1.0))
        tau_info = 10 ** np.random.uniform(np.log10(1.0), np.log10(5.0))
        gam_aggressive = 10 ** np.random.uniform(np.log10(0.3), np.log10(1.0))
        gam_safe = 10 ** np.random.uniform(np.log10(1.0), np.log10(10.0))
        taus = [tau_info, tau_info, tau_info]
        gams = [gam_safe, gam_safe, gam_safe]
        taus[agent] = tau_noisy
        gams[agent] = gam_aggressive  # aggressive trader on noisy signal
        yield tuple(taus), tuple(gams)


# ---------------------------------------------------------
# 1-R² with heterogeneous CARA reference
# ---------------------------------------------------------

def one_minus_R2_het(Pg, u, taus, gammas):
    """Regress logit(p) on the CARA FR predictor
       T_CARA = Σ (τ_k/γ_k) u_k  /  Σ (1/γ_k).
    1 - R² is the fraction of variance NOT explained by this linear combo."""
    G = u.shape[0]
    taus = np.asarray(taus, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    w = (1.0 / gammas) / (1.0 / gammas).sum()
    coef = w * taus
    y = np.log(Pg / (1.0 - Pg)).reshape(-1)
    T = np.empty(G ** 3)
    k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                T[k] = coef[0]*u[i] + coef[1]*u[j] + coef[2]*u[l]
                k += 1
    y_c = y - y.mean(); T_c = T - T.mean()
    Syy = float((y_c*y_c).sum()); STT = float((T_c*T_c).sum())
    SyT = float((y_c*T_c).sum())
    if Syy == 0.0 or STT == 0.0:
        return 0.0
    R2 = (SyT*SyT) / (Syy*STT)
    return max(0.0, 1.0 - R2)


# ---------------------------------------------------------
# Per-config solver (damping ladder, early exit on convergence)
# ---------------------------------------------------------

ALPHA_LADDER  = [1.0, 0.3, 0.1, 0.03]
MAXITER_TABLE = {1.0: 3000, 0.3: 6000, 0.1: 15000, 0.03: 30000}


def solve_with_ladder(taus, gammas):
    best = {"PhiI": float("inf"), "Finf": float("inf"),
            "iters": 0, "time": 0.0, "alpha": None,
            "P_star": None, "converged": False}
    for alpha in ALPHA_LADDER:
        mi = MAXITER_TABLE[alpha]
        t0 = time.time()
        try:
            res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                                  maxiters=mi, abstol=ABSTOL, alpha=alpha)
        except Exception as e:
            sys.stderr.write(f"[error tau={taus} gamma={gammas} alpha={alpha}]: {e}\n")
            continue
        dt = time.time() - t0
        PhiI = res["history"][-1] if res["history"] else float("inf")
        Finf = float(np.abs(res["residual"]).max())
        cand = {"PhiI": PhiI, "Finf": Finf,
                "iters": len(res["history"]), "time": dt,
                "alpha": alpha, "P_star": res["P_star"],
                "converged": (PhiI < ABSTOL) and (Finf < F_TOL)}
        # Prefer strictly converged, else lowest ‖Φ-I‖
        better = False
        if cand["converged"] and not best["converged"]:
            better = True
        elif cand["converged"] and best["converged"] and cand["time"] < best["time"]:
            better = True
        elif not best["converged"] and cand["PhiI"] < best["PhiI"]:
            better = True
        if better:
            best = cand
        if cand["converged"]:
            break
    return best


# ---------------------------------------------------------
# Main loop with CSV streaming
# ---------------------------------------------------------

def main():
    u = rh.build_grid(G, UMAX)
    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i_r, j_r, l_r = idx(U_REPORT[0]), idx(U_REPORT[1]), idx(U_REPORT[2])

    # combine candidate generators
    cands = list(coarse_grid()) \
          + list(random_loguniform(500)) \
          + list(extreme_misaligned(200))
    # dedup
    seen = set(); unique = []
    for t, g in cands:
        key = (tuple(round(x, 6) for x in t), tuple(round(x, 6) for x in g))
        if key in seen: continue
        seen.add(key); unique.append((t, g))
    print(f"total candidate configs: {len(unique)}  (budget {args.budget_hours:.1f} h)")
    sys.stdout.flush()

    # warmup numba JIT
    _ = rh.solve_picard(5, 2.0, 0.5, maxiters=3, abstol=1e-3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_1","tau_2","tau_3","gamma_1","gamma_2","gamma_3",
                    "alpha","iters","time_s","PhiI","Finf",
                    "oneR2_het","oneR2_eq","p_star",
                    "mu_1","mu_2","mu_3","pr_gap","converged"])
        f.flush()

        t_start = time.time()
        n_done = 0; n_conv = 0; best_1mR2 = 0.0; best_tag = None
        for (taus, gammas) in unique:
            if time.time() - t_start > BUDGET_S:
                print(f"\nbudget exhausted after {n_done} configs")
                break
            best = solve_with_ladder(taus, gammas)
            if best["P_star"] is None:
                continue
            P = best["P_star"]
            mu = rh.posteriors_at(i_r, j_r, l_r, float(P[i_r,j_r,l_r]), P, u,
                                   np.asarray(taus, dtype=float))
            one_het = one_minus_R2_het(P, u, taus, gammas)
            one_eq  = rh.one_minus_R2(P, u, taus)
            w.writerow(
                list(taus) + list(gammas) +
                [f"{best['alpha']}", best["iters"],
                 f"{best['time']:.2f}",
                 f"{best['PhiI']:.3e}", f"{best['Finf']:.3e}",
                 f"{one_het:.6e}", f"{one_eq:.6e}",
                 f"{float(P[i_r,j_r,l_r]):.10f}",
                 f"{mu[0]:.8f}", f"{mu[1]:.8f}", f"{mu[2]:.8f}",
                 f"{mu[0]-mu[1]:.6f}", int(best["converged"])]
            )
            f.flush()

            n_done += 1
            if best["converged"]:
                n_conv += 1
                if one_het > best_1mR2:
                    best_1mR2 = one_het
                    best_tag = (taus, gammas, best["alpha"], best["iters"])

            # periodic status
            if n_done % 25 == 0 or n_done == 1:
                tot_s = time.time() - t_start
                eta_h = (tot_s / n_done) * (len(unique) - n_done) / 3600.0
                print(f"[{n_done:4d}/{len(unique)}] converged={n_conv}  "
                      f"best 1-R²_het={best_1mR2:.3e}  at={best_tag}  "
                      f"elapsed={tot_s/60:.1f}min  ETA={eta_h:.1f}h")
                sys.stdout.flush()

    print(f"\n=== search complete: {n_done} tried, {n_conv} strictly converged ===")
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()
