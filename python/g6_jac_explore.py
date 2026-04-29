"""Explore the Jacobian-corrected fixed point at G=6, tau=2, gamma=2.

Tries multiple seeds and solver settings to find the best approximation
of the Jac-corrected REE. Saves all converged tensors and tabulates.
"""

import json, subprocess, sys, time, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_jac.py"


def call(label, gammas, seed_npz, **kw):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed_npz),
        "--label", label,
        "--method", kw.get("method", "picard"),
        "--max-iter", str(kw.get("max_iter", 500)),
        "--tol", str(kw.get("tol", 1e-14)),
        "--damping", str(kw.get("damping", 0.3)),
        "--anderson", str(kw.get("anderson", 5)),
        "--anderson-beta", str(kw.get("anderson_beta", 0.7)),
    ]
    if kw.get("method") == "newton":
        cmd += ["--gmres-max-iter", str(kw.get("gmres_max_iter", 100)),
                "--gmres-tol", str(kw.get("gmres_tol", 1e-3)),
                "--fd-eps", str(kw.get("fd_eps", 1e-5)),
                "--newton-damping", str(kw.get("newton_damping", 1.0))]
    if kw.get("symmetric"):
        cmd.append("--symmetric")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1200:] + "\nSTDERR:\n" + out.stderr[-1200:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G6_tau2_jac_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def best_resid(s):
    return float(s.get("residual_inf", float("inf"))) if s is not None else float("inf")


def trial(label, gammas, seed_npz, **kw):
    t0 = time.time()
    s = call(label, gammas, seed_npz, **kw)
    elapsed = time.time() - t0
    resid = best_resid(s)
    R2 = s.get("revelation_deficit") if s else None
    slope = s.get("slope") if s else None
    rep = s.get("representative_realization", {}) if s else {}
    R2_str = f"{R2:.4e}" if R2 is not None else "NA"
    slope_str = f"{slope:.4f}" if slope is not None else "NA"
    print(f"  [{label}]  resid={resid:.3e}  1-R²={R2_str}  slope={slope_str}  "
          f"price={rep.get('price', float('nan')):.5f}  ({elapsed:.1f}s)",
          flush=True)
    return s


def main():
    gammas = (2.0, 2.0, 2.0)
    cursor_seed = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"

    print("=" * 80)
    print("Jacobian-corrected REE search at gamma=(2,2,2), G=6, tau=2")
    print("=" * 80, flush=True)

    # 1. Long damped Picard from cursor seed (symmetric)
    print("\n--- 1. Long damped Picard (sym) from cursor seed ---", flush=True)
    s1 = trial("explore_long_picard_sym", gammas, cursor_seed,
               method="picard", damping=0.15, anderson=8, anderson_beta=0.5,
               max_iter=2000, tol=1e-13, symmetric=True)

    # 2. Continue from result of 1 with even smaller damping, longer
    if s1:
        seed2 = RESULTS / "G6_tau2_jac_het2_2_2_explore_long_picard_sym_prices.npz"
        if seed2.exists():
            print("\n--- 2. Continue from #1 with α=0.05, 4000 iters ---", flush=True)
            s2 = trial("explore_continue_sym", gammas, seed2,
                       method="picard", damping=0.05, anderson=10, anderson_beta=0.3,
                       max_iter=4000, tol=1e-14, symmetric=True)

    # 3. Newton from result of 2 (or 1)
    seed3 = RESULTS / "G6_tau2_jac_het2_2_2_explore_continue_sym_prices.npz"
    if not seed3.exists():
        seed3 = RESULTS / "G6_tau2_jac_het2_2_2_explore_long_picard_sym_prices.npz"
    if seed3.exists():
        print("\n--- 3. Newton-Krylov polish (sym) ---", flush=True)
        s3 = trial("explore_newton_sym", gammas, seed3,
                   method="newton", max_iter=40, tol=1e-15,
                   gmres_max_iter=120, gmres_tol=5e-4, fd_eps=1e-5,
                   newton_damping=1.0, symmetric=True)

    # 4. Without symmetrize — confirm symmetric basin
    if seed3.exists():
        print("\n--- 4. Without symmetrize: continue from sym fixed point ---", flush=True)
        s4 = trial("explore_unsym", gammas, seed3,
                   method="picard", damping=0.1, anderson=8, anderson_beta=0.5,
                   max_iter=2000, tol=1e-13)

    # 5. From no-learning seed (sanity check)
    NL_SCRIPT = REPO / "python" / "full_ree_solver.py"
    nl_seed = RESULTS / "G6_tau2_gamma2_no_learning_seed.npz"
    if not nl_seed.exists():
        # Reuse cursor's no-learning generation
        cmd = ["python3", str(NL_SCRIPT), "--G", "6", "--umax", "2", "--tau", "2",
               "--gamma", "2", "--seed", "no-learning", "--max-iter", "1",
               "--save-array", "--label", "nl_seed_for_jac"]
        subprocess.run(cmd, capture_output=True)
        # Cursor saves as G6_tau2_gamma2_nl_seed_for_jac_prices.npz
        candidates = list(RESULTS.glob("G6_tau2_gamma2_nl*prices.npz"))
        if candidates:
            nl_seed = candidates[0]
            print(f"  using NL seed: {nl_seed.name}", flush=True)
        else:
            print("  could not produce NL seed; skipping #5", flush=True)
            nl_seed = None
    if nl_seed and nl_seed.exists():
        print("\n--- 5. From no-learning seed (sym) ---", flush=True)
        s5 = trial("explore_from_nl_sym", gammas, nl_seed,
                   method="picard", damping=0.1, anderson=8, anderson_beta=0.5,
                   max_iter=2000, tol=1e-13, symmetric=True)


if __name__ == "__main__":
    main()
