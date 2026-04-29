"""(γ, τ) sweep at G=20, smooth Φ (fast), h=0.01.

Centered around the converged (γ=0.1, τ=2) tensor that gave 1−R²=0.0453.
Continuation seeds from converged neighbors. Strict tolerance: residual ≤ 1e-13.

  γ values: 0.05, 0.10, 0.20, 0.50, 1.0, 5.0, 20.0
  τ values: 1.0, 1.5, 2.0, 2.5, 3.0

Snake order: outer τ, inner γ (γ ladder restarts at γ=0.1 for each new τ via
prior-row seed).
"""

import json, subprocess, sys, time, csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth_fast.py"
G = 20; H = 0.01

GAMMAS = [0.05, 0.10, 0.20, 0.50, 1.0, 5.0, 20.0]
TAUS   = [1.0, 1.5, 2.0, 2.5, 3.0]
ACCEPT_TOL = 1.0e-13

# Initial seed: the deeply converged G=20 γ=0.1 τ=2 h=0.01 tensor (residual 9.7e-14)
INITIAL_SEED = RESULTS / "G20_tau2_smoothfasth0.01_het0.1_0.1_0.1_g0.1_G20_deep_prices.npz"


def call(label, gammas, tau, seed, max_iter=600, tol=1e-14,
         damping=0.3, anderson=5, anderson_beta=0.7):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", str(tau),
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed),
        "--label", label, "--h", str(H),
        "--max-iter", str(max_iter), "--tol", str(tol),
        "--damping", str(damping),
        "--anderson", str(anderson), "--anderson-beta", str(anderson_beta),
        "--symmetric",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau{tau:g}_smoothfasth{H:g}_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def best_resid(s):
    return float(s.get("residual_inf", float("inf"))) if s else float("inf")


def attempt(label, gammas, tau, seed, attempt_idx):
    if attempt_idx == 0:
        kw = dict(max_iter=400, tol=1e-14, damping=0.3, anderson=5, anderson_beta=0.7)
    elif attempt_idx == 1:
        kw = dict(max_iter=1000, tol=1e-14, damping=0.2, anderson=5, anderson_beta=0.7)
    else:
        kw = dict(max_iter=2000, tol=1e-14, damping=0.1, anderson=8, anderson_beta=0.5)
    return call(f"{label}_a{attempt_idx}", gammas, tau, seed, **kw)


def find_seed_path(gammas, tau, label):
    g_str = "_".join(f"{g:g}" for g in gammas)
    return RESULTS / f"G{G}_tau{tau:g}_smoothfasth{H:g}_het{g_str}_{label}_prices.npz"


def main():
    if not INITIAL_SEED.exists():
        print(f"FATAL: {INITIAL_SEED.name} missing"); return
    print(f"Initial seed: {INITIAL_SEED.name}")

    rows = []
    cur_seed = INITIAL_SEED
    last_anchor_seed = INITIAL_SEED   # the (γ=0.1, τ_prev) tensor reused across τ rows
    print(f"\n{'τ':>5} {'γ':>6} {'iters':>5} {'resid':>11} {'1-R²':>11} {'slope':>7} {'time(s)':>8}")
    print("-" * 70)

    for ti, tau in enumerate(TAUS):
        # For each new τ, reset the inner γ-ladder seed to the most recent
        # γ=0.1 result at the same row's previous τ if available.
        cur_seed = last_anchor_seed
        for gam in GAMMAS:
            gammas = (gam, gam, gam)
            label = f"swG20_t{tau:g}_g{gam:g}"
            accepted = None
            elapsed_total = 0.0
            for k in range(3):
                t0 = time.time()
                s = attempt(label, gammas, tau, cur_seed, k)
                elapsed = time.time() - t0
                elapsed_total += elapsed
                if s is None: continue
                r = best_resid(s)
                if r <= ACCEPT_TOL:
                    accepted = s
                    cur_seed = find_seed_path(gammas, tau, f"{label}_a{k}")
                    if gam == 0.1:
                        last_anchor_seed = cur_seed
                    break
            if accepted is None:
                resid = best_resid(s) if s else float('inf')
                print(f"{tau:>5g} {gam:>6g} HALT (best={resid:.2e})")
                break
            iters = accepted["iterations"]
            r = accepted["residual_inf"]
            R2 = accepted["revelation_deficit"]
            sl = accepted["slope"]
            print(f"{tau:>5g} {gam:>6g} {iters:>5} {r:>11.3e} {R2:>11.4e} {sl:>7.4f} {elapsed_total:>8.1f}",
                  flush=True)
            rows.append({"tau": tau, "gamma": gam, **accepted})

    # Save
    with open(RESULTS / f"sweep_gamma_tau_G{G}_h{H:g}.json", "w") as f:
        json.dump(rows, f, indent=2)
    csv_path = RESULTS / f"sweep_gamma_tau_G{G}_h{H:g}.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["tau", "gamma", "iters", "residual_inf", "1-R2", "slope",
                    "max_FR_err", "price_canon"])
        for r in rows:
            rep = r.get("representative_realization", {})
            w.writerow([r["tau"], r["gamma"], r["iterations"], r["residual_inf"],
                        r["revelation_deficit"], r["slope"], r["max_fr_error"],
                        rep.get("price")])
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    if rows:
        print("\n2D 1-R² table (rows=τ, cols=γ):")
        hdr = "τ\\γ"
        print(f"{hdr:>8}", end="")
        for g in GAMMAS:
            print(f"{g:>10g}", end="")
        print()
        for tau in TAUS:
            print(f"{tau:>8g}", end="")
            for gam in GAMMAS:
                hit = next((r for r in rows if r["tau"] == tau and r["gamma"] == gam), None)
                if hit:
                    print(f"{hit['revelation_deficit']:>10.4e}", end="")
                else:
                    print(f"{'--':>10}", end="")
            print()


if __name__ == "__main__":
    main()
