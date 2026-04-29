"""
Fine gamma ladder at G=6, tau=2.

Steps: gamma in {0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10}.
Each step:
  1. Picard with Anderson from the previous accepted tensor.
  2. Newton polish from the Picard result.
  3. ACCEPT only if best residual <= 1e-14.
  4. If not accepted, retry with: more Picard iters, smaller damping, more Newton iters.
  5. If still not accepted after retries, halt the ladder.

The starting tensor is the polished gamma=0.5 tensor at residual 7.77e-16.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver.py"

ACCEPT_TOL = 1.0e-14
GAMMAS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]

# starting accepted tensor (gamma=0.5 polish, residual ~7.77e-16)
START_TENSOR = REPO / "results/full_ree/G6_tau2_gamma0.5_G6_newton_polish_prices.npz"


def call_solver(label, gamma, seed_npz, method="picard", **kw):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gamma", str(gamma),
        "--seed", "array",
        "--seed-array", str(seed_npz),
        "--label", label,
        "--method", method,
        "--max-iter", str(kw.get("max_iter", 600)),
        "--damping", str(kw.get("damping", 0.3)),
        "--tol", str(kw.get("tol", 1e-15)),
        "--save-array",
    ]
    if method == "picard":
        cmd += ["--anderson", str(kw.get("anderson", 5)),
                "--anderson-beta", str(kw.get("anderson_beta", 0.7))]
    if method == "newton":
        cmd += ["--gmres-max-iter", str(kw.get("gmres_max_iter", 80)),
                "--gmres-tol", str(kw.get("gmres_tol", 1e-12)),
                "--fd-eps", str(kw.get("fd_eps", 1e-7)),
                "--newton-damping", str(kw.get("newton_damping", 1.0))]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1500:])
        sys.stdout.write("\nSTDERR:\n" + out.stderr[-1500:])
        return None
    summary_path = RESULTS / f"G6_tau2_gamma{gamma}_{label}_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def best_residual(summary):
    if summary is None:
        return float("inf")
    return float(summary.get("residual_inf", float("inf")))


def attempt_step(gamma, seed_npz, attempt_idx):
    """Returns (best_summary, best_npz) or (None, None)."""
    # Picard step
    if attempt_idx == 0:
        picard_kw = dict(damping=0.3, anderson=5, anderson_beta=0.7,
                         max_iter=600, tol=1e-15)
        newton_kw = dict(max_iter=15, tol=1e-16,
                         gmres_max_iter=80, gmres_tol=1e-12, fd_eps=1e-7,
                         newton_damping=1.0)
    elif attempt_idx == 1:
        # tighter Picard, more Newton
        picard_kw = dict(damping=0.2, anderson=5, anderson_beta=0.7,
                         max_iter=2000, tol=1e-15)
        newton_kw = dict(max_iter=30, tol=1e-16,
                         gmres_max_iter=120, gmres_tol=1e-13, fd_eps=1e-7,
                         newton_damping=0.8)
    else:
        # last resort: very small Picard damping, exhaustive
        picard_kw = dict(damping=0.1, anderson=8, anderson_beta=0.5,
                         max_iter=4000, tol=1e-15)
        newton_kw = dict(max_iter=60, tol=1e-16,
                         gmres_max_iter=160, gmres_tol=1e-14, fd_eps=1e-8,
                         newton_damping=0.5)

    p_label = f"fine_picard_{attempt_idx}"
    p_sum = call_solver(p_label, gamma, seed_npz, method="picard", **picard_kw)
    if p_sum is None:
        return None, None
    p_npz = RESULTS / f"G6_tau2_gamma{gamma}_{p_label}_prices.npz"

    if best_residual(p_sum) <= ACCEPT_TOL:
        return p_sum, p_npz

    # Newton polish from Picard
    n_label = f"fine_newton_{attempt_idx}"
    n_sum = call_solver(n_label, gamma, p_npz, method="newton", **newton_kw)
    n_npz = RESULTS / f"G6_tau2_gamma{gamma}_{n_label}_prices.npz"
    if n_sum is not None and best_residual(n_sum) <= ACCEPT_TOL:
        return n_sum, n_npz

    # Pick the better of the two
    if n_sum is not None and best_residual(n_sum) < best_residual(p_sum):
        return n_sum, n_npz
    return p_sum, p_npz


def main():
    # Validate starting tensor exists
    if not START_TENSOR.exists():
        print(f"FATAL: starting tensor missing: {START_TENSOR}", flush=True)
        return

    accepted = {0.50: (None, START_TENSOR)}  # gamma -> (summary, npz)
    cur_seed = START_TENSOR
    rows = []

    for gamma in GAMMAS:
        if gamma == 0.50:
            print(f"\n=== gamma=0.50 (baseline, accepted from prior polish) ===", flush=True)
            # Read the existing summary
            with open(REPO / "results/full_ree/G6_tau2_gamma0.5_G6_newton_polish_summary.json") as f:
                base_summary = json.load(f)
            rows.append((0.50, base_summary, "ACCEPTED (baseline)"))
            cur_seed = START_TENSOR
            continue

        print(f"\n=== gamma={gamma:.2f}  seed={cur_seed.name} ===", flush=True)
        accepted_this = None
        accepted_npz = None
        for attempt in range(3):
            t0 = time.time()
            summary, npz = attempt_step(gamma, cur_seed, attempt)
            elapsed = time.time() - t0
            resid = best_residual(summary)
            R2 = summary.get("revelation_deficit") if summary else None
            R2_str = f"{R2:.4e}" if R2 is not None else "NA"
            print(f"  attempt {attempt}: resid={resid:.3e}  1-R²={R2_str}  "
                  f"({elapsed:.1f}s)", flush=True)
            if resid <= ACCEPT_TOL:
                accepted_this = summary
                accepted_npz = npz
                break

        if accepted_this is None:
            print(f"  HALT: gamma={gamma:.2f} did not converge to {ACCEPT_TOL:.0e} after 3 attempts; "
                  f"best residual={resid:.3e}", flush=True)
            rows.append((gamma, summary, f"REJECTED (best={resid:.3e})"))
            break

        rep = accepted_this.get("representative_realization", {})
        print(f"  ACCEPTED: resid={best_residual(accepted_this):.3e}  "
              f"1-R²={accepted_this['revelation_deficit']:.4e}  "
              f"max_FR_err={accepted_this['max_fr_error']:.4f}  "
              f"price={rep.get('price'):.5f}  FR={rep.get('fr_price'):.5f}", flush=True)
        rows.append((gamma, accepted_this, "ACCEPTED"))
        cur_seed = accepted_npz

    # Final table
    print("\n" + "=" * 100)
    print(f"{'gamma':>6} | {'resid':>11} | {'1-R²':>11} | {'max FR':>9} | "
          f"{'price':>9} | {'FR':>9} | {'status':>30}")
    print("-" * 100)
    table = []
    for gamma, summary, status in rows:
        if summary is None:
            print(f"{gamma:>6.2f} | {'-':>11} | {'-':>11} | {'-':>9} | {'-':>9} | {'-':>9} | {status:>30}")
            continue
        rep = summary.get("representative_realization", {})
        resid = summary.get("residual_inf", float("nan"))
        R2 = summary.get("revelation_deficit", float("nan"))
        mfe = summary.get("max_fr_error", float("nan"))
        pr = rep.get("price", float("nan"))
        fr = rep.get("fr_price", float("nan"))
        print(f"{gamma:>6.2f} | {resid:>11.3e} | {R2:>11.4e} | {mfe:>9.4f} | "
              f"{pr:>9.5f} | {fr:>9.5f} | {status:>30}")
        table.append({
            "gamma": gamma, "resid": resid, "1-R2": R2, "max_FR_err": mfe,
            "price": pr, "fr_price": fr, "status": status,
        })
    out = RESULTS / "ladder_fine_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
