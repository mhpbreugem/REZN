"""
Heterogeneous-gamma ladder at G=6, tau=2.

Starts from the converged homogeneous gamma=(2,2,2) tensor and gradually
tilts toward gamma_target = (1, 2, 4) using a heterogeneity parameter
t in {0.0, 0.1, ..., 1.0}.

  gamma(t) = (2 - t, 2, 2 + 2t)

At t=0: (2, 2, 2) homogeneous.
At t=1: (1, 2, 4) full heterogeneity.

Each step: continue from the previous accepted tensor with the het solver
(no symmetrization, since het gammas break the (i,j,k) permutation symmetry).
ACCEPT if residual <= 1e-14 in 3 attempts, else HALT.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het.py"

ACCEPT_TOL = 1.0e-14
T_LADDER = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
START_TENSOR = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"


def gammas_at(t):
    return (2.0 - t, 2.0, 2.0 + 2.0 * t)


def gammas_str(g):
    return ",".join(f"{x:g}" for x in g)


def call_solver(label, gammas, seed_npz, **kw):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gammas", gammas_str(gammas),
        "--seed-array", str(seed_npz),
        "--label", label,
        "--max-iter", str(kw.get("max_iter", 600)),
        "--damping", str(kw.get("damping", 0.3)),
        "--tol", str(kw.get("tol", 1e-15)),
        "--anderson", str(kw.get("anderson", 5)),
        "--anderson-beta", str(kw.get("anderson_beta", 0.7)),
    ]
    if kw.get("symmetric", False):
        cmd.append("--symmetric")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1500:])
        sys.stdout.write("\nSTDERR:\n" + out.stderr[-1500:])
        return None
    summary_path = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def best_residual(s):
    return float(s.get("residual_inf", float("inf"))) if s is not None else float("inf")


def attempt_step(t, gammas, seed_npz, attempt_idx, symmetric):
    if attempt_idx == 0:
        kw = dict(damping=0.3, anderson=5, anderson_beta=0.7,
                  max_iter=600, tol=1e-15, symmetric=symmetric)
    elif attempt_idx == 1:
        kw = dict(damping=0.2, anderson=5, anderson_beta=0.7,
                  max_iter=2000, tol=1e-15, symmetric=symmetric)
    else:
        kw = dict(damping=0.1, anderson=8, anderson_beta=0.5,
                  max_iter=4000, tol=1e-15, symmetric=symmetric)
    label = f"hetladder_t{t:g}_{attempt_idx}"
    summary = call_solver(label, gammas, seed_npz, **kw)
    npz = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_prices.npz"
    return summary, npz


def main():
    if not START_TENSOR.exists():
        print(f"FATAL: {START_TENSOR} missing")
        return
    cur_seed = START_TENSOR
    rows = []
    for t in T_LADDER:
        gammas = gammas_at(t)
        symmetric = (t == 0.0)
        print(f"\n=== t={t:g}  gammas={gammas}  seed={cur_seed.name} ===", flush=True)
        accepted_summary = None
        accepted_npz = None
        for attempt in range(3):
            t0 = time.time()
            summary, npz = attempt_step(t, gammas, cur_seed, attempt, symmetric)
            elapsed = time.time() - t0
            resid = best_residual(summary)
            R2 = summary.get("revelation_deficit") if summary else None
            R2_str = f"{R2:.4e}" if R2 is not None else "NA"
            print(f"  attempt {attempt}: resid={resid:.3e}  1-R²={R2_str}  ({elapsed:.1f}s)",
                  flush=True)
            if resid <= ACCEPT_TOL:
                accepted_summary = summary
                accepted_npz = npz
                break
        if accepted_summary is None:
            print(f"  HALT t={t:g}: best resid={resid:.3e}", flush=True)
            rows.append((t, gammas, summary, f"REJECTED ({resid:.3e})"))
            break
        rep = accepted_summary.get("representative_realization", {})
        slope = accepted_summary.get("slope", float("nan"))
        print(f"  ACCEPTED: resid={best_residual(accepted_summary):.3e}  "
              f"1-R²={accepted_summary['revelation_deficit']:.4e}  "
              f"slope={slope:.4f}  max_FR={accepted_summary['max_fr_error']:.4f}  "
              f"price={rep.get('price'):.5f}  posts={[round(p,4) for p in rep.get('posteriors',[])]}",
              flush=True)
        rows.append((t, gammas, accepted_summary, "ACCEPTED"))
        cur_seed = accepted_npz

    print("\n" + "=" * 110)
    print(f"{'t':>5} | {'gammas':>15} | {'resid':>11} | {'1-R²':>11} | "
          f"{'slope':>8} | {'max FR':>8} | {'price':>9} | {'mu1':>7} | {'mu2':>7} | {'mu3':>7}")
    print("-" * 110)
    table = []
    for t, gammas, summary, status in rows:
        if summary is None:
            print(f"{t:>5g} | {gammas_str(gammas):>15} | -- | -- | -- | -- | -- | -- | -- | --  {status}")
            continue
        rep = summary.get("representative_realization", {})
        ps = rep.get("posteriors", [None]*3)
        print(f"{t:>5g} | {gammas_str(gammas):>15} | "
              f"{summary['residual_inf']:>11.3e} | {summary['revelation_deficit']:>11.4e} | "
              f"{summary.get('slope',0):>8.4f} | {summary['max_fr_error']:>8.4f} | "
              f"{rep.get('price',float('nan')):>9.5f} | "
              f"{ps[0]:>7.4f} | {ps[1]:>7.4f} | {ps[2]:>7.4f}")
        table.append({
            "t": t, "gammas": list(gammas),
            "residual_inf": summary["residual_inf"],
            "revelation_deficit": summary["revelation_deficit"],
            "slope": summary.get("slope"),
            "max_fr_error": summary["max_fr_error"],
            "price": rep.get("price"),
            "posteriors": ps,
            "status": status,
        })
    out = RESULTS / "het_ladder_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
