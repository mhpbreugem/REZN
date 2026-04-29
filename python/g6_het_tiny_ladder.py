"""
Heterogeneous-gamma TINY-step ladder at G=6, tau=2.

Starting tensor: converged homogeneous gamma=(2,2,2) tensor
PLUS a 1e-10 anti-symmetric perturbation that breaks (i,j,k) symmetry
but keeps the iterate inside the basin of any nearby asymmetric fixed point.

Ladder over a small heterogeneity parameter t, starting at t=0.001 and
ramping up multiplicatively.  gamma(t) = (2-t, 2, 2+2t)  (linear tilt).

  t in 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20,
       0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00

At each step, run het Picard-Anderson from the previous accepted tensor.
Symmetrization is OFF for all rungs (t > 0).
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het.py"

ACCEPT_TOL = 1.0e-13
T_LADDER = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
]
HOMOG_TENSOR = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"
PERTURBED_SEED = REPO / "results/full_ree/G6_tau2_gamma2_perturbed_1em10.npz"
PERTURB_SCALE = 1.0e-10
PERTURB_RNG_SEED = 17


def gammas_at(t):
    return (2.0 - t, 2.0, 2.0 + 2.0 * t)


def gammas_str(g):
    return ",".join(f"{x:g}" for x in g)


def make_perturbed_seed():
    """Add an anti-symmetric perturbation in (i,j,k) at scale 1e-10
    to the converged homogeneous tensor."""
    base = np.load(HOMOG_TENSOR)
    P = base["P"].copy()
    grid = base["grid"]
    rng = np.random.default_rng(PERTURB_RNG_SEED)
    noise = rng.standard_normal(P.shape) * PERTURB_SCALE
    # Anti-symmetric component: zero on the diagonal P(u,u,u), non-zero off
    # We don't strictly enforce anti-symmetry; just an iid noise field at 1e-10.
    P_pert = np.clip(P + noise, 1.0e-8, 1.0 - 1.0e-8)
    np.savez(PERTURBED_SEED, P=P_pert, grid=grid)
    print(f"  Perturbed seed: scale={PERTURB_SCALE:.1e}, "
          f"max |delta|={np.max(np.abs(P_pert - P)):.3e}, "
          f"saved to {PERTURBED_SEED.name}", flush=True)


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
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:])
        sys.stdout.write("\nSTDERR:\n" + out.stderr[-1000:])
        return None
    summary_path = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def best_residual(s):
    return float(s.get("residual_inf", float("inf"))) if s is not None else float("inf")


def attempt_step(t, gammas, seed_npz, attempt_idx):
    if attempt_idx == 0:
        kw = dict(damping=0.3, anderson=5, anderson_beta=0.7,
                  max_iter=400, tol=1e-15)
    elif attempt_idx == 1:
        kw = dict(damping=0.2, anderson=5, anderson_beta=0.7,
                  max_iter=1500, tol=1e-15)
    else:
        kw = dict(damping=0.1, anderson=8, anderson_beta=0.5,
                  max_iter=4000, tol=1e-15)
    label = f"tinyladder_t{t:g}_{attempt_idx}"
    summary = call_solver(label, gammas, seed_npz, **kw)
    npz = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_prices.npz"
    return summary, npz


def main():
    if not HOMOG_TENSOR.exists():
        print(f"FATAL: {HOMOG_TENSOR} missing")
        return
    print(f"Building perturbed seed from {HOMOG_TENSOR.name} ...", flush=True)
    make_perturbed_seed()

    cur_seed = PERTURBED_SEED
    rows = []
    for t in T_LADDER:
        gammas = gammas_at(t)
        print(f"\n=== t={t:g}  gammas={gammas}  seed={cur_seed.name} ===", flush=True)
        accepted_summary = None
        accepted_npz = None
        for attempt in range(3):
            t0 = time.time()
            summary, npz = attempt_step(t, gammas, cur_seed, attempt)
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
        ps = rep.get("posteriors", [None]*3)
        print(f"  ACCEPTED: resid={best_residual(accepted_summary):.3e}  "
              f"1-R²={accepted_summary['revelation_deficit']:.4e}  "
              f"slope={slope:.4f}  max_FR={accepted_summary['max_fr_error']:.4f}  "
              f"price={rep.get('price'):.5f}  posts={[round(x,4) for x in ps]}",
              flush=True)
        rows.append((t, gammas, accepted_summary, "ACCEPTED"))
        cur_seed = accepted_npz

    print("\n" + "=" * 120)
    print(f"{'t':>6} | {'gammas':>16} | {'resid':>11} | {'1-R²':>11} | "
          f"{'slope':>8} | {'max FR':>8} | {'price':>9} | {'mu1':>7} | {'mu2':>7} | {'mu3':>7}")
    print("-" * 120)
    table = []
    for t, gammas, summary, status in rows:
        if summary is None:
            print(f"{t:>6g} | {gammas_str(gammas):>16} | -- | -- | -- | -- | -- | -- | -- | --   {status}")
            continue
        rep = summary.get("representative_realization", {})
        ps = rep.get("posteriors", [None]*3)
        print(f"{t:>6g} | {gammas_str(gammas):>16} | "
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
            "posteriors": ps, "status": status,
        })
    out = RESULTS / "het_tiny_ladder_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
