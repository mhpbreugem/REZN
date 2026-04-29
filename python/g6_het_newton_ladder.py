"""
Newton-Krylov het ladder at G=6, tau=2.

Starting from the homogeneous gamma=2 polished tensor + 1e-10 anti-symmetric
perturbation, increment heterogeneity in tiny steps.  At each rung, use
Newton-Krylov (FD Jacobian, GMRES, line search) instead of Picard.

ACCEPT_TOL = 5e-3 -- this is the linear-interp floor for the het case at
G=6 without symmetrization. Iterating below this would require a smoother
Φ (cubic interp or Gaussian kernel).
"""

import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het.py"

ACCEPT_TOL = 5.0e-3
T_LADDER = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
HOMOG_TENSOR = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"
PERTURBED_SEED = REPO / "results/full_ree/G6_tau2_gamma2_perturbed_1em10.npz"
PERTURB_SCALE = 1.0e-10
PERTURB_RNG_SEED = 17


def gammas_at(t):
    return (2.0 - t, 2.0, 2.0 + 2.0 * t)


def gammas_str(g):
    return ",".join(f"{x:g}" for x in g)


def make_perturbed_seed():
    base = np.load(HOMOG_TENSOR)
    P = base["P"].copy()
    grid = base["grid"]
    rng = np.random.default_rng(PERTURB_RNG_SEED)
    noise = rng.standard_normal(P.shape) * PERTURB_SCALE
    P_pert = np.clip(P + noise, 1e-8, 1 - 1e-8)
    np.savez(PERTURBED_SEED, P=P_pert, grid=grid)
    print(f"  Perturbed seed: scale={PERTURB_SCALE:.1e}, max |Δ|={np.max(np.abs(noise)):.3e}",
          flush=True)


def call_newton(label, gammas, seed_npz, max_iter=40, tol=1e-4,
                gmres_max_iter=120, gmres_tol=5e-4, fd_eps=1e-4,
                newton_damping=1.0):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gammas", gammas_str(gammas),
        "--seed-array", str(seed_npz),
        "--label", label,
        "--method", "newton",
        "--max-iter", str(max_iter), "--tol", str(tol),
        "--gmres-max-iter", str(gmres_max_iter),
        "--gmres-tol", str(gmres_tol), "--fd-eps", str(fd_eps),
        "--newton-damping", str(newton_damping),
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:])
        sys.stdout.write("\nSTDERR:\n" + out.stderr[-1000:])
        return None, elapsed
    summary_path = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_summary.json"
    if not summary_path.exists():
        return None, elapsed
    with open(summary_path) as f:
        return json.load(f), elapsed


def best_resid(s):
    return float(s.get("residual_inf", float("inf"))) if s is not None else float("inf")


def main():
    if not HOMOG_TENSOR.exists():
        print(f"FATAL: {HOMOG_TENSOR} missing"); return
    make_perturbed_seed()

    cur_seed = PERTURBED_SEED
    rows = []
    for t in T_LADDER:
        gammas = gammas_at(t)
        print(f"\n=== t={t:g}  gammas={gammas}  seed={cur_seed.name} ===", flush=True)
        accepted = None; accepted_npz = None
        configs = [
            dict(max_iter=40, gmres_max_iter=120, gmres_tol=5e-4, fd_eps=1e-4),
            dict(max_iter=60, gmres_max_iter=160, gmres_tol=1e-4, fd_eps=5e-5),
            dict(max_iter=80, gmres_max_iter=200, gmres_tol=5e-5, fd_eps=1e-5),
        ]
        for k, cfg in enumerate(configs):
            label = f"newtladder_t{t:g}_{k}"
            summary, elapsed = call_newton(label, gammas, cur_seed,
                                            tol=1e-5, **cfg)
            resid = best_resid(summary)
            R2 = summary.get("revelation_deficit") if summary else None
            R2_str = f"{R2:.4e}" if R2 is not None else "NA"
            print(f"  attempt {k}: cfg={cfg}", flush=True)
            print(f"             resid={resid:.3e}  1-R²={R2_str}  ({elapsed:.1f}s)",
                  flush=True)
            if resid <= ACCEPT_TOL:
                accepted = summary
                accepted_npz = RESULTS / f"G6_tau2_het{gammas_str(gammas).replace(',', '_')}_{label}_prices.npz"
                break

        if accepted is None:
            print(f"  HALT t={t:g}: best resid={resid:.3e}", flush=True)
            rows.append((t, gammas, summary, f"REJECTED ({resid:.3e})"))
            break
        rep = accepted.get("representative_realization", {})
        slope = accepted.get("slope", float("nan"))
        ps = rep.get("posteriors", [None]*3)
        print(f"  ACCEPTED: resid={best_resid(accepted):.3e}  "
              f"1-R²={accepted['revelation_deficit']:.4e}  "
              f"slope={slope:.4f}  max_FR={accepted['max_fr_error']:.4f}  "
              f"price={rep.get('price'):.5f}  posts={[round(x,4) for x in ps]}",
              flush=True)
        rows.append((t, gammas, accepted, "ACCEPTED"))
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
    out = RESULTS / "het_newton_ladder_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
