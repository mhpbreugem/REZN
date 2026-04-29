"""
gamma ladder at G=6, tau=2: starting from the converged gamma=0.5 tensor,
step gamma down to 0.1 by reducing it gradually and continuing each solve
from the previous tensor.

Each step uses Picard with Anderson acceleration, then a Newton polish.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver.py"

# starting tensor: the polished G=6, gamma=0.5 tensor from the prior step
GAMMA_LADDER = [
    (0.5,  "results/full_ree/G6_tau2_gamma0.5_G6_newton_polish_prices.npz"),
    (0.4,  None),
    (0.3,  None),
    (0.25, None),
    (0.2,  None),
    (0.15, None),
    (0.1,  None),
]


def run(label, gamma, seed_npz, method="picard", damping=0.3, anderson=5,
        anderson_beta=0.7, max_iter=600, tol=1e-13,
        gmres_max_iter=80, gmres_tol=1e-10, fd_eps=1e-7,
        newton_damping=1.0):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gamma", str(gamma),
        "--seed", "array",
        "--seed-array", str(seed_npz),
        "--label", label,
        "--method", method,
        "--max-iter", str(max_iter),
        "--damping", str(damping),
        "--tol", str(tol),
        "--save-array",
        "--progress",
    ]
    if method == "picard":
        cmd += ["--anderson", str(anderson), "--anderson-beta", str(anderson_beta)]
    if method == "newton":
        cmd += ["--gmres-max-iter", str(gmres_max_iter),
                "--gmres-tol", str(gmres_tol),
                "--fd-eps", str(fd_eps),
                "--newton-damping", str(newton_damping)]
    print(f"\n+++ running: gamma={gamma}, method={method}, label={label}", flush=True)
    print(f"    seed: {seed_npz}", flush=True)
    print(f"    cmd:  {' '.join(cmd)}", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout[-2500:])
    if out.returncode != 0:
        print("STDERR:", out.stderr[-2000:])
        return None
    summary_path = RESULTS / f"G6_tau2_gamma{gamma}_{label}_summary.json"
    if not summary_path.exists():
        print(f"  summary file not found: {summary_path}")
        return None
    with open(summary_path) as f:
        return json.load(f)


def main():
    cur_seed = REPO / GAMMA_LADDER[0][1]
    rows = []
    for gamma, seed_override in GAMMA_LADDER:
        if seed_override is not None:
            seed = REPO / seed_override
        else:
            seed = cur_seed
        # picard step
        picard_label = f"ladder_picard"
        picard_summary = run(picard_label, gamma, seed, method="picard",
                              damping=0.3, anderson=5, anderson_beta=0.7,
                              max_iter=600, tol=1e-13)
        if picard_summary is None:
            print(f"  Picard failed at gamma={gamma}; stopping ladder")
            break
        picard_npz = RESULTS / f"G6_tau2_gamma{gamma}_{picard_label}_prices.npz"

        # newton polish
        polish_label = f"ladder_polish"
        polish_summary = run(polish_label, gamma, picard_npz, method="newton",
                              max_iter=15, tol=1e-15,
                              gmres_max_iter=80, gmres_tol=1e-12, fd_eps=1e-7,
                              newton_damping=1.0)
        polish_npz = RESULTS / f"G6_tau2_gamma{gamma}_{polish_label}_prices.npz"

        if polish_summary is not None and polish_npz.exists():
            cur_seed = polish_npz
            best = polish_summary
        else:
            cur_seed = picard_npz
            best = picard_summary

        rep = best.get("representative_realization", {})
        rows.append({
            "gamma": gamma,
            "picard_iters": picard_summary["iterations"],
            "picard_resid": picard_summary["residual_inf"],
            "polish_iters": polish_summary["iterations"] if polish_summary else None,
            "best_resid": best["residual_inf"],
            "1-R2": best["revelation_deficit"],
            "max_fr_error": best["max_fr_error"],
            "price_canon": rep.get("price"),
            "fr_price_canon": rep.get("fr_price"),
            "mu1": rep.get("posteriors", [None])[0],
            "mu2": rep.get("posteriors", [None, None])[1],
        })

    # final table
    print("\n" + "=" * 90)
    print(f"{'gamma':>6} | {'resid':>11} | {'1-R²':>11} | {'max FR':>9} | {'price':>9} | {'FR':>9} | {'μ1':>8} | {'μ2':>8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['gamma']:>6.3f} | {r['best_resid']:>11.4e} | {r['1-R2']:>11.4e} | "
              f"{r['max_fr_error']:>9.4f} | {r['price_canon']:>9.5f} | "
              f"{r['fr_price_canon']:>9.5f} | {r['mu1']:>8.5f} | {r['mu2']:>8.5f}")

    # save table to results/
    out_path = RESULTS / "ladder_gamma_table.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
