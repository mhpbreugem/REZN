"""G-scan with soft-Jacobian-corrected linear-interp Φ.

Same pattern as g_scan_smooth.py but uses the Jacobian-corrected method.
Hard Jacobian (1/|slope|) is unstable — soft (1/sqrt(slope^2+reg^2)) is
controllable via reg.

For each G, run a regularization sub-sweep reg ∈ {0.05, 0.02, 0.01, 0.005}.
"""

import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_jac_soft.py"
CURSOR_SEED = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"

G_VALUES = [6, 9, 12]
REG_VALUES = [0.05, 0.02, 0.01, 0.005]


def run(G, reg, seed):
    label = f"gscan_G{G}_reg{reg:g}"
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", "2,2,2",
        "--seed-array", str(seed),
        "--label", label, "--reg", str(reg),
        "--max-iter", "600", "--tol", "1e-13",
        "--damping", "0.15", "--anderson", "8", "--anderson-beta", "0.5",
        "--symmetric",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None, elapsed
    summary_path = RESULTS / f"G{G}_tau2_jacsoft_reg{reg:g}_het2_2_2_{label}_summary.json"
    if not summary_path.exists():
        return None, elapsed
    with open(summary_path) as f:
        return json.load(f), elapsed


def main():
    results = []
    print(f"{'G':>3} | {'reg':>7} | {'iters':>5} | {'resid':>11} | {'1-R²':>11} | "
          f"{'slope':>8} | {'max_FR':>9} | {'time(s)':>7}")
    print("-" * 90)
    for G in G_VALUES:
        for reg in REG_VALUES:
            s, elapsed = run(G, reg, CURSOR_SEED)
            if s is None:
                print(f"{G:>3} | {reg:>7g} | FAILED ({elapsed:.1f}s)")
                continue
            print(f"{G:>3} | {reg:>7g} | {s['iterations']:>5} | "
                  f"{s['residual_inf']:>11.3e} | {s['revelation_deficit']:>11.4e} | "
                  f"{s['slope']:>8.4f} | {s['max_fr_error']:>9.4f} | "
                  f"{elapsed:>7.1f}", flush=True)
            results.append({"G": G, "reg": reg, **s})
    out = RESULTS / "jacsoft_g_scan_table.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
