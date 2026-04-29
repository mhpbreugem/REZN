"""G-scan with smooth (kernel) Φ at homogeneous gamma=(2,2,2), tau=2.

For each G in {6, 9, 12, 15}, run a bandwidth sub-sweep to study how the
fixed-point's 1-R² and slope behave as h → 0 at fixed G, and as G → ∞.

This is the cleanest test of whether the paper's PR claim survives at the
G → ∞ limit, or whether 1-R² shrinks to zero with grid refinement.

Always uses the no-learning seed at each G (interpolated via the solver).
"""

import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth.py"
CURSOR_SEED = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"

G_VALUES = [6, 9, 12, 15]
H_VALUES = [0.05, 0.02, 0.01, 0.005, 0.002]


def run(G, h, seed):
    label = f"gscan_G{G}_h{h:g}"
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", "2,2,2",
        "--seed-array", str(seed),
        "--label", label, "--h", str(h),
        "--method", "picard",
        "--max-iter", "400", "--tol", "1e-14",
        "--damping", "0.3", "--anderson", "5", "--anderson-beta", "0.7",
        "--symmetric",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None, elapsed
    summary_path = RESULTS / f"G{G}_tau2_smoothh{h:g}_het2_2_2_{label}_summary.json"
    if not summary_path.exists():
        return None, elapsed
    with open(summary_path) as f:
        return json.load(f), elapsed


def main():
    results = []
    print(f"{'G':>3} | {'h':>7} | {'iters':>5} | {'resid':>11} | {'1-R²':>11} | "
          f"{'slope':>8} | {'max_FR':>9} | {'time(s)':>7}")
    print("-" * 90)
    for G in G_VALUES:
        for h in H_VALUES:
            s, elapsed = run(G, h, CURSOR_SEED)
            if s is None:
                print(f"{G:>3} | {h:>7g} | FAILED ({elapsed:.1f}s)")
                continue
            print(f"{G:>3} | {h:>7g} | {s['iterations']:>5} | "
                  f"{s['residual_inf']:>11.3e} | {s['revelation_deficit']:>11.4e} | "
                  f"{s['slope']:>8.4f} | {s['max_fr_error']:>9.4f} | "
                  f"{elapsed:>7.1f}", flush=True)
            results.append({"G": G, "h": h, "iters": s["iterations"],
                            "resid": s["residual_inf"],
                            "1-R2": s["revelation_deficit"],
                            "slope": s["slope"],
                            "max_FR": s["max_fr_error"],
                            "elapsed": elapsed})

    # Persist
    out = RESULTS / "smooth_g_scan_table.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # h→0 extrapolation per G (linear in h²)
    print("\n--- h → 0 extrapolation per G (linear fit in h²) ---")
    print(f"{'G':>3} | {'1-R² @ h→0':>12} | {'slope @ h→0':>12}")
    for G in G_VALUES:
        rows = [r for r in results if r["G"] == G]
        if len(rows) < 2: continue
        h2 = np.array([r["h"] ** 2 for r in rows])
        R2 = np.array([r["1-R2"] for r in rows])
        sl = np.array([r["slope"] for r in rows])
        # linear fit y = a + b*h²
        a_R2 = np.polyfit(h2, R2, 1)
        a_sl = np.polyfit(h2, sl, 1)
        print(f"{G:>3} | {a_R2[1]:>12.4e} | {a_sl[1]:>12.4f}")


if __name__ == "__main__":
    main()
