"""G=12 PR test: γ=0.1 vs γ=20 at h=0.005 with longer max_iter.

The fast vectorized smooth solver at G=12 takes ~2.4s/iter. With 600 iters
it should reach a residual ~1e-10 even for slow Picard.
"""

import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth_fast.py"
G = 12; H = 0.005

# Seeds: γ=0.1 and γ=20 G=6 converged tensors (will be interpolated)
SEED_01 = RESULTS / "G6_tau2_smoothh0.005_het0.1_0.1_0.1_gladder_down_g0.1_prices.npz"
SEED_20 = RESULTS / "G6_tau2_smoothh0.005_het20_20_20_gladder_up_g20_prices.npz"


def run(label, gammas, seed):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed),
        "--label", label, "--h", str(H),
        "--max-iter", "800", "--tol", "1e-12",
        "--damping", "0.3", "--anderson", "5", "--anderson-beta", "0.7",
        "--symmetric", "--progress",
    ]
    print(f"\nRunning: γ={gammas}, label={label}", flush=True)
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-2000:] + "\nSTDERR:\n" + out.stderr[-2000:])
        return None, elapsed
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau2_smoothfasth{H:g}_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None, elapsed
    with open(sf) as f:
        return json.load(f), elapsed


def main():
    rows = []
    for tag, gam, seed in [("g0.1_PR", (0.1, 0.1, 0.1), SEED_01),
                            ("g20_PR",  (20.0, 20.0, 20.0), SEED_20)]:
        s, elapsed = run(tag, gam, seed)
        if s is None:
            print(f"  {tag}: FAILED ({elapsed:.1f}s)")
            continue
        rep = s.get("representative_realization", {})
        print(f"  {tag}: iters={s['iterations']} resid={s['residual_inf']:.3e} "
              f"1-R²={s['revelation_deficit']:.6e} slope={s['slope']:.4f} "
              f"max_FR={s['max_fr_error']:.4f} ({elapsed:.1f}s)", flush=True)
        rows.append({"gammas": list(gam), **s})
    if len(rows) == 2:
        net = rows[0]["revelation_deficit"] - rows[1]["revelation_deficit"]
        print(f"\nNET PR (γ=0.1 minus γ=20) = {net:+.6e}")
    out = RESULTS / f"G{G}_PR_test_h{H:g}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
