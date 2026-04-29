"""γ ladder at G=15, smooth Φ (fast vectorized), h=0.005.

Continuation from G=12 γ-ladder converged tensors interpolated to G=15.
"""

import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth_fast.py"
G = 15; H = 0.005

# Use G=12 γ=2 result as initial seed (interpolated to G=15 via solver)
GAMMAS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0]
SEED = RESULTS / "G12_tau2_smoothfasth0.005_het0.1_0.1_0.1_g0.1_PR_prices.npz"


def call(label, gammas, seed):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed),
        "--label", label, "--h", str(H),
        "--max-iter", "500", "--tol", "1e-12",
        "--damping", "0.3", "--anderson", "5", "--anderson-beta", "0.7",
        "--symmetric",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau2_smoothfasth{H:g}_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def main():
    cur_seed = SEED
    rows = []
    print(f"{'γ':>6} | {'iters':>5} | {'resid':>11} | {'1-R²':>11} | "
          f"{'slope':>7} | {'max_FR':>8} | {'price':>9} | {'time':>7}")
    print("-" * 80)
    for gam in GAMMAS:
        gammas = (gam, gam, gam)
        label = f"g15ladder_g{gam:g}"
        t0 = time.time()
        s = call(label, gammas, cur_seed)
        elapsed = time.time() - t0
        if s is None:
            print(f"{gam:>6g} | FAILED ({elapsed:.1f}s)")
            break
        rep = s.get("representative_realization", {})
        print(f"{gam:>6g} | {s['iterations']:>5} | {s['residual_inf']:>11.3e} | "
              f"{s['revelation_deficit']:>11.4e} | {s['slope']:>7.4f} | "
              f"{s['max_fr_error']:>8.4f} | {rep.get('price'):>9.5f} | {elapsed:>7.1f}",
              flush=True)
        rows.append({"gamma": gam, **s})
        cur_seed = RESULTS / f"G{G}_tau2_smoothfasth{H:g}_het{gam:g}_{gam:g}_{gam:g}_{label}_prices.npz"
        if not cur_seed.exists():
            print(f"  seed save missing: {cur_seed.name}")
            break

    out_path = RESULTS / f"G{G}_smooth_gamma_ladder_h{H:g}.json"
    with open(out_path, "w") as f:
        json.dump({"G": G, "h": H, "rows": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")

    if rows:
        baseline = rows[-1]["revelation_deficit"]
        print(f"\nbaseline γ={rows[-1]['gamma']}: 1-R² = {baseline:.4e}")
        print(f"\n{'γ':>6} | {'1-R²':>11} | {'NET PR':>12} | {'slope':>7}")
        for r in rows:
            net = r["revelation_deficit"] - baseline
            print(f"{r['gamma']:>6g} | {r['revelation_deficit']:>11.4e} | "
                  f"{net:>+12.4e} | {r['slope']:>7.4f}")


if __name__ == "__main__":
    main()
