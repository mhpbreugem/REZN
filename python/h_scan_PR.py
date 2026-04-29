"""Bandwidth scan at G=6 for γ=0.1 (strongest PR) and γ=20 (CARA baseline).

If NET PR (γ=0.1 minus γ=20) survives as h → 0, that is hard evidence of
genuine PR independent of kernel bias.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth.py"
G = 6

H_VALUES = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
# seed: γ=0.1 ladder result
SEED_01 = RESULTS / "G6_tau2_smoothh0.005_het0.1_0.1_0.1_gladder_down_g0.1_prices.npz"
SEED_20 = RESULTS / "G6_tau2_smoothh0.005_het20_20_20_gladder_up_g20_prices.npz"


def call(label, gammas, seed_npz, h):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed_npz),
        "--label", label, "--h", str(h),
        "--method", "picard", "--max-iter", "1000", "--tol", "1e-14",
        "--damping", "0.3", "--anderson", "5", "--anderson-beta", "0.7",
        "--symmetric",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau2_smoothh{h:g}_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def main():
    rows = []
    for h in H_VALUES:
        for tag, gam, seed in [("g0.1", (0.1, 0.1, 0.1), SEED_01),
                                ("g20",  (20.0, 20.0, 20.0), SEED_20)]:
            label = f"hscan_{tag}_h{h:g}"
            t0 = time.time()
            s = call(label, gam, seed, h)
            elapsed = time.time() - t0
            if s is None:
                print(f"  h={h:g} {tag}: FAILED ({elapsed:.1f}s)")
                continue
            print(f"  h={h:7g}  γ={gam[0]:g}  iters={s['iterations']:4d}  "
                  f"resid={s['residual_inf']:.3e}  1-R²={s['revelation_deficit']:.6e}  "
                  f"slope={s['slope']:.4f}  ({elapsed:.1f}s)", flush=True)
            rows.append({"h": h, "gamma": gam[0], **s})

    # NET vs h
    by_h = {}
    for r in rows:
        by_h.setdefault(r["h"], {})[r["gamma"]] = r
    print("\nh → 0 extrapolation:")
    print(f"{'h':>8} | {'1-R² γ=0.1':>12} | {'1-R² γ=20':>12} | {'NET (PR)':>12} | "
          f"{'slope γ=0.1':>11} | {'slope γ=20':>11}")
    for h in sorted(by_h.keys(), reverse=True):
        d = by_h[h]
        r01 = d.get(0.1)
        r20 = d.get(20.0)
        if r01 and r20:
            net = r01["revelation_deficit"] - r20["revelation_deficit"]
            print(f"{h:>8g} | {r01['revelation_deficit']:>12.5e} | "
                  f"{r20['revelation_deficit']:>12.5e} | {net:>+12.5e} | "
                  f"{r01['slope']:>11.4f} | {r20['slope']:>11.4f}")

    out = RESULTS / "smooth_h_scan_PR_G6.json"
    with open(out, "w") as f:
        json.dump({"G": G, "rows": rows}, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
