"""Gamma ladder with smooth Φ at G=12 (or G=9 fallback), tau=2.

Uses the converged smooth fixed point at gamma=(2,2,2) as starting tensor,
then ramps gamma down through {1.5, 1.0, 0.75, 0.5, 0.3, 0.1} and up through
{3, 5, 10, 20}. Reports 1-R^2 and slope per rung.

Bandwidth fixed at h=0.005 (the most informative non-divergent value).
"""

import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth.py"
G_USE = 9    # the largest where we have a converged h=0.005 seed
H = 0.005

# starting tensor: prefer G_USE/h=0.005 from g_scan; fall back to interpolating cursor
def find_start_tensor():
    for G in (12, 9, 6):
        cand = RESULTS / f"G{G}_tau2_smoothh{H:g}_het2_2_2_gscan_G{G}_h{H:g}_prices.npz"
        if cand.exists():
            return G, cand
    return 6, REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"


GAMMAS_DOWN = [2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10]
GAMMAS_UP   = [2.0, 3.0, 5.0, 10.0, 20.0]


def call(label, gammas, seed_npz, G):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed_npz),
        "--label", label, "--h", str(H),
        "--method", "picard",
        "--max-iter", "600", "--tol", "1e-13",
        "--damping", "0.3", "--anderson", "5", "--anderson-beta", "0.7",
    ]
    if gammas[0] == gammas[1] == gammas[2]:
        cmd.append("--symmetric")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau2_smoothh{H:g}_het{g_str}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def run_chain(direction, gammas_list, start_seed, G):
    rows = []
    cur_seed = start_seed
    for gam in gammas_list:
        gammas = (gam, gam, gam)
        label = f"gladder_{direction}_g{gam:g}"
        t0 = time.time()
        s = call(label, gammas, cur_seed, G)
        elapsed = time.time() - t0
        if s is None:
            print(f"  γ={gam:g}  FAILED ({elapsed:.1f}s)")
            break
        rep = s.get("representative_realization", {})
        print(f"  γ={gam:g}  iters={s['iterations']}  resid={s['residual_inf']:.3e}  "
              f"1-R²={s['revelation_deficit']:.4e}  slope={s['slope']:.4f}  "
              f"max_FR={s['max_fr_error']:.4f}  price={rep.get('price'):.5f}  "
              f"({elapsed:.1f}s)", flush=True)
        rows.append({"gamma": gam, "iters": s["iterations"],
                     "resid": s["residual_inf"],
                     "1-R2": s["revelation_deficit"],
                     "slope": s["slope"],
                     "max_FR": s["max_fr_error"],
                     "price": rep.get("price")})
        cur_seed = RESULTS / f"G{G}_tau2_smoothh{H:g}_het{gam:g}_{gam:g}_{gam:g}_{label}_prices.npz"
        if not cur_seed.exists():
            print(f"    seed save missing: {cur_seed.name}")
            break
    return rows


def main():
    G, start_seed = find_start_tensor()
    print(f"Using G={G}, h={H}, starting seed = {start_seed.name}")

    print("\n--- ladder DOWN from γ=2 ---")
    down = run_chain("down", GAMMAS_DOWN, start_seed, G)

    print("\n--- ladder UP from γ=2 ---")
    up = run_chain("up", GAMMAS_UP, start_seed, G)

    out = {"G": G, "h": H, "down": down, "up": up}
    out_path = RESULTS / f"smooth_gamma_ladder_G{G}_h{H:g}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    # combined table
    print("\n=== summary ===")
    print(f"{'γ':>6} | {'1-R²':>11} | {'slope':>7} | {'price':>9} | {'resid':>11}")
    for r in (down + up):
        print(f"{r['gamma']:>6g} | {r['1-R2']:>11.4e} | {r['slope']:>7.4f} | "
              f"{r['price']:>9.5f} | {r['resid']:>11.3e}")


if __name__ == "__main__":
    main()
