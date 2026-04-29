"""Het-γ ladder at G=6 with smooth Φ.

Starting from the converged γ=(0.1,0.1,0.1) homogeneous tensor (PR-strongest
case), tilts γ heterogeneity using gamma(t) = (γ0 - dγ*t, γ0, γ0 + dγ*t).
γ0 = 0.1 (highest PR), dγ = 0.05 → at t=1: γ = (0.05, 0.1, 0.15).
Then a stronger tilt up to γ=(0.025, 0.1, 0.4) at higher t.

Without --symmetric (heterogeneous γ breaks permutation symmetry).
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth.py"
G = 6; H = 0.005

# starting tensor: γ=(0.1, 0.1, 0.1)
START = RESULTS / f"G{G}_tau2_smoothh{H:g}_het0.1_0.1_0.1_gladder_down_g0.1_prices.npz"

# tilt: for t in T_LADDER, gammas = base + tilt*t
BASE = (0.1, 0.1, 0.1)

def gammas_at(t):
    """Multiplicative tilt: γ_1 = γ_2 / r, γ_3 = γ_2 * r, where r grows with t."""
    r = 1.0 + t   # at t=0: r=1, no het. At t=2: r=3, γ=(0.033, 0.1, 0.3).
    return (BASE[1] / r, BASE[1], BASE[1] * r)


T_LADDER = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]


def gammas_str(g):
    return ",".join(f"{x:g}" for x in g)


def call(label, gammas, seed_npz, max_iter=600, tol=1e-13, damping=0.3):
    cmd = [
        "python3", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", gammas_str(gammas),
        "--seed-array", str(seed_npz),
        "--label", label, "--h", str(H),
        "--method", "picard",
        "--max-iter", str(max_iter), "--tol", str(tol),
        "--damping", str(damping), "--anderson", "5", "--anderson-beta", "0.7",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:] + "\nSTDERR:\n" + out.stderr[-1000:])
        return None
    sf = RESULTS / f"G{G}_tau2_smoothh{H:g}_het{gammas_str(gammas).replace(',', '_')}_{label}_summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        return json.load(f)


def main():
    if not START.exists():
        print(f"FATAL: {START} missing"); return
    print(f"Starting from {START.name}")
    cur_seed = START
    rows = []
    for t in T_LADDER:
        gammas = gammas_at(t)
        label = f"hetladder_t{t:g}"
        t0 = time.time()
        s = call(label, gammas, cur_seed)
        elapsed = time.time() - t0
        if s is None:
            print(f"  t={t:g}  γ={gammas}  FAILED ({elapsed:.1f}s)")
            break
        rep = s.get("representative_realization", {})
        ps = rep.get("posteriors", [None]*3)
        print(f"  t={t:g}  γ=({gammas[0]:.4f},{gammas[1]:.2f},{gammas[2]:.4f})  "
              f"iters={s['iterations']}  resid={s['residual_inf']:.3e}  "
              f"1-R²={s['revelation_deficit']:.6e}  slope={s['slope']:.4f}  "
              f"price={rep.get('price'):.5f}  posts={[round(x,4) for x in ps]}  "
              f"({elapsed:.1f}s)", flush=True)
        rows.append({"t": t, "gammas": list(gammas),
                     "iters": s["iterations"], "resid": s["residual_inf"],
                     "1-R2": s["revelation_deficit"], "slope": s["slope"],
                     "max_FR": s["max_fr_error"],
                     "price": rep.get("price"), "posteriors": ps})
        cur_seed = RESULTS / f"G{G}_tau2_smoothh{H:g}_het{gammas_str(gammas).replace(',', '_')}_{label}_prices.npz"
        if not cur_seed.exists():
            print(f"    seed save missing: {cur_seed.name}")
            break
    out_path = RESULTS / "smooth_het_ladder_G6_h0.005.json"
    with open(out_path, "w") as f:
        json.dump({"G": G, "h": H, "base": list(BASE), "rows": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
