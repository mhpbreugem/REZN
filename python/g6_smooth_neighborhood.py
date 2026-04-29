"""Search neighborhood of homogeneous-tau PR fixed point with smooth Φ.

Steps:
  1. Bandwidth sweep h in {0.001, 0.002, 0.005, 0.01, 0.02} at gamma=(2,2,2)
     to confirm the smooth fixed point is bandwidth-robust.
  2. From the converged smooth-homog tensor, add three perturbation patterns
     and run smooth Picard from each to convergence:
        a. iid Gaussian at 1e-6
        b. Anti-symmetric (i<->k flip)  at 1e-3
        c. Tilt along (u1 - u3)         at 1e-3
  3. Report each converged tensor's slope, 1-R^2, posteriors at canonical.
"""

import json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth.py"
HOMOG_NEW_TENSOR = REPO / "results/full_ree/G6_tau2_smoothh0.005_het2_2_2_smooth_smoke_homog_prices.npz"


def call_solver(label, gammas, seed_npz, h, **kw):
    cmd = [
        "python3", str(SOLVER),
        "--G", "6", "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed_npz),
        "--label", label, "--h", str(h),
        "--method", kw.get("method", "picard"),
        "--max-iter", str(kw.get("max_iter", 200)),
        "--tol", str(kw.get("tol", 1e-14)),
        "--damping", str(kw.get("damping", 0.3)),
        "--anderson", str(kw.get("anderson", 5)),
        "--anderson-beta", str(kw.get("anderson_beta", 0.7)),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout[-1000:])
        sys.stdout.write("\nSTDERR:\n" + out.stderr[-1000:])
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    summary_path = RESULTS / f"G6_tau2_smoothh{h:g}_het{g_str}_{label}_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def perturbed_seed_path(name):
    return REPO / f"results/full_ree/G6_tau2_smooth_perturb_{name}.npz"


def make_perturbed_seed(base_npz, kind, scale, rng_seed=42):
    base = np.load(base_npz)
    P = base["P"].copy(); grid = base["grid"]; G = len(grid)
    if kind == "iid":
        rng = np.random.default_rng(rng_seed)
        delta = rng.standard_normal(P.shape) * scale
    elif kind == "antisym":
        # Anti-symmetric in (i, k) swap
        rng = np.random.default_rng(rng_seed)
        D = rng.standard_normal(P.shape)
        delta = (D - np.transpose(D, (2, 1, 0))) * scale * 0.5
    elif kind == "tilt":
        # Tilt linearly with u1 - u3
        u = grid
        U1, _, U3 = np.meshgrid(u, u, u, indexing="ij")
        delta = (U1 - U3) * scale
    else:
        raise ValueError(kind)
    P_pert = np.clip(P + delta, 1e-8, 1 - 1e-8)
    out = perturbed_seed_path(f"{kind}_{scale:g}")
    np.savez(out, P=P_pert, grid=grid)
    return out, float(np.max(np.abs(delta)))


def main():
    # --- Step 1: bandwidth sweep at gamma=(2,2,2) ---
    print("=" * 80)
    print("STEP 1: bandwidth sweep at homogeneous gamma=(2,2,2)")
    print("=" * 80, flush=True)
    bw_results = []
    seed = REPO / "results/full_ree/G6_tau2_gamma2_up_picard_0_prices.npz"
    for h in [0.02, 0.01, 0.005, 0.002, 0.001]:
        t0 = time.time()
        s = call_solver(f"bw_{h:g}", (2.0, 2.0, 2.0), seed, h, max_iter=400)
        elapsed = time.time() - t0
        if s is None:
            print(f"  h={h:g}: FAILED")
            continue
        bw_results.append({"h": h, **s})
        print(f"  h={h:6g}  resid={s['residual_inf']:.3e}  "
              f"1-R²={s['revelation_deficit']:.4e}  slope={s['slope']:.4f}  "
              f"max_FR={s['max_fr_error']:.4f}  ({elapsed:.1f}s)", flush=True)

    # --- Step 2: perturbations from the smooth homog tensor ---
    if not HOMOG_NEW_TENSOR.exists():
        print(f"\n[skip neighborhood search: {HOMOG_NEW_TENSOR.name} missing]")
        return

    print("\n" + "=" * 80)
    print(f"STEP 2: neighborhood search from {HOMOG_NEW_TENSOR.name}")
    print("=" * 80, flush=True)

    h = 0.005
    perturbations = [
        ("iid", 1e-6),
        ("iid", 1e-4),
        ("iid", 1e-2),
        ("antisym", 1e-4),
        ("antisym", 1e-2),
        ("tilt", 1e-3),
        ("tilt", 1e-1),
    ]
    perturb_results = []
    for kind, scale in perturbations:
        seed_path, max_pert = make_perturbed_seed(HOMOG_NEW_TENSOR, kind, scale)
        print(f"\n  perturb {kind} scale={scale:g} (max |Δ|={max_pert:.2e})", flush=True)
        t0 = time.time()
        s = call_solver(f"pert_{kind}_{scale:g}", (2.0, 2.0, 2.0), seed_path, h,
                        max_iter=400)
        elapsed = time.time() - t0
        if s is None:
            print(f"    FAILED ({elapsed:.1f}s)"); continue
        rep = s.get("representative_realization", {})
        ps = rep.get("posteriors", [None]*3)
        print(f"    iters={s['iterations']}  resid={s['residual_inf']:.3e}  "
              f"1-R²={s['revelation_deficit']:.4e}  slope={s['slope']:.4f}  "
              f"price@canon={rep.get('price'):.6f}  posts={[round(p,4) for p in ps]}  "
              f"({elapsed:.1f}s)", flush=True)
        perturb_results.append({"kind": kind, "scale": scale, **s})

    # --- Save merged table ---
    out = RESULTS / "smooth_neighborhood_table.json"
    with open(out, "w") as f:
        json.dump({"bandwidth_sweep": bw_results,
                   "perturbations": perturb_results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
