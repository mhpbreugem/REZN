"""h-bandwidth ladder at G=20, τ=2, γ=0.1.

Starts from the deeply-converged G=20 γ=0.1 h=0.01 tensor and shrinks h
geometrically: 0.01 → 0.005 → 0.002 → 0.001 → 0.0005 → 0.0002 → 0.0001.

Strict tolerance: each rung must reach residual ≤ 1e-13.

Also runs the same h ladder at γ=20 (CARA limit) to give the artifact
baseline for NET PR at each h.
"""

import json, subprocess, sys, time, csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
SOLVER = REPO / "python" / "full_ree_solver_het_smooth_fast.py"
G = 20

H_LADDER = [
    # ~5% steps in the high-h region where iterate moves a lot
    0.0050, 0.00475, 0.00450, 0.00425, 0.00400, 0.00375, 0.00350,
    0.00325, 0.00300, 0.00280, 0.00260, 0.00240, 0.00220, 0.00200,
    0.00180, 0.00160, 0.00140, 0.00120, 0.00100,
    # ~20% steps in the noise-floor region
    0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,
]
ACCEPT_TOL = 1.0e-13

# Seeds: deeply converged G=20 h=0.005 tensors (h=0.005 already gave us
# a clean fixed point in the previous run).
SEED_G01 = RESULTS / "G20_tau2_smoothfasth0.005_het0.1_0.1_0.1_hladderG20_g0.1_h0.005_a0_prices.npz"
SEED_G20 = RESULTS / "G20_tau2_smoothfasth0.01_het20_20_20_g20_G20_h01_prices.npz"


def call(label, gammas, h, seed, max_iter=600, tol=1e-14,
         damping=0.3, anderson=5, anderson_beta=0.7):
    cmd = [
        "python3", "-u", str(SOLVER),
        "--G", str(G), "--umax", "2", "--tau", "2",
        "--gammas", ",".join(f"{g:g}" for g in gammas),
        "--seed-array", str(seed),
        "--label", label, "--h", str(h),
        "--max-iter", str(max_iter), "--tol", str(tol),
        "--damping", str(damping),
        "--anderson", str(anderson), "--anderson-beta", str(anderson_beta),
        "--symmetric",
        "--progress",
    ]
    print(f"  >> running {label}: γ={gammas}, h={h}, seed={Path(seed).name}", flush=True)
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        return None
    g_str = "_".join(f"{g:g}" for g in gammas)
    sf = RESULTS / f"G{G}_tau2_smoothfasth{h:g}_het{g_str}_{label}_summary.json"
    if not sf.exists(): return None
    with open(sf) as f:
        return json.load(f)


def find_seed_path(gammas, h, label):
    g_str = "_".join(f"{g:g}" for g in gammas)
    return RESULTS / f"G{G}_tau2_smoothfasth{h:g}_het{g_str}_{label}_prices.npz"


def best_resid(s):
    return float(s.get("residual_inf", float("inf"))) if s else float("inf")


def attempt(label, gammas, h, seed, attempt_idx):
    # Inner tol matches ACCEPT_TOL=1e-13 -- no point converging below.
    if attempt_idx == 0:
        kw = dict(max_iter=300, tol=ACCEPT_TOL, damping=0.3, anderson=5, anderson_beta=0.7)
    elif attempt_idx == 1:
        kw = dict(max_iter=1000, tol=ACCEPT_TOL, damping=0.2, anderson=5, anderson_beta=0.7)
    else:
        kw = dict(max_iter=2000, tol=ACCEPT_TOL, damping=0.1, anderson=8, anderson_beta=0.5)
    return call(f"{label}_a{attempt_idx}", gammas, h, seed, **kw)


def run_ladder(gammas, initial_seed, tag):
    if not initial_seed.exists():
        print(f"FATAL: {initial_seed.name} missing"); return []
    cur_seed = initial_seed
    rows = []
    print(f"\n=== Ladder for γ={gammas} ===")
    for h in H_LADDER:
        label = f"hladderG20_{tag}_h{h:g}"
        accepted = None
        elapsed_total = 0.0
        for k in range(3):
            t0 = time.time()
            s = attempt(label, gammas, h, cur_seed, k)
            elapsed = time.time() - t0
            elapsed_total += elapsed
            if s is None: continue
            if best_resid(s) <= ACCEPT_TOL:
                accepted = s
                cur_seed = find_seed_path(gammas, h, f"{label}_a{k}")
                break
        if accepted is None:
            r = best_resid(s) if s else float("inf")
            print(f"  HALT h={h}: best resid={r:.2e}")
            break
        print(f"  h={h:>8g}  iters={accepted['iterations']:>4}  "
              f"resid={accepted['residual_inf']:.3e}  "
              f"1-R²={accepted['revelation_deficit']:.6e}  "
              f"slope={accepted['slope']:.4f}  ({elapsed_total:.1f}s)", flush=True)
        rows.append({"h": h, "gammas": list(gammas), **accepted})
    return rows


def main():
    print(f"Initial γ=0.1 seed: {SEED_G01.name}")
    print(f"Initial γ=20 seed:  {SEED_G20.name}")

    rows_01 = run_ladder((0.1, 0.1, 0.1), SEED_G01, "g0.1")
    rows_20 = run_ladder((20.0, 20.0, 20.0), SEED_G20, "g20")

    out = RESULTS / f"G{G}_h_ladder_PR.json"
    with open(out, "w") as f:
        json.dump({"g0.1": rows_01, "g20": rows_20}, f, indent=2)
    print(f"\nSaved to {out}")

    # NET PR table
    print("\n" + "=" * 80)
    print(f"{'h':>10} {'1-R² γ=0.1':>14} {'1-R² γ=20':>14} {'NET PR':>14} {'slope γ=0.1':>13} {'slope γ=20':>13}")
    print("-" * 80)
    for h in H_LADDER:
        r01 = next((r for r in rows_01 if r["h"] == h), None)
        r20 = next((r for r in rows_20 if r["h"] == h), None)
        if r01 and r20:
            net = r01["revelation_deficit"] - r20["revelation_deficit"]
            print(f"{h:>10g} {r01['revelation_deficit']:>14.6e} {r20['revelation_deficit']:>14.6e} "
                  f"{net:>+14.6e} {r01['slope']:>13.4f} {r20['slope']:>13.4f}")


if __name__ == "__main__":
    main()
