#!/usr/bin/env python3
"""
Orchestrate all τ-sweep runs for Fig 4A.
Runs γ=4.0 first (fastest, closest to CARA), then γ=1.0.
Each run warm-starts from the nearest converged τ.
Saves a JSON summary after every run.

Run from /home/user/REZN/:
    nohup python python/run_sweeps.py > /tmp/sweeps.log 2>&1 &
"""

import subprocess, sys, os, json, time

SOLVER = 'python python/solver_v3_mp.py'
OUT    = 'results/full_ree'
SEED   = f'{OUT}/posterior_v3_G20_umax5_notrim_mp300.json'  # γ=0.5 τ=2 mp300 seed

G        = 20
UMAX     = 5.0
TOL      = 1e-25   # mp50 target — beyond machine precision
MAX_ITER_COLD = 500    # cold: Phase1 Picard (~200) + Phase2 LM-Newton (~20 steps)
MAX_ITER_WARM = 300    # warm: already near fixed point, Phase2 Newton directly
ALPHA    = 0.3         # Picard damping (Phase1 Anderson + Phase2)
ANDERSON = 5           # Anderson history depth (Phase 1 only)

# τ values per FIGURES_TODO.md
TAU_ALL = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]

# Already done: τ=2.0 for all γ (from Fig 4B sweep — but checkpoint not saved,
# so we start fresh from no-learning).  τ=2.0 included in chain to confirm.

# Warm-start chains: for each γ, walk up and down from τ=2.0
CHAINS = {
    4.0: {  # nearest to CARA — do first, fastest convergence
        'up':   [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0],
        'down': [2.0, 1.5, 1.0, 0.8,  0.5,  0.3],
    },
    1.0: {  # do second
        'up':   [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0],
        'down': [2.0, 1.5, 1.0, 0.8,  0.5,  0.3],
    },
}

def ckpt_path(gamma, tau):
    g = int(round(gamma * 100))
    t = int(round(tau * 10))
    return f'{OUT}/task3_g{g:03d}_t{t:04d}.json'

def run_one(gamma, tau, seed, label):
    out = ckpt_path(gamma, tau)
    seed_arg = f'--seed {seed}' if seed else ''
    max_iter = MAX_ITER_WARM if seed else MAX_ITER_COLD
    cmd = (f'{SOLVER} --gamma {gamma} --tau {tau} --out {out} '
           f'{seed_arg} --G {G} --umax {UMAX} --tol {TOL} '
           f'--max_iter {max_iter} --alpha {ALPHA} --anderson {ANDERSON}')
    print(f'\n{"="*70}', flush=True)
    print(f'[{label}] γ={gamma} τ={tau}  max_iter={max_iter}', flush=True)
    print(f'  cmd: {cmd}', flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - t0
    ok = (ret.returncode == 0) and os.path.exists(out)
    status = 'OK' if ok else 'FAILED'
    print(f'  → {status}  {elapsed:.0f}s', flush=True)
    return out if ok else None

summary = {}

for gamma, chains in CHAINS.items():
    print(f'\n{"#"*70}', flush=True)
    print(f'# γ = {gamma}  ({"nearest to CARA" if gamma==4.0 else ""})', flush=True)
    summary[gamma] = {}

    # Walk up from τ=2.0
    # Use SEED (γ=0.5 τ=2 fully converged) as anchor warm-start; gives F_max~0.49
    # vs no-learning F_max~0.98, halving the distance for Phase-2 Newton.
    prev = SEED if os.path.exists(SEED) else None
    for tau in chains['up']:
        seed = prev  # warm-start from previous τ (or None → no-learning)
        label = f'γ={gamma} τ={tau} (up)'
        result = run_one(gamma, tau, seed, label)
        if result:
            # Read F_max and 1-R² from checkpoint
            try:
                with open(result) as f: d = json.load(f)
                F = d.get('F_max','?')
                summary[gamma][tau] = {'F_max': F, 'ckpt': result}
                print(f'  F_max={F}', flush=True)
            except:
                pass
            prev = result  # warm-start next τ from this
        else:
            print(f'  SKIPPING τ≥{tau} (previous run failed)', flush=True)
            break

    # Walk down from τ=2.0 (use the τ=2.0 checkpoint as seed)
    seed_t2 = ckpt_path(gamma, 2.0) if os.path.exists(ckpt_path(gamma, 2.0)) else None
    prev = seed_t2
    for tau in chains['down'][1:]:  # skip τ=2.0 (already done)
        result = run_one(gamma, tau, prev, f'γ={gamma} τ={tau} (down)')
        if result:
            try:
                with open(result) as f: d = json.load(f)
                F = d.get('F_max','?')
                summary[gamma][tau] = {'F_max': F, 'ckpt': result}
                print(f'  F_max={F}', flush=True)
            except:
                pass
            prev = result
        else:
            print(f'  SKIPPING τ≤{tau} (previous run failed)', flush=True)
            break

    # Print summary for this γ
    print(f'\n--- γ={gamma} summary ---', flush=True)
    for tau in sorted(summary[gamma]):
        info = summary[gamma][tau]
        print(f'  τ={tau:<5} F_max={info["F_max"]}', flush=True)

# Save run summary
with open(f'{OUT}/sweep_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print(f'\nAll done. Summary saved to {OUT}/sweep_summary.json', flush=True)
