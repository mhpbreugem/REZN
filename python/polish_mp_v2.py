#!/usr/bin/env python3
"""
mpmath polisher v2 — Newton with float64 Jacobian preconditioner.

Each polishing pass:
  1. Build the dense float64 Jacobian J_f64 via finite differences (400
     f64 phi evals, ~1 sec).
  2. LU-factor J_f64 once (numpy, ~milliseconds).
  3. Newton iteration in mpmath:
        delta = J_f64^-1 * (-F_mp)        (linear solve in f64, F coerced)
        mu   <- mu + delta                (in mpmath)
     Each step reduces ||F||_mp by ~1e-16 (the f64 epsilon), so a residual
     of 1e-13 reaches 1e-50 in 2-3 Newton iterations.

The f64 Jacobian is approximate (J_f64 != J_mp), but the discrepancy is
O(eps_f64) and only slows convergence to linear at rate ~eps_f64 per step.
That is fast enough to reach mp100 / 1e-50 in a handful of steps.
"""

from __future__ import annotations

import json
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy.linalg import lu_factor, lu_solve

sys.path.insert(0, os.path.dirname(__file__))

from solve_posterior_v3 import phi_step as phi_step_f64
from polish_mpmath import (
    phi_step_mp,
    signal_density_mp,
    F_func_flat,
    to_mp_str,
    mpf,
)


def build_jacobian_f64(mu_flat_f64, u_grid_f64, p_grids_f64, tau_f64,
                       gamma_f64, eps=1e-7, active_indices=None):
    """Dense Jacobian of F = Phi(mu) - mu via forward finite differences in
    float64.  If `active_indices` is provided, only return the submatrix
    J[active, active] — the rest of the system is assumed frozen."""
    G = u_grid_f64.size
    n = G * G
    F0 = (phi_step_f64(mu_flat_f64.reshape(G, G), u_grid_f64, p_grids_f64,
                       tau_f64, gamma_f64).reshape(-1) - mu_flat_f64)
    if active_indices is None:
        active_indices = list(range(n))
    n_act = len(active_indices)
    J = np.empty((n_act, n_act))
    e = np.zeros(n)
    for col_idx, k in enumerate(active_indices):
        e[k] = eps
        mu_pert = np.clip(mu_flat_f64 + e, 1e-15, 1.0 - 1e-15)
        F_pert = (phi_step_f64(mu_pert.reshape(G, G), u_grid_f64,
                               p_grids_f64, tau_f64, gamma_f64).reshape(-1)
                  - mu_pert)
        diff = (F_pert - F0) / eps
        J[:, col_idx] = diff[active_indices]
        e[k] = 0.0
    return J


def polish_one(checkpoint_path, dps=100, target=mp.mpf("1e-50"),
               max_iter=8, weight_floor=1e-6, fd_eps=1e-8,
               n_refine=6):
    """Polish one checkpoint via Newton with f64-Jacobian preconditioner."""
    mp.mp.dps = dps
    target_mp = target if isinstance(target, mp.mpf) else mp.mpf(str(target))

    with open(checkpoint_path) as f:
        d = json.load(f)
    G = d["G"]
    UMAX = d["UMAX"]
    tau_f64 = float(d["tau"])
    gamma_f64 = float(d["gamma"])
    tau = mp.mpf(repr(tau_f64))
    gamma = mp.mpf(repr(gamma_f64))

    print(f"=== {os.path.basename(checkpoint_path)} ===  (dps={dps})")
    print(f"    G={G}, tau={tau_f64}, gamma={gamma_f64}")

    u_grid_f64 = np.linspace(-UMAX, UMAX, G)
    u_grid_mp = [mpf(repr(float(x))) for x in u_grid_f64]
    p_grids_f64 = np.array([[float(x) for x in row] for row in d["p_grid"]])
    p_grids = [[mpf(s) for s in row] for row in d["p_grid"]]

    mu_f64 = np.array([[float(x) for x in row] for row in d["mu_strings"]])
    mu_flat = []
    for i in range(G):
        for j in range(G):
            mu_flat.append(mpf(d["mu_strings"][i][j]))

    # Density helpers in mp
    f0_grid = [signal_density_mp(u, 0, tau) for u in u_grid_mp]
    f1_grid = [signal_density_mp(u, 1, tau) for u in u_grid_mp]

    # Active mask
    u_arr = u_grid_f64
    f0 = np.sqrt(tau_f64 / (2 * np.pi)) * np.exp(-tau_f64 / 2 * (u_arr + 0.5) ** 2)
    f1 = np.sqrt(tau_f64 / (2 * np.pi)) * np.exp(-tau_f64 / 2 * (u_arr - 0.5) ** 2)
    row_w = 0.5 * (f0 + f1)
    active_rows = row_w > weight_floor
    active_mask = np.zeros((G, G), dtype=bool)
    active_mask[active_rows, :] = True
    print(f"    active rows: {np.where(active_rows)[0].tolist()}")

    # Active sub-system: only solve Newton on cells with non-trivial weight
    flat_active_arr = np.array(
        [bool(active_mask[k // G, k % G]) for k in range(G * G)])
    active_indices = [k for k in range(G * G) if flat_active_arr[k]]
    n_active = len(active_indices)
    print(f"    {n_active} active cells (of {G*G})")

    # Build f64 Jacobian (sub-matrix on active cells) and LU-factor it
    print("    building f64 Jacobian...", flush=True)
    t0 = time.time()
    J_f64 = build_jacobian_f64(mu_f64.reshape(-1), u_grid_f64, p_grids_f64,
                                 tau_f64, gamma_f64, eps=fd_eps,
                                 active_indices=active_indices)
    lu_piv = lu_factor(J_f64)
    print(f"    Jacobian built+factored in {time.time()-t0:.1f}s")

    history = []
    # Save the original (f64) iterate so we can freeze inactive (corner)
    # cells back to it — they typically refuse to converge in this
    # algorithm and Newton with the f64 Jacobian destabilises them.
    mu_flat_init = list(mu_flat)
    flat_active = [bool(active_mask[k // G, k % G]) for k in range(G * G)]

    def stats(F_flat_mp):
        F_max = max(abs(x) for x in F_flat_mp)
        sorted_abs = sorted(abs(x) for x in F_flat_mp)
        F_med = sorted_abs[len(sorted_abs) // 2]
        F_act = mpf(0)
        for i in range(G):
            for j in range(G):
                if active_mask[i, j]:
                    a = abs(F_flat_mp[i * G + j])
                    if a > F_act:
                        F_act = a
        return F_max, F_med, F_act

    eps_clip_safe = mpf("10") ** (-(dps - 5))
    t0 = time.time()
    F_act_prev = None
    for it in range(1, max_iter + 1):
        F_flat = F_func_flat(mu_flat, u_grid_mp, p_grids, tau, gamma,
                              f0_grid, f1_grid, G)
        F_max, F_med, F_act = stats(F_flat)
        elapsed = time.time() - t0
        print(f"    iter {it:2d}  F_max={mp.nstr(F_max, 6)}  "
              f"F_act={mp.nstr(F_act, 6)}  F_med={mp.nstr(F_med, 6)}  "
              f"({elapsed:.1f}s)", flush=True)
        history.append({
            "iter": it,
            "F_max": to_mp_str(F_max, dps),
            "F_med": to_mp_str(F_med, dps),
            "F_active": to_mp_str(F_act, dps),
        })

        if F_act < target_mp:
            print(f"    CONVERGED")
            break

        if F_act_prev is not None and F_act >= F_act_prev * mpf("0.95"):
            print(f"    STALLED")
            break
        F_act_prev = F_act

        # Newton step on active cells only.  rhs_active = -F[active].
        rhs_f64 = np.array([float(-F_flat[k]) for k in active_indices])
        delta_active_f64 = lu_solve(lu_piv, rhs_f64)

        delta = [mpf(0)] * (G * G)
        for col_idx, k in enumerate(active_indices):
            delta[k] = mpf(repr(float(delta_active_f64[col_idx])))

        # Iterative refinement of the Newton solve on the active subsystem.
        for refine in range(n_refine):
            mu_pert = [mu_flat[k] + delta[k] for k in range(G * G)]
            for k in range(G * G):
                if mu_pert[k] < eps_clip_safe:
                    mu_pert[k] = eps_clip_safe
                elif mu_pert[k] > 1 - eps_clip_safe:
                    mu_pert[k] = 1 - eps_clip_safe
            F_pert = F_func_flat(mu_pert, u_grid_mp, p_grids, tau, gamma,
                                  f0_grid, f1_grid, G)
            r_active_max = max(abs(F_pert[k]) for k in active_indices)
            if r_active_max < target_mp * mpf("0.01"):
                break
            r_f64 = np.array([float(-F_pert[k]) for k in active_indices])
            corr_active_f64 = lu_solve(lu_piv, r_f64)
            for col_idx, k in enumerate(active_indices):
                delta[k] = delta[k] + mpf(repr(float(corr_active_f64[col_idx])))

        # Apply only to active cells; freeze inactive (corner) cells at
        # their f64-converged values.
        mu_flat = [mu_flat[k] + delta[k] if flat_active[k] else mu_flat_init[k]
                    for k in range(G * G)]
        for k in range(G * G):
            x = mu_flat[k]
            if hasattr(x, "imag") and x.imag != 0:
                x = x.real
            if x < eps_clip_safe:
                x = eps_clip_safe
            elif x > 1 - eps_clip_safe:
                x = 1 - eps_clip_safe
            mu_flat[k] = x

        # Print refinement diagnostic
        delta_norm = max(abs(delta[k]) for k in range(G * G) if flat_active[k])
        print(f"        ||delta||active={mp.nstr(delta_norm, 4)}",
              flush=True)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. "
          f"||F||inf={mp.nstr(F_max, 4)}, "
          f"||F||active={mp.nstr(F_act, 4)}, "
          f"||F||med={mp.nstr(F_med, 4)}\n", flush=True)

    mu = [[mu_flat[i * G + j] for j in range(G)] for i in range(G)]
    return {
        "G": G,
        "UMAX": UMAX,
        "tau": tau_f64,
        "gamma": gamma_f64,
        "trim": d.get("trim", 0.0),
        "dps": dps,
        "F_max": to_mp_str(F_max, dps),
        "F_med": to_mp_str(F_med, dps),
        "F_active": to_mp_str(F_act, dps),
        "u_grid": [to_mp_str(u, dps) for u in u_grid_mp],
        "p_grid": [[to_mp_str(x, dps) for x in row] for row in p_grids],
        "mu_strings": [[to_mp_str(x, dps) for x in row] for row in mu],
        "history": history,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=100)
    ap.add_argument("--tol", type=str, default="1e-50")
    ap.add_argument("--maxiter", type=int, default=8)
    ap.add_argument("--only", type=float, default=None)
    args = ap.parse_args()

    indir = "results/full_ree"
    files = sorted([f for f in os.listdir(indir)
                    if f.startswith("task3_g100_t") and f.endswith("_mp50.json")])
    if args.only is not None:
        token = f"t{int(round(args.only * 100)):04d}"
        files = [f for f in files if token in f]

    target = mp.mpf(args.tol)
    print(f"Polishing {len(files)} checkpoints at dps={args.dps}, "
          f"target={args.tol}")

    for fname in files:
        path = os.path.join(indir, fname)
        out_name = fname.replace("_mp50.json", f"_mp{args.dps}.json")
        out_path = os.path.join(indir, out_name)
        if os.path.exists(out_path):
            print(f"=== {fname} === skip (already polished -> {out_name})")
            continue
        try:
            payload = polish_one(path, dps=args.dps, target=target,
                                  max_iter=args.maxiter)
        except Exception as exc:
            print(f"    FAILED on {fname}: {exc}")
            continue
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"    wrote {out_name}")


if __name__ == "__main__":
    main()
