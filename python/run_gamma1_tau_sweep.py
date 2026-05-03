#!/usr/bin/env python3
"""
Terminal 2: gamma=1.0 tau-sweep at G=20, UMAX=5.

Strategy (per PARALLEL_SOLVER.md):
  1. Warm-start gamma=1.0, tau=2.0 from the gamma=0.5 mp300 seed.
  2. Walk down: tau = 1.5, 1.0, 0.8, 0.5, 0.3 (each warm-started from
     the previous tau result).
  3. Walk up: tau = 3.0, 4.0, 5.0, 7.0, 10.0, 15.0.

Each tau is solved with scipy.optimize.newton_krylov on the residual
F(mu) = Phi(mu) - mu, where Phi is the Bayes update from
solve_posterior_v3.phi_step.

Bail criterion: ||F||_inf > 0.1 after maxiter steps -> skip and continue.

Outputs:
  - results/full_ree/task3_g100_tXXXX_mp50.json   (per tau)
  - results/full_ree/fig4A_g100_tau_sweep.json    (summary)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence

sys.path.insert(0, os.path.dirname(__file__))

from solve_posterior_v3 import (
    build_p_grids,
    interp_seed_to_grid,
    phi_step,
    save_checkpoint,
    weighted_1mR2,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

G = 20
UMAX = 5.0
GAMMA = 1.0
DPS = 50
TOL = 1e-12          # float64 tolerance; bail-out is 0.1 per spec
BAIL_FMAX = 0.1
MAX_NK_ITER = 50

TAUS_DOWN = [1.5, 1.0, 0.8, 0.5, 0.3]
TAUS_UP = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
TAU_SEED = 2.0      # already converged at gamma=0.5 (the mp300 seed)
ALL_TAUS = sorted([TAU_SEED] + TAUS_DOWN + TAUS_UP)

OUTDIR = "results/full_ree"
SEED_PATH = os.path.join(OUTDIR, "posterior_v3_G20_umax5_notrim_mp300.json")


def tau_str(tau: float) -> str:
    """Stable filename token for a tau value, e.g. 0.3 -> '0030', 15.0 -> '1500'."""
    return f"{int(round(tau * 100)):04d}"


def checkpoint_path(tau: float) -> str:
    return os.path.join(OUTDIR, f"task3_g100_t{tau_str(tau)}_mp50.json")


def solve_tau(mu0: np.ndarray, u_grid: np.ndarray, tau: float, gamma: float,
              maxiter: int = MAX_NK_ITER):
    """Solve at one tau. Returns (mu, p_grids, F_max, F_med, history)."""
    p_lo, p_hi, p_grids = build_p_grids(u_grid, tau, gamma, G_p=G, margin=0.0)

    # Re-interpolate mu0 onto the new p_grids if shape matches but mu0
    # came from a different tau's p-grid.
    history: list = []
    n_iter_holder = [0]
    t0 = time.time()

    def F_func(mu_flat: np.ndarray) -> np.ndarray:
        mu = mu_flat.reshape(G, G)
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        F = phi_step(mu, u_grid, p_grids, tau, gamma) - mu
        n_iter_holder[0] += 1
        Fa = np.abs(F)
        history.append({
            "call": n_iter_holder[0],
            "F_max": float(Fa.max()),
            "F_med": float(np.median(Fa)),
        })
        return F.reshape(-1)

    try:
        mu_sol_flat = newton_krylov(
            F_func,
            mu0.reshape(-1),
            f_tol=TOL,
            method="lgmres",
            maxiter=maxiter,
            verbose=False,
        )
    except NoConvergence as exc:
        # NK didn't reach the tolerance; keep the last iterate.  Some cells
        # (typically the corners with negligible signal weight) refuse to
        # converge below ~0.1, but the bulk reaches ~1e-9.
        mu_sol_flat = exc.args[0]
    except Exception as exc:
        print(f"  newton_krylov error: {exc}")
        mu_sol_flat = mu0.reshape(-1).copy()
        for _ in range(20):
            F = F_func(mu_sol_flat)
            mu_sol_flat = mu_sol_flat + 0.3 * F
            mu_sol_flat = np.clip(mu_sol_flat, 1e-15, 1.0 - 1e-15)

    elapsed = time.time() - t0
    mu_sol = np.clip(mu_sol_flat.reshape(G, G), 1e-15, 1.0 - 1e-15)
    F_final = phi_step(mu_sol, u_grid, p_grids, tau, gamma) - mu_sol
    F_max = float(np.abs(F_final).max())
    F_med = float(np.median(np.abs(F_final)))

    # Compress history (keep one entry per ~5 calls, plus first/last)
    if len(history) > 20:
        keep_idx = list(range(0, len(history), max(1, len(history) // 15)))
        if keep_idx[-1] != len(history) - 1:
            keep_idx.append(len(history) - 1)
        history = [history[k] for k in keep_idx]

    return mu_sol, p_grids, F_max, F_med, elapsed, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=float, default=None,
                    help="Run only this tau (debugging)")
    args = ap.parse_args()

    print(f"=== Terminal 2: gamma={GAMMA} tau-sweep at G={G}, UMAX={UMAX} ===")

    with open(SEED_PATH) as f:
        seed = json.load(f)
    print(f"Loaded seed: gamma={seed['gamma']} tau={seed['tau']} G={seed['G']}")

    u_grid = np.linspace(-UMAX, UMAX, G)
    summary_points: list[dict] = []
    saved_mu: dict[float, np.ndarray] = {}
    saved_pgrid: dict[float, np.ndarray] = {}

    # --- Step 1: tau = 2.0 (warm from gamma=0.5 seed)
    p_lo, p_hi, p_grids_tau2 = build_p_grids(u_grid, TAU_SEED, GAMMA,
                                              G_p=G, margin=0.0)
    mu0_tau2 = interp_seed_to_grid(seed, u_grid, p_grids_tau2,
                                    gamma_target=GAMMA)

    def run(tau: float, mu0: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        print(f"\n--- tau = {tau} ---")
        mu_sol, p_grids, F_max, F_med, elapsed, history = solve_tau(
            mu0, u_grid, tau, GAMMA)
        print(f"  ||F||inf = {F_max:.3e}  ||F||med = {F_med:.3e}  "
              f"({elapsed:.1f}s)")

        if F_max > BAIL_FMAX:
            print(f"  BAIL: ||F||inf > {BAIL_FMAX}")
            return mu_sol, p_grids, {"tau": tau, "bailed": True,
                                       "F_max": F_max, "F_med": F_med}

        one_mR2, slope, n_trip = weighted_1mR2(u_grid, p_grids, mu_sol,
                                                 tau, GAMMA)
        print(f"  1-R^2 = {one_mR2:.6f}  slope = {slope:.6f}  "
              f"n_triples = {n_trip}")

        save_checkpoint(
            checkpoint_path(tau),
            G=G, UMAX=UMAX, tau=tau, gamma=GAMMA, trim=0.0, dps=DPS,
            F_max=F_max, F_med=F_med,
            u_grid=u_grid, p_grids=p_grids, mu=mu_sol,
            history=history,
        )
        print(f"  Saved -> {checkpoint_path(tau)}")
        return mu_sol, p_grids, {"tau": tau, "1-R2": one_mR2,
                                   "slope": slope, "n_triples": n_trip,
                                   "F_max": F_max, "F_med": F_med}

    if args.only is None:
        # Helper: load a previously-saved checkpoint as warm-start.
        def load_existing(tau: float):
            path = checkpoint_path(tau)
            if not os.path.exists(path):
                return None, None
            with open(path) as f:
                d = json.load(f)
            mu = np.array([[float(x) for x in row] for row in d["mu_strings"]])
            pgr = np.array([[float(x) for x in row] for row in d["p_grid"]])
            return mu, pgr

        # Tau = 2.0 first
        existing_mu, existing_pgr = load_existing(TAU_SEED)
        if existing_mu is not None:
            print(f"--- tau = {TAU_SEED} (skip, checkpoint exists) ---")
            saved_mu[TAU_SEED] = existing_mu
            saved_pgrid[TAU_SEED] = existing_pgr
        else:
            mu_sol, p_grids, info = run(TAU_SEED, mu0_tau2)
            if not info.get("bailed"):
                saved_mu[TAU_SEED] = mu_sol
                saved_pgrid[TAU_SEED] = p_grids
                summary_points.append(info)

        # Walk down
        prev_tau = TAU_SEED
        for tau in TAUS_DOWN:
            existing_mu, existing_pgr = load_existing(tau)
            if existing_mu is not None:
                print(f"--- tau = {tau} (skip, checkpoint exists) ---")
                saved_mu[tau] = existing_mu
                saved_pgrid[tau] = existing_pgr
                prev_tau = tau
                continue
            mu0 = remap_warmstart(saved_mu[prev_tau], saved_pgrid[prev_tau],
                                   u_grid, tau, GAMMA)
            mu_sol, p_grids, info = run(tau, mu0)
            if not info.get("bailed"):
                saved_mu[tau] = mu_sol
                saved_pgrid[tau] = p_grids
                prev_tau = tau
                summary_points.append(info)

        # Walk up — always from the LAST SUCCESSFUL previous tau, even if a
        # higher tau bailed (so we don't propagate a corrupted iterate).
        prev_tau = TAU_SEED
        for tau in TAUS_UP:
            existing_mu, existing_pgr = load_existing(tau)
            if existing_mu is not None:
                print(f"--- tau = {tau} (skip, checkpoint exists) ---")
                saved_mu[tau] = existing_mu
                saved_pgrid[tau] = existing_pgr
                prev_tau = tau
                continue
            mu0 = remap_warmstart(saved_mu[prev_tau], saved_pgrid[prev_tau],
                                   u_grid, tau, GAMMA)
            mu_sol, p_grids, info = run(tau, mu0)
            if info.get("bailed"):
                print(f"  -> bailed; will warm next tau from successful tau={prev_tau}")
                continue
            saved_mu[tau] = mu_sol
            saved_pgrid[tau] = p_grids
            prev_tau = tau
            summary_points.append(info)
    else:
        tau = args.only
        if tau == TAU_SEED:
            mu0 = mu0_tau2
        else:
            # Warm from prev existing checkpoint if available
            mu0 = mu0_tau2  # fallback
        mu_sol, p_grids, info = run(tau, mu0)
        if not info.get("bailed"):
            summary_points.append(info)

    # Sort summary by tau
    summary_points.sort(key=lambda d: d["tau"])
    summary_path = os.path.join(OUTDIR, "fig4A_g100_tau_sweep.json")
    with open(summary_path, "w") as f:
        json.dump({"gamma": GAMMA,
                   "params": {"G": G, "UMAX": UMAX, "trim": 0.0,
                              "dps": DPS, "tol": TOL,
                              "weighting": "ex-ante 0.5*(f0^3+f1^3)"},
                   "points": summary_points}, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")
    print(f"Successful taus: {[p['tau'] for p in summary_points]}")


def remap_warmstart(prev_mu: np.ndarray, prev_p_grids: np.ndarray,
                    u_grid: np.ndarray, new_tau: float,
                    gamma: float) -> np.ndarray:
    """
    Remap a converged mu from one tau's p-grid onto the new tau's p-grid.
    Both grids share the same u_grid; only p-ranges and within-row spacings
    differ.  Linear interpolation in logit(p) within each row.
    """
    new_p_lo, new_p_hi, new_p_grids = build_p_grids(u_grid, new_tau, gamma,
                                                      G_p=prev_mu.shape[1],
                                                      margin=0.0)
    G_u, G_p = prev_mu.shape
    out = np.empty_like(prev_mu)
    for i in range(G_u):
        old_p = prev_p_grids[i, :]
        new_p = new_p_grids[i, :]
        old_lp = np.log(old_p) - np.log1p(-old_p)
        new_lp = np.log(new_p) - np.log1p(-new_p)
        out[i, :] = np.interp(new_lp, old_lp, prev_mu[i, :])
    return np.clip(out, 1e-15, 1.0 - 1e-15)


if __name__ == "__main__":
    main()
