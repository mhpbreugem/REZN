"""
Newton-Krylov solver on F(P_inner) = P_inner - Sym(Phi(extrapolate(P_inner))) = 0.

Settings:
  - G_in=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0
  - Endogenous-extrapolation ghost ring (rebuilt each Phi call from inner arg).
  - f_tol = 1e-12, maxiter = 60, scipy.optimize.newton_krylov.
  - Per-Newton-iteration line: "ITER <n> 1-R^2=... ||F||=... t=...s"

Run: python3 newton_solve.py
"""

import sys
import time
import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence

from level_k_extrap import (
    Lam, logit, f_v, market_clear, no_learning_price,
    extrapolate_padded, build_initial_padded,
    contour_passes, Phi_inner, compute_R2, symmetrize,
)

# --------------------------------------------------------------------------
# Configure problem
# --------------------------------------------------------------------------

G_in = 5
N_pad = 2
UMAX = 2.0
UMAX_PAD = 3.0
TAU = 2.0

u_inner = np.linspace(-UMAX, UMAX, G_in)
pad_left = np.linspace(-UMAX_PAD, -UMAX, N_pad + 1)[:-1]
pad_right = np.linspace(UMAX, UMAX_PAD, N_pad + 1)[1:]
u_full = np.concatenate([pad_left, u_inner, pad_right])
PAD = N_pad


def make_residual(gamma, tau, kind):
    """F(x_flat) = x - Sym(Phi(extrapolate(x))).  x is the inner G_in^3 array, flat."""
    def F(x_flat):
        P_inner = x_flat.reshape((G_in, G_in, G_in))
        P_full = extrapolate_padded(P_inner, G_in, N_pad)
        P_new = Phi_inner(P_full, u_full, G_in, N_pad, gamma, tau, kind)
        P_new = symmetrize(P_new)
        return (P_inner - P_new).flatten()
    return F


def initial_inner(gamma, tau, kind):
    P_full0 = build_initial_padded(u_full, gamma, tau, kind)
    return P_full0[PAD:PAD + G_in, PAD:PAD + G_in, PAD:PAD + G_in].flatten()


# --------------------------------------------------------------------------
# Driver: Newton-Krylov with per-iteration callback
# --------------------------------------------------------------------------

def newton_run(gamma, tau, kind, label, f_tol=1e-12, maxiter=200,
               inner_maxiter=12, perturb_scale=0.05, stall_atol=1e-5,
               max_perturbs=8, seed=0):
    """
    Outer loop: calls newton_krylov in chunks of inner_maxiter iterations.
    Detects stall (residual barely moves between chunks) and perturbs iterate
    with symmetrized Gaussian noise scaled to the current residual, then restarts.
    """
    print(f"\n{'=' * 64}", flush=True)
    print(f"  NEWTON-KRYLOV + PERTURB-ON-STALL  {label}  gamma={gamma}  tau={tau}", flush=True)
    print(f"  G_in={G_in}, N_pad={N_pad}, UMAX={UMAX}, UMAX_PAD={UMAX_PAD}", flush=True)
    print(f"  f_tol={f_tol:.1e}, max_total_iter={maxiter}, inner_maxiter={inner_maxiter}", flush=True)
    print(f"  perturb_scale={perturb_scale}, stall_atol={stall_atol}, max_perturbs={max_perturbs}", flush=True)
    print(f"{'=' * 64}", flush=True)

    F = make_residual(gamma, tau, kind)
    x = initial_inner(gamma, tau, kind)
    rng = np.random.default_rng(seed)

    R2_NL = compute_R2(x.reshape((G_in, G_in, G_in)), u_inner, tau)
    F0 = F(x)
    print(f"ITER   0  1-R^2={R2_NL:.8f}  ||F||={np.max(np.abs(F0)):.4e}  t=0.0s  (no-learning seed)",
          flush=True)

    counter = {"n": 0, "t0": time.time()}

    def cb(x_cur, f_cur):
        counter["n"] += 1
        P_inner = x_cur.reshape((G_in, G_in, G_in))
        R2 = compute_R2(P_inner, u_inner, tau)
        Finf = float(np.max(np.abs(f_cur)))
        F2 = float(np.linalg.norm(f_cur))
        elapsed = time.time() - counter["t0"]
        print(f"ITER {counter['n']:3d}  1-R^2={R2:.8f}  ||F||inf={Finf:.4e}  ||F||2={F2:.4e}  t={elapsed:.1f}s",
              flush=True)

    last_resid = float(np.max(np.abs(F0)))
    n_perturbs = 0
    status = "running"

    while counter["n"] < maxiter and n_perturbs <= max_perturbs:
        try:
            x = newton_krylov(F, x, f_tol=f_tol,
                              maxiter=min(inner_maxiter, maxiter - counter["n"]),
                              method="lgmres", verbose=False, callback=cb,
                              line_search="armijo")
            status = "converged"
            break
        except NoConvergence as e:
            x = np.asarray(e.args[0])
            cur_resid = float(np.max(np.abs(F(x))))
            rel_change = abs(cur_resid - last_resid) / max(cur_resid, 1e-30)
            stalled = rel_change < stall_atol
            elapsed = time.time() - counter["t0"]
            if stalled and counter["n"] < maxiter:
                # Symmetric Gaussian perturbation, scaled to current residual
                noise = rng.standard_normal((G_in, G_in, G_in)) * perturb_scale * cur_resid
                P_inner = x.reshape((G_in, G_in, G_in))
                P_perturbed = symmetrize(P_inner + noise)
                P_perturbed = np.clip(P_perturbed, 1e-4, 1 - 1e-4)
                x = P_perturbed.flatten()
                n_perturbs += 1
                new_resid = float(np.max(np.abs(F(x))))
                print(f"  >>> STALL at ||F||={cur_resid:.4e} (rel change {rel_change:.2e}); "
                      f"perturb #{n_perturbs} scale={perturb_scale*cur_resid:.4e} "
                      f"-> ||F||={new_resid:.4e}  t={elapsed:.1f}s", flush=True)
                last_resid = new_resid
            else:
                last_resid = cur_resid
        except Exception as e:
            print(f"ITER FAIL: {type(e).__name__}: {e}", flush=True)
            status = f"error: {e}"
            break

    if status == "running":
        status = f"no convergence (max_total_iter={maxiter}, perturbs={n_perturbs})"

    sol = x

    P_inner = sol.reshape((G_in, G_in, G_in))
    R2_final = compute_R2(P_inner, u_inner, tau)
    F_final = F(sol)
    print(f"\nDONE  {label}: status={status}", flush=True)
    print(f"  final 1-R^2 = {R2_final:.8f}", flush=True)
    print(f"  final ||F||inf = {np.max(np.abs(F_final)):.4e}", flush=True)
    print(f"  final ||F||2   = {np.linalg.norm(F_final):.4e}", flush=True)

    # Reference realization
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    print(f"  price at (+1,-1,+1) = {P_inner[i_p1, i_m1, i_p1]:.6f}", flush=True)

    return {
        "label": label, "gamma": gamma, "tau": tau, "kind": kind,
        "status": status, "P_inner": P_inner, "R2": R2_final,
        "F_inf": float(np.max(np.abs(F_final))), "F_2": float(np.linalg.norm(F_final)),
        "iter": counter["n"],
    }


if __name__ == "__main__":
    res_cara = newton_run(gamma=1.0, tau=TAU, kind="cara", label="CARA",
                          f_tol=1e-12, maxiter=200, inner_maxiter=12,
                          perturb_scale=0.05, stall_atol=1e-5, max_perturbs=8)
    res_crra = newton_run(gamma=0.5, tau=TAU, kind="crra", label="CRRA(0.5)",
                          f_tol=1e-12, maxiter=200, inner_maxiter=12,
                          perturb_scale=0.05, stall_atol=1e-5, max_perturbs=8)

    print("\n" + "=" * 64, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 64, flush=True)
    if res_cara and res_crra:
        net = res_crra["R2"] - res_cara["R2"]
        print(f"  CARA      : 1-R^2={res_cara['R2']:.8f}   "
              f"||F||={res_cara['F_inf']:.2e}  iters={res_cara['iter']}  ({res_cara['status']})",
              flush=True)
        print(f"  CRRA(0.5) : 1-R^2={res_crra['R2']:.8f}   "
              f"||F||={res_crra['F_inf']:.2e}  iters={res_crra['iter']}  ({res_crra['status']})",
              flush=True)
        print(f"  NET (CRRA - CARA) : {net:+.8f}", flush=True)
