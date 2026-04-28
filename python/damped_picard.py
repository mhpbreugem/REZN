"""
Damped Picard on the level-k Phi map, starting from no-learning prices.

  P_inner^(k+1) = (1 - alpha) * P_inner^(k) + alpha * Phi(P_ext^(k))

The ghost ring (2-SD buffer) is held fixed at the analytic no-learning prices
throughout the iteration -- it serves as a constant boundary condition.
Inner block (5x5x5) is updated each iter.

CRRA gamma=0.5, tau=2. G_inner=5. alpha=0.3. f_tol on max-delta.
"""

import numpy as np
from level1_step import (
    TAU, GAMMA, G_INNER, u_inner, u_ext, inner_idx_in_ext, PAD,
    build_P_ext, phi_step, deficit, Lam, logit, f_v,
)


def run_damped_picard(alpha=0.3, maxiter=80, tol=1e-7, verbose=True):
    P_ext = build_P_ext()                 # analytic no-learning everywhere
    P_inner = P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)].copy()
    R2_0, slope_0, intc_0, _ = deficit(P_inner)
    if verbose:
        print("=" * 78)
        print(f"DAMPED PICARD  alpha={alpha}  maxiter={maxiter}  tol={tol:.0e}")
        print(f"  CRRA gamma={GAMMA}, tau={TAU}, G_inner={G_INNER}")
        print(f"  Ring = analytic no-learning (held fixed)")
        print("=" * 78)
        print(f"Iter  0  (no-learning)   slope={slope_0:.4f}   1-R^2={R2_0:.6f}")

    history = [(0, 0.0, 0.0, R2_0, slope_0)]
    P_inner_prev = P_inner.copy()

    for k in range(1, maxiter + 1):
        # Phi step using current extended P
        P_phi, _ = phi_step(P_ext)
        # Damped update on the inner block
        P_inner_new = (1.0 - alpha) * P_inner_prev + alpha * P_phi

        delta = float(np.max(np.abs(P_inner_new - P_inner_prev)))
        # Refresh inner block of P_ext (ring stays at analytic no-learning)
        P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)] = P_inner_new
        # Also compute the un-damped step magnitude (residual proxy)
        F_inf = float(np.max(np.abs(P_phi - P_inner_prev)))

        R2, slope, intc, _ = deficit(P_inner_new)
        history.append((k, delta, F_inf, R2, slope))

        if verbose:
            print(f"Iter {k:2d}   delta={delta:.4e}   ||F||={F_inf:.4e}   "
                  f"slope={slope:.4f}   1-R^2={R2:.6f}")

        if delta < tol:
            if verbose:
                print(f"  converged: delta < {tol:.0e}")
            P_inner_prev = P_inner_new
            break
        P_inner_prev = P_inner_new

    return P_inner_prev, P_ext, history


def report_canonical(P_inner, P_ext, label="final"):
    print()
    print(f"--- Canonical realization (u1,u2,u3) = (+1,-1,+1) at {label} ---")
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    p = P_inner[i_p1, i_m1, i_p1]
    Tstar = TAU * (u_inner[i_p1] + u_inner[i_m1] + u_inner[i_p1])
    print(f"  T* = {Tstar:.4f}")
    print(f"  p = {p:.6f}")
    print(f"  Lambda(T*) = {Lam(Tstar):.6f}    (FR target)")
    print(f"  Lambda(T*/3) = {Lam(Tstar / 3):.6f}  (no-learning level)")
    print(f"  logit(p) / T* = {logit(p) / Tstar:.4f}")


def report_slice(P_inner, label="final", own_idx=2):
    print(f"--- Slice P[u_1 = {u_inner[own_idx]:+.0f}, :, :] at {label} ---")
    print(f"            u_3 =  {'   '.join(f'{u:+.0f}' for u in u_inner)}")
    for j in range(G_INNER):
        row = "  ".join(f"{P_inner[own_idx, j, l]:.4f}" for l in range(G_INNER))
        print(f"   u_2 = {u_inner[j]:+.0f}  {row}")


if __name__ == "__main__":
    P_final, P_ext_final, hist = run_damped_picard(alpha=0.3, maxiter=80, tol=1e-7)

    print()
    print("=" * 78)
    print("FINAL STATE")
    print("=" * 78)
    R2, slope, intc, n = deficit(P_final)
    print(f"  weighted regression: logit(p) = {intc:+.4f} + {slope:+.4f} * T*")
    print(f"  1 - R^2 = {R2:.6f}   ({n} interior triples)")

    # No-learning baseline for comparison
    R2_0 = hist[0][3]
    print(f"  Baseline (no-learning): 1-R^2 = {R2_0:.6f}")
    print(f"  Net learning effect:    Delta(1-R^2) = {R2 - R2_0:+.6f}")

    report_canonical(P_final, P_ext_final, label="final")
    print()
    report_slice(P_final, label="final", own_idx=2)
    print()

    # Trajectory summary every few iters
    print("--- Trajectory (every iter) ---")
    print("  iter   delta        ||F||        slope    1-R^2")
    for k, delta, Finf, R2k, slope_k in hist:
        print(f"  {k:4d}   {delta:.4e}   {Finf:.4e}   {slope_k:.4f}   {R2k:.6f}")
