"""
Classic (undamped) Picard on the level-k Phi map.

  P_inner^(k+1) = Phi(P_ext^(k))

Each outer iteration is exactly one more level of higher-order belief:
  - Level 0: agents use prior posteriors mu_k = Lambda(tau u_k). Market clears.
  - Level k -> k+1: agents *commonly believe* the price function is P_ext^(k).
    Each agent traces her contour through P_ext^(k), backs out posteriors,
    and the market clears at p^(k+1)[i,j,l] for each realization.

The 2-SD ghost ring is held at analytic no-learning prices throughout (it's
the agents' commonly held boundary belief about prices at extreme signals).
The inner block is updated each iter by undamped Phi.

CRRA gamma=0.5, tau=2. G_inner=5. Maxiter=40, tol=1e-7.
"""

import numpy as np
from level1_step import (
    TAU, GAMMA, G_INNER, u_inner, u_ext, inner_idx_in_ext, PAD,
    build_P_ext, phi_step, deficit, Lam, logit, f_v,
)


def run_undamped_picard(maxiter=40, tol=1e-7, verbose=True):
    P_ext = build_P_ext()
    P_inner = P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)].copy()
    R2_0, slope_0, intc_0, _ = deficit(P_inner)
    if verbose:
        print("=" * 78)
        print(f"UNDAMPED (CLASSIC) PICARD  alpha = 1  maxiter={maxiter}  tol={tol:.0e}")
        print(f"  CRRA gamma={GAMMA}, tau={TAU}, G_inner={G_INNER}")
        print(f"  Each outer iter = one more level of higher-order belief")
        print(f"  Ring = analytic no-learning (held fixed)")
        print("=" * 78)
        print(f"Level 0  (no-learning)   slope={slope_0:.4f}   1-R^2={R2_0:.6f}")

    history = [(0, 0.0, R2_0, slope_0)]
    P_inner_prev = P_inner.copy()

    for k in range(1, maxiter + 1):
        P_phi, _ = phi_step(P_ext)               # one Phi call -> level k from level k-1
        delta = float(np.max(np.abs(P_phi - P_inner_prev)))
        # Refresh inner block; ring untouched
        P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)] = P_phi
        R2, slope, intc, _ = deficit(P_phi)
        history.append((k, delta, R2, slope))
        if verbose:
            i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
            i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
            p_can = P_phi[i_p1, i_m1, i_p1]
            print(f"Level {k:2d}  delta={delta:.4e}   slope={slope:.4f}   "
                  f"1-R^2={R2:.6f}   p(canonical)={p_can:.4f}")
        if delta < tol:
            print(f"  converged at level {k}")
            P_inner_prev = P_phi
            break
        P_inner_prev = P_phi

    return P_inner_prev, P_ext, history


if __name__ == "__main__":
    P_final, P_ext_final, hist = run_undamped_picard(maxiter=40, tol=1e-7)

    print()
    print("=" * 78)
    print("FINAL STATE")
    print("=" * 78)
    R2, slope, intc, n = deficit(P_final)
    print(f"  weighted regression: logit(p) = {intc:+.4f} + {slope:+.4f} * T*")
    print(f"  1 - R^2 = {R2:.6f}   ({n} interior triples)")
    print(f"  Baseline (no-learning): 1-R^2 = {hist[0][2]:.6f}")
    print()

    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    p_can = P_final[i_p1, i_m1, i_p1]
    Tstar = TAU * (u_inner[i_p1] + u_inner[i_m1] + u_inner[i_p1])
    print(f"  Canonical realization (u1,u2,u3) = (+1,-1,+1):")
    print(f"    p = {p_can:.6f}")
    print(f"    Lambda(T*)  = {Lam(Tstar):.6f}    (FR target)")
    print(f"    Lambda(T*/3) = {Lam(Tstar / 3):.6f}  (no-learning)")
    print(f"    logit(p) / T* = {logit(p_can) / Tstar:.4f}    (slope per single point)")
    print()

    # Slice
    print(f"--- Slice P_final[u_1 = +0, :, :] ---")
    print(f"            u_3 =  {'   '.join(f'{u:+.0f}' for u in u_inner)}")
    for j in range(G_INNER):
        row = "  ".join(f"{P_final[2, j, l]:.4f}" for l in range(G_INNER))
        print(f"   u_2 = {u_inner[j]:+.0f}  {row}")
    print()

    # Full trajectory
    print("--- Full trajectory ---")
    print("  level   delta        slope    1-R^2")
    for k, delta, R2k, slope_k in hist:
        print(f"  {k:5d}   {delta:.4e}   {slope_k:.4f}   {R2k:.6f}")
