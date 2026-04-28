"""
Level-k contour solver with padded ghost ring.

Each iteration k -> k+1 is a level-k thinking step:
  - All three agents observe the commonly believed price function P^(k).
  - Each agent traces her contour through P^(k), forms her posterior, demands.
  - Market clearing yields P^(k+1)[i,j,l] for every realization.
  - Iteration is undamped (alpha = 1).

The grid is padded: an inner G x G x G block on u in [-UMAX, +UMAX] surrounded
by a ring of N_PAD ghost cells per side on u in [-UMAX_PAD, +UMAX_PAD]. The
ring is initialized with no-learning analytic prices (always defined, no
interpolation) and is updated each level along with the interior.

Boundary contour issues are handled by tracing through the *full* padded grid;
posteriors and the 1-R^2 statistic are reported only on the inner G^3.

Run: python3 level_k_padded.py
"""

import numpy as np
from scipy.optimize import brentq

# ----- model -----

def Lam(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def logit(p):
    return np.log(p / (1.0 - p))

def f_v(u, tau, v):
    """Signal density f_v(u) under state v in {0,1}. v=1 -> mean +1/2; v=0 -> -1/2."""
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - mean) ** 2)

def cara_demand(mu, p, gamma):
    return (logit(mu) - logit(p)) / gamma

def crra_demand(mu, p, gamma, W=1.0):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)

def market_clear(mus, gamma, kind="crra", W=1.0):
    """Solve sum_k x_k(mu_k, p) = 0 for p."""
    if kind == "cara":
        return Lam(np.mean(logit(np.clip(mus, 1e-6, 1 - 1e-6))))
    def Z(p):
        return sum(crra_demand(mu, p, gamma, W) for mu in mus)
    lo, hi = 1e-5, 1 - 1e-5
    Zlo, Zhi = Z(lo), Z(hi)
    if Zlo * Zhi > 0:
        # all-buy or all-sell: return the boundary closest to zero crossing
        return hi if Zlo > 0 else lo
    return brentq(Z, lo, hi, xtol=1e-12)

# ----- price grid initialization (no-learning) -----

def no_learning_price(u_triple, gamma, tau, kind):
    u1, u2, u3 = u_triple
    mus = [Lam(tau * u1), Lam(tau * u2), Lam(tau * u3)]
    return market_clear(mus, gamma, kind)

def build_price_array(u_grid, gamma, tau, kind):
    n = len(u_grid)
    P = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                P[i, j, l] = no_learning_price((u_grid[i], u_grid[j], u_grid[l]),
                                               gamma, tau, kind)
    return P

# ----- 2-pass contour through one slice -----

def contour_passes(slice_2d, u_grid, target_p, tau):
    """
    slice_2d: (n,n) prices on u_grid x u_grid.
    Trace contour {(a,b): slice(a,b) = target_p} via 2-pass linear root-find.
    Returns A_0, A_1 = sum of f_v(a) f_v(b) along the contour.
    """
    n = len(u_grid)
    A0 = 0.0
    A1 = 0.0

    # Pass A: sweep a-axis (rows) on grid, root-find b off grid in each row.
    for i in range(n):
        row = slice_2d[i, :]
        diffs = row - target_p
        for j in range(n - 1):
            d0, d1 = diffs[j], diffs[j + 1]
            if d0 * d1 < 0:  # sign change -> crossing
                t = d0 / (d0 - d1)  # linear interp parameter
                b_star = u_grid[j] + t * (u_grid[j + 1] - u_grid[j])
                A0 += f_v(u_grid[i], tau, 0) * f_v(b_star, tau, 0)
                A1 += f_v(u_grid[i], tau, 1) * f_v(b_star, tau, 1)
            elif d0 == 0:
                A0 += f_v(u_grid[i], tau, 0) * f_v(u_grid[j], tau, 0)
                A1 += f_v(u_grid[i], tau, 1) * f_v(u_grid[j], tau, 1)

    # Pass B: sweep b-axis (cols) on grid, root-find a off grid in each col.
    for j in range(n):
        col = slice_2d[:, j]
        diffs = col - target_p
        for i in range(n - 1):
            d0, d1 = diffs[i], diffs[i + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                a_star = u_grid[i] + t * (u_grid[i + 1] - u_grid[i])
                A0 += f_v(a_star, tau, 0) * f_v(u_grid[j], tau, 0)
                A1 += f_v(a_star, tau, 1) * f_v(u_grid[j], tau, 1)
            elif d0 == 0:
                A0 += f_v(u_grid[i], tau, 0) * f_v(u_grid[j], tau, 0)
                A1 += f_v(u_grid[i], tau, 1) * f_v(u_grid[j], tau, 1)

    # Average the two passes
    return 0.5 * A0, 0.5 * A1

# ----- one Phi evaluation -----

def Phi(P, u_grid, gamma, tau, kind):
    n = len(u_grid)
    P_new = np.empty_like(P)
    for i in range(n):
        for j in range(n):
            for l in range(n):
                p = P[i, j, l]
                # Agent 1 slice: P[i, :, :], own signal u_grid[i]
                A0_1, A1_1 = contour_passes(P[i, :, :], u_grid, p, tau)
                # Agent 2 slice: P[:, j, :], own signal u_grid[j]
                A0_2, A1_2 = contour_passes(P[:, j, :], u_grid, p, tau)
                # Agent 3 slice: P[:, :, l], own signal u_grid[l]
                A0_3, A1_3 = contour_passes(P[:, :, l], u_grid, p, tau)

                u1, u2, u3 = u_grid[i], u_grid[j], u_grid[l]

                def post(u_own, A0, A1):
                    num1 = f_v(u_own, tau, 1) * A1
                    num0 = f_v(u_own, tau, 0) * A0
                    if num0 + num1 == 0:
                        return Lam(tau * u_own)  # fall back to prior posterior
                    return num1 / (num0 + num1)

                mu1 = post(u1, A0_1, A1_1)
                mu2 = post(u2, A0_2, A1_2)
                mu3 = post(u3, A0_3, A1_3)
                mu1 = np.clip(mu1, 1e-6, 1 - 1e-6)
                mu2 = np.clip(mu2, 1e-6, 1 - 1e-6)
                mu3 = np.clip(mu3, 1e-6, 1 - 1e-6)

                try:
                    p_new = market_clear([mu1, mu2, mu3], gamma, kind)
                except ValueError:
                    p_new = p
                P_new[i, j, l] = np.clip(p_new, 1e-4, 1 - 1e-4)
    return P_new

# ----- 1-R^2 on the inner block -----

def compute_R2(P_inner, u_inner, tau):
    n = len(u_inner)
    rows = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                p = P_inner[i, j, l]
                if p < 1e-4 or p > 1 - 1e-4:
                    continue
                T = tau * (u_inner[i] + u_inner[j] + u_inner[l])
                w = 0.5 * (
                    f_v(u_inner[i], tau, 1) * f_v(u_inner[j], tau, 1) * f_v(u_inner[l], tau, 1)
                    + f_v(u_inner[i], tau, 0) * f_v(u_inner[j], tau, 0) * f_v(u_inner[l], tau, 0)
                )
                rows.append((logit(p), T, w))
    Y = np.array([r[0] for r in rows])
    X = np.array([r[1] for r in rows])
    W = np.array([r[2] for r in rows])
    W = W / W.sum()
    Ymean = (W * Y).sum()
    Xmean = (W * X).sum()
    cov = (W * (Y - Ymean) * (X - Xmean)).sum()
    vy = (W * (Y - Ymean) ** 2).sum()
    vx = (W * (X - Xmean) ** 2).sum()
    if vy * vx == 0:
        return 0.0
    R2 = cov ** 2 / (vy * vx)
    return 1.0 - R2

# ----- driver -----

def run(G_inner=5, N_pad=2, UMAX=2.0, UMAX_PAD=4.0,
        gamma=0.5, tau=2.0, kind="crra", levels=15, verbose=True):
    """Level-k Picard with padded ghost ring."""
    G_full = G_inner + 2 * N_pad
    u_inner = np.linspace(-UMAX, UMAX, G_inner)
    # Place pad points outside on each side, evenly to UMAX_PAD
    pad_left = np.linspace(-UMAX_PAD, -UMAX, N_pad + 1)[:-1]
    pad_right = np.linspace(UMAX, UMAX_PAD, N_pad + 1)[1:]
    u_full = np.concatenate([pad_left, u_inner, pad_right])

    P = build_price_array(u_full, gamma, tau, kind)

    # Indices of the inner block within the full grid
    inner_idx = slice(N_pad, N_pad + G_inner)

    history = []
    P_no_learn_inner = P[inner_idx, inner_idx, inner_idx].copy()
    R2_NL = compute_R2(P_no_learn_inner, u_inner, tau)
    if verbose:
        print(f"\n{'='*64}")
        print(f"  kind={kind}, gamma={gamma}, tau={tau}")
        print(f"  G_inner={G_inner}, N_pad={N_pad}, UMAX={UMAX}, UMAX_PAD={UMAX_PAD}")
        print(f"{'='*64}")
        print(f"Level 0 (no-learning):  1-R^2 = {R2_NL:.6f}")

    for k in range(levels):
        P_next = Phi(P, u_full, gamma, tau, kind)
        # Symmetrize over 6 permutations of (i,j,l)
        P_next_sym = np.zeros_like(P_next)
        for ax in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            P_next_sym += np.transpose(P_next, ax)
        P_next_sym /= 6.0

        delta = np.max(np.abs(P_next_sym - P))
        # residual ||P - Phi(P)||_inf at current iterate (pre-update)
        F_inf = np.max(np.abs(Phi(P, u_full, gamma, tau, kind) - P))
        P = P_next_sym

        P_inner = P[inner_idx, inner_idx, inner_idx]
        R2 = compute_R2(P_inner, u_inner, tau)
        history.append((k + 1, delta, F_inf, R2))
        if verbose:
            print(f"Level {k+1:2d}: ||delta||={delta:.4e}  ||F||={F_inf:.4e}  1-R^2={R2:.6f}")

        if delta < 1e-7:
            if verbose:
                print(f"  converged at level {k+1}")
            break

    return P, u_full, history


if __name__ == "__main__":
    print("\n### CARA at G_inner=5, tau=2, levels=15 ###")
    P_cara, u_cara, h_cara = run(G_inner=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0,
                                  gamma=1.0, tau=2.0, kind="cara", levels=15)
    print("\n### CRRA gamma=0.5 at G_inner=5, tau=2, levels=15 ###")
    P_crra, u_crra, h_crra = run(G_inner=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0,
                                  gamma=0.5, tau=2.0, kind="crra", levels=15)
    print("\n### Baseline: same G=5 WITHOUT padding (N_pad=0) ###")
    print("    CARA:")
    P_cara_np, _, h_cara_np = run(G_inner=5, N_pad=0, UMAX=2.0, UMAX_PAD=2.0,
                                   gamma=1.0, tau=2.0, kind="cara", levels=10)
    print("    CRRA gamma=0.5:")
    P_crra_np, _, h_crra_np = run(G_inner=5, N_pad=0, UMAX=2.0, UMAX_PAD=2.0,
                                   gamma=0.5, tau=2.0, kind="crra", levels=10)

    # Report on the standard reference realization (1,-1,1) inside the grid.
    n_full = len(u_cara)
    n_inner = 5
    pad = 2
    # find indices of u=+1, -1, +1 in inner grid (which spans -2..+2 with G=5)
    u_inner = u_cara[pad:pad + n_inner]
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    print("\n### Reference realization (u1,u2,u3) = (+1,-1,+1) ###")
    print(f"  CARA price : {P_cara[pad+i_p1, pad+i_m1, pad+i_p1]:.4f}")
    print(f"  CRRA price : {P_crra[pad+i_p1, pad+i_m1, pad+i_p1]:.4f}")
