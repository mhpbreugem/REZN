"""
Level-k contour solver with endogenous-extrapolation ghost ring.

Each level k -> k+1:
  1. All three agents observe the current padded price array P_full^(k).
  2. For every inner cell (i,j,l), each agent extracts her slice through the
     full padded array, traces her contour, computes her posterior, demands.
  3. Market clearing yields P_inner^(k+1)[i,j,l] for every inner realization.
  4. The ghost ring of P_full^(k+1) is rebuilt by linear extrapolation along
     each axis from the current inner iterate (NOT from the no-learning
     baseline). This way the boundary anchor moves with the iterate instead
     of pinning the solution to the FR shape.

Run: python3 level_k_extrap.py
"""

import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Model primitives
# --------------------------------------------------------------------------

def Lam(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def logit(p):
    return np.log(p / (1.0 - p))

def f_v(u, tau, v):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - mean) ** 2)

def crra_demand(mu, p, gamma, W=1.0):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)

def market_clear(mus, gamma, kind="crra", W=1.0):
    if kind == "cara":
        return Lam(np.mean(logit(np.clip(mus, 1e-6, 1 - 1e-6))))
    def Z(p):
        return sum(crra_demand(mu, p, gamma, W) for mu in mus)
    lo, hi = 1e-5, 1 - 1e-5
    Zlo, Zhi = Z(lo), Z(hi)
    if Zlo * Zhi > 0:
        return hi if Zlo > 0 else lo
    return brentq(Z, lo, hi, xtol=1e-12)

def no_learning_price(u_triple, gamma, tau, kind):
    mus = [Lam(tau * u) for u in u_triple]
    return market_clear(mus, gamma, kind)


# --------------------------------------------------------------------------
# Endogenous extrapolation: build padded array from inner iterate
# --------------------------------------------------------------------------

EPS = 1e-4

def extrapolate_padded(P_inner, G_in, N_pad):
    """
    Build a (G_full)^3 padded array from a (G_in)^3 inner array by linear
    extrapolation along each axis, applied sequentially. Ghost cells are
    filled iteratively from the nearest two cells along that axis.
    """
    G_full = G_in + 2 * N_pad
    pad = N_pad
    P = np.empty((G_full, G_full, G_full))
    P[:] = np.nan
    P[pad:pad + G_in, pad:pad + G_in, pad:pad + G_in] = P_inner

    def extrap_axis(arr, axis):
        # left ghost: indices pad-1, pad-2, ..., 0
        for k in range(1, N_pad + 1):
            i_target = pad - k
            i_a = pad - k + 1
            i_b = pad - k + 2
            sl_t = [slice(None)] * 3
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_t[axis] = i_target
            sl_a[axis] = i_a
            sl_b[axis] = i_b
            arr[tuple(sl_t)] = 2 * arr[tuple(sl_a)] - arr[tuple(sl_b)]
        # right ghost: indices pad+G_in, ..., G_full-1
        for k in range(1, N_pad + 1):
            i_target = pad + G_in - 1 + k
            i_a = pad + G_in - 1 + k - 1
            i_b = pad + G_in - 1 + k - 2
            sl_t = [slice(None)] * 3
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_t[axis] = i_target
            sl_a[axis] = i_a
            sl_b[axis] = i_b
            arr[tuple(sl_t)] = 2 * arr[tuple(sl_a)] - arr[tuple(sl_b)]

    extrap_axis(P, 0)  # fills (ghost-0, inner-1, inner-2)
    extrap_axis(P, 1)  # fills (any-0, ghost-1, inner-2) using prior fills
    extrap_axis(P, 2)  # fills (any-0, any-1, ghost-2)

    return np.clip(P, EPS, 1 - EPS)


def build_initial_padded(u_full, gamma, tau, kind):
    """No-learning prices on the full padded grid -- only used at level 0."""
    n = len(u_full)
    P = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                P[i, j, l] = no_learning_price((u_full[i], u_full[j], u_full[l]),
                                               gamma, tau, kind)
    return P


# --------------------------------------------------------------------------
# 2-pass contour through one slice
# --------------------------------------------------------------------------

def contour_passes(slice_2d, u_grid, target_p, tau):
    n = len(u_grid)
    A0 = 0.0
    A1 = 0.0
    # Pass A: sweep rows on grid, root-find col off grid.
    for i in range(n):
        row = slice_2d[i, :]
        diffs = row - target_p
        for j in range(n - 1):
            d0, d1 = diffs[j], diffs[j + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                b_star = u_grid[j] + t * (u_grid[j + 1] - u_grid[j])
                A0 += f_v(u_grid[i], tau, 0) * f_v(b_star, tau, 0)
                A1 += f_v(u_grid[i], tau, 1) * f_v(b_star, tau, 1)
            elif d0 == 0:
                A0 += f_v(u_grid[i], tau, 0) * f_v(u_grid[j], tau, 0)
                A1 += f_v(u_grid[i], tau, 1) * f_v(u_grid[j], tau, 1)
    # Pass B: sweep cols on grid, root-find row off grid.
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
    return 0.5 * A0, 0.5 * A1


# --------------------------------------------------------------------------
# One Phi evaluation: compute new inner block using full padded P for contours
# --------------------------------------------------------------------------

def Phi_inner(P_full, u_full, G_in, N_pad, gamma, tau, kind):
    pad = N_pad
    P_new_inner = np.empty((G_in, G_in, G_in))
    for ii in range(G_in):
        for jj in range(G_in):
            for ll in range(G_in):
                i_full = pad + ii
                j_full = pad + jj
                l_full = pad + ll
                p = P_full[i_full, j_full, l_full]
                u1, u2, u3 = u_full[i_full], u_full[j_full], u_full[l_full]

                # Each agent's slice through the FULL padded array
                A0_1, A1_1 = contour_passes(P_full[i_full, :, :], u_full, p, tau)
                A0_2, A1_2 = contour_passes(P_full[:, j_full, :], u_full, p, tau)
                A0_3, A1_3 = contour_passes(P_full[:, :, l_full], u_full, p, tau)

                def post(u_own, A0, A1):
                    n1 = f_v(u_own, tau, 1) * A1
                    n0 = f_v(u_own, tau, 0) * A0
                    if n0 + n1 == 0:
                        return Lam(tau * u_own)
                    return n1 / (n0 + n1)

                mu1 = np.clip(post(u1, A0_1, A1_1), 1e-6, 1 - 1e-6)
                mu2 = np.clip(post(u2, A0_2, A1_2), 1e-6, 1 - 1e-6)
                mu3 = np.clip(post(u3, A0_3, A1_3), 1e-6, 1 - 1e-6)

                try:
                    p_new = market_clear([mu1, mu2, mu3], gamma, kind)
                except ValueError:
                    p_new = p
                P_new_inner[ii, jj, ll] = np.clip(p_new, EPS, 1 - EPS)
    return P_new_inner


# --------------------------------------------------------------------------
# 1-R^2 on the inner block
# --------------------------------------------------------------------------

def compute_R2(P_inner, u_inner, tau):
    n = len(u_inner)
    Y, X, W = [], [], []
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
                Y.append(logit(p)); X.append(T); W.append(w)
    Y = np.array(Y); X = np.array(X); W = np.array(W)
    W = W / W.sum()
    Ymean = (W * Y).sum(); Xmean = (W * X).sum()
    cov = (W * (Y - Ymean) * (X - Xmean)).sum()
    vy = (W * (Y - Ymean) ** 2).sum()
    vx = (W * (X - Xmean) ** 2).sum()
    if vy * vx == 0:
        return 0.0
    return 1.0 - cov ** 2 / (vy * vx)


# --------------------------------------------------------------------------
# Symmetrize over the 6 permutations
# --------------------------------------------------------------------------

def symmetrize(P):
    out = np.zeros_like(P)
    for ax in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        out += np.transpose(P, ax)
    return out / 6.0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(G_in=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0,
        gamma=0.5, tau=2.0, kind="crra", levels=20, verbose=True):
    G_full = G_in + 2 * N_pad
    u_inner = np.linspace(-UMAX, UMAX, G_in)
    pad_left = np.linspace(-UMAX_PAD, -UMAX, N_pad + 1)[:-1]
    pad_right = np.linspace(UMAX, UMAX_PAD, N_pad + 1)[1:]
    u_full = np.concatenate([pad_left, u_inner, pad_right])

    # Level 0: no-learning prices everywhere on the padded grid
    P_full = build_initial_padded(u_full, gamma, tau, kind)
    pad = N_pad
    P_inner = P_full[pad:pad + G_in, pad:pad + G_in, pad:pad + G_in].copy()

    R2_NL = compute_R2(P_inner, u_inner, tau)
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  kind={kind}, gamma={gamma}, tau={tau}")
        print(f"  G_in={G_in}, N_pad={N_pad}, UMAX={UMAX}, UMAX_PAD={UMAX_PAD}")
        print(f"  Ring rebuilt by linear extrapolation from inner iterate each level")
        print(f"{'=' * 64}")
        print(f"Level 0 (no-learning):  1-R^2 = {R2_NL:.6f}")

    history = [(0, 0.0, 0.0, R2_NL)]
    for k in range(levels):
        # Each level: compute new inner using current padded array, then rebuild ring.
        P_new_inner = Phi_inner(P_full, u_full, G_in, N_pad, gamma, tau, kind)
        P_new_inner = symmetrize(P_new_inner)

        delta = np.max(np.abs(P_new_inner - P_inner))

        # Rebuild padded grid from the new inner iterate
        P_full = extrapolate_padded(P_new_inner, G_in, N_pad)
        P_inner = P_new_inner

        R2 = compute_R2(P_inner, u_inner, tau)
        history.append((k + 1, delta, delta, R2))
        if verbose:
            print(f"Level {k + 1:2d}: ||delta||={delta:.4e}  1-R^2={R2:.6f}")

        if delta < 1e-7:
            if verbose:
                print(f"  converged at level {k + 1}")
            break

    return P_full, P_inner, u_full, u_inner, history


if __name__ == "__main__":
    print("\n### CARA at G_in=5, tau=2 (endogenous-extrapolation ring) ###")
    _, P_cara_in, _, u_inner, h_cara = run(G_in=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0,
                                            gamma=1.0, tau=2.0, kind="cara", levels=20)

    print("\n### CRRA gamma=0.5 at G_in=5, tau=2 (endogenous-extrapolation ring) ###")
    _, P_crra_in, _, _, h_crra = run(G_in=5, N_pad=2, UMAX=2.0, UMAX_PAD=3.0,
                                      gamma=0.5, tau=2.0, kind="crra", levels=20)

    # Reference realization (1, -1, 1)
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    print("\n### Reference realization (u1,u2,u3) = (+1,-1,+1) ###")
    print(f"  CARA price : {P_cara_in[i_p1, i_m1, i_p1]:.4f}  (target 0.8808)")
    print(f"  CRRA price : {P_crra_in[i_p1, i_m1, i_p1]:.4f}  (target 0.9077)")

    # Net signal
    R2_cara_final = h_cara[-1][3]
    R2_crra_final = h_crra[-1][3]
    print("\n### Final 1-R^2 ###")
    print(f"  CARA (artifact floor): {R2_cara_final:.6f}")
    print(f"  CRRA                 : {R2_crra_final:.6f}")
    print(f"  NET (CRRA - CARA)    : {R2_crra_final - R2_cara_final:+.6f}")
