"""
One Phi step (level-1 best response) on the no-learning prices, CRRA gamma=0.5, tau=2.

Setup:
  - Inner grid u_inner = {-2, -1, 0, 1, 2}, G_inner = 5.
  - 2-SD buffer at sigma = 1/sqrt(tau) = 0.7071: 2*sigma = 1.4142.
  - Extended grid: u_ext = {-3.4142, -2.7071, -2, -1, 0, 1, 2, 2.7071, 3.4142},
    G_ext = 9 points per axis. Inner spacing 1.0, buffer spacing ~0.71.
  - Buffer prices are filled with the same analytic no-learning market clearing,
    so the extended array is exact no-learning everywhere.
  - For each inner realization (i,j,l), each agent extracts her 9x9 extended slice,
    runs 2-pass contour tracing with linear interpolation, allowing crossings
    anywhere in [-3.4142, +3.4142]^2.
  - Posterior via Bayes; market clearing produces P^(1)[i,j,l]. No iteration.
"""

import numpy as np
from scipy.optimize import brentq

# ---------- parameters ----------
TAU = 2.0
GAMMA = 0.5
W = 1.0
G_INNER = 5
UMAX_INNER = 2.0
SIGMA = 1.0 / np.sqrt(TAU)        # ~0.7071
BUFFER = 2.0 * SIGMA              # ~1.4142

u_inner = np.linspace(-UMAX_INNER, UMAX_INNER, G_INNER)
u_ext = np.array([-UMAX_INNER - BUFFER, -UMAX_INNER - SIGMA,
                  *u_inner.tolist(),
                  +UMAX_INNER + SIGMA, +UMAX_INNER + BUFFER])
G_EXT = len(u_ext)                # 9
PAD = 2                           # buffer cells per side

# Indices of the inner grid inside u_ext:
inner_idx_in_ext = np.arange(PAD, PAD + G_INNER)   # [2,3,4,5,6]


# ---------- model primitives ----------
def Lam(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def logit(p): return np.log(p / (1.0 - p))

def f_v(u, tau, v):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2.0 * (u - mean) ** 2)

def crra_demand(mu, p, gamma=GAMMA, W=W):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)

def crra_clear(mus):
    def Z(p): return sum(crra_demand(mu, p) for mu in mus)
    return brentq(Z, 1e-9, 1 - 1e-9, xtol=1e-14)


# ---------- build extended no-learning price array ----------
def build_P_ext():
    P = np.empty((G_EXT, G_EXT, G_EXT))
    for i in range(G_EXT):
        for j in range(G_EXT):
            for l in range(G_EXT):
                mus = [Lam(TAU * u_ext[i]), Lam(TAU * u_ext[j]), Lam(TAU * u_ext[l])]
                P[i, j, l] = crra_clear(mus)
    return P


# ---------- 2-pass contour with linear interpolation ----------
def trace_contour(slice_2d, u_grid, target_p):
    """
    Returns list of crossings as tuples (axis, a_grid, b_offgrid, w0, w1) where
    axis indicates which sweep (A or B), a_grid is on the grid axis, b_offgrid is
    the linearly interpolated other coordinate. Also returns A0, A1 sums.
    """
    n = len(u_grid)
    crossings = []
    A0 = 0.0; A1 = 0.0

    # Pass A: sweep rows on grid (axis 0), root-find col off grid (axis 1)
    for i in range(n):
        row = slice_2d[i, :]
        diffs = row - target_p
        for j in range(n - 1):
            d0, d1 = diffs[j], diffs[j + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                b_star = u_grid[j] + t * (u_grid[j + 1] - u_grid[j])
                w0 = f_v(u_grid[i], TAU, 0) * f_v(b_star, TAU, 0)
                w1 = f_v(u_grid[i], TAU, 1) * f_v(b_star, TAU, 1)
                A0 += w0; A1 += w1
                crossings.append(("A", u_grid[i], b_star, w0, w1))
            elif d0 == 0:
                w0 = f_v(u_grid[i], TAU, 0) * f_v(u_grid[j], TAU, 0)
                w1 = f_v(u_grid[i], TAU, 1) * f_v(u_grid[j], TAU, 1)
                A0 += w0; A1 += w1
                crossings.append(("A", u_grid[i], u_grid[j], w0, w1))

    # Pass B: sweep cols on grid (axis 1), root-find row off grid (axis 0)
    for j in range(n):
        col = slice_2d[:, j]
        diffs = col - target_p
        for i in range(n - 1):
            d0, d1 = diffs[i], diffs[i + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                a_star = u_grid[i] + t * (u_grid[i + 1] - u_grid[i])
                w0 = f_v(a_star, TAU, 0) * f_v(u_grid[j], TAU, 0)
                w1 = f_v(a_star, TAU, 1) * f_v(u_grid[j], TAU, 1)
                A0 += w0; A1 += w1
                crossings.append(("B", a_star, u_grid[j], w0, w1))
            elif d0 == 0:
                w0 = f_v(u_grid[i], TAU, 0) * f_v(u_grid[j], TAU, 0)
                w1 = f_v(u_grid[i], TAU, 1) * f_v(u_grid[j], TAU, 1)
                A0 += w0; A1 += w1
                crossings.append(("B", u_grid[i], u_grid[j], w0, w1))

    return crossings, 0.5 * A0, 0.5 * A1


# ---------- one Phi step ----------
def phi_step(P_ext, verbose_at=None):
    """
    Compute P^(1)[i,j,l] for inner indices i,j,l in 0..G_INNER-1.
    If verbose_at is a tuple (i,j,l) of inner indices, dump full diagnostics.
    Returns P_inner_new (G_INNER^3) and a dict with diagnostics for verbose_at.
    """
    diag = {}
    P_new = np.empty((G_INNER, G_INNER, G_INNER))
    for ii in range(G_INNER):
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                i = inner_idx_in_ext[ii]
                j = inner_idx_in_ext[jj]
                l = inner_idx_in_ext[ll]
                p0 = P_ext[i, j, l]
                u1, u2, u3 = u_ext[i], u_ext[j], u_ext[l]

                # Each agent's 9x9 slice through P_ext
                slice_1 = P_ext[i, :, :]
                slice_2 = P_ext[:, j, :]
                slice_3 = P_ext[:, :, l]

                cs1, A0_1, A1_1 = trace_contour(slice_1, u_ext, p0)
                cs2, A0_2, A1_2 = trace_contour(slice_2, u_ext, p0)
                cs3, A0_3, A1_3 = trace_contour(slice_3, u_ext, p0)

                def post(u_own, A0, A1):
                    n1 = f_v(u_own, TAU, 1) * A1
                    n0 = f_v(u_own, TAU, 0) * A0
                    if n0 + n1 == 0:
                        return Lam(TAU * u_own)
                    return n1 / (n0 + n1)

                mu1 = post(u1, A0_1, A1_1)
                mu2 = post(u2, A0_2, A1_2)
                mu3 = post(u3, A0_3, A1_3)
                mu1 = float(np.clip(mu1, 1e-9, 1 - 1e-9))
                mu2 = float(np.clip(mu2, 1e-9, 1 - 1e-9))
                mu3 = float(np.clip(mu3, 1e-9, 1 - 1e-9))

                p1 = crra_clear([mu1, mu2, mu3])
                P_new[ii, jj, ll] = p1

                if verbose_at == (ii, jj, ll):
                    diag = {
                        "u": (u1, u2, u3), "p0": p0, "p1": p1,
                        "agent_data": [
                            ("agent 1", u1, cs1, A0_1, A1_1, mu1),
                            ("agent 2", u2, cs2, A0_2, A1_2, mu2),
                            ("agent 3", u3, cs3, A0_3, A1_3, mu3),
                        ],
                    }
    return P_new, diag


# ---------- 1 - R^2 on inner grid ----------
def deficit(P_inner):
    Y, X, Wts = [], [], []
    for i in range(G_INNER):
        for j in range(G_INNER):
            for l in range(G_INNER):
                p = P_inner[i, j, l]
                if not (1e-9 < p < 1 - 1e-9):
                    continue
                T = TAU * (u_inner[i] + u_inner[j] + u_inner[l])
                w = 0.5 * (
                    f_v(u_inner[i], TAU, 1) * f_v(u_inner[j], TAU, 1) * f_v(u_inner[l], TAU, 1)
                    + f_v(u_inner[i], TAU, 0) * f_v(u_inner[j], TAU, 0) * f_v(u_inner[l], TAU, 0)
                )
                Y.append(logit(p)); X.append(T); Wts.append(w)
    Y, X, Wts = np.array(Y), np.array(X), np.array(Wts)
    Wts = Wts / Wts.sum()
    Ybar = (Wts * Y).sum(); Xbar = (Wts * X).sum()
    cov = (Wts * (Y - Ybar) * (X - Xbar)).sum()
    vy = (Wts * (Y - Ybar) ** 2).sum()
    vx = (Wts * (X - Xbar) ** 2).sum()
    R2 = cov ** 2 / (vy * vx) if vy * vx > 0 else 0.0
    slope = cov / vx if vx > 0 else 0.0
    intercept = Ybar - slope * Xbar
    return 1.0 - R2, slope, intercept, len(Y)


# ---------- driver ----------
if __name__ == "__main__":
    print("=" * 78)
    print("LEVEL-1 STEP (one Phi update) on the no-learning prices")
    print("CRRA, gamma = {:.2f}, tau = {:.2f}".format(GAMMA, TAU))
    print("=" * 78)
    print(f"  Inner grid u_inner: {u_inner.tolist()}  (G_inner = {G_INNER})")
    print(f"  sigma = 1/sqrt(tau) = {SIGMA:.6f}   2*sigma = {BUFFER:.6f}")
    print(f"  Extended grid u_ext: {[f'{u:+.4f}' for u in u_ext]}  (G_ext = {G_EXT})")
    print(f"  Inner spacing = {u_inner[1] - u_inner[0]:.4f}, "
          f"buffer spacing = {u_ext[1] - u_ext[0]:.4f}")
    print(f"  Buffer prices: same analytic no-learning market clearing as inner.")
    print(f"  Contour root-find: linear interp in p-space, 2-pass.")
    print(f"  No iteration, no symmetrization.")
    print()

    print("Building extended no-learning price array (9x9x9 = 729 cells)...")
    P_ext = build_P_ext()
    print("  done.\n")

    # Inner block of P_ext = P^(0)_inner
    P0 = P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)]
    R2_0, slope0, intc0, n0 = deficit(P0)
    print(f"P^(0) (no-learning) on inner grid:")
    print(f"  weighted regression: logit(p) = {intc0:+.4f} + {slope0:+.4f} * T*")
    print(f"  1 - R^2 = {R2_0:.6f}   ({n0} interior triples)")
    print()

    # Locate canonical realization (+1,-1,+1) in inner indices
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    canonical_idx = (i_p1, i_m1, i_p1)
    canonical_u = (u_inner[i_p1], u_inner[i_m1], u_inner[i_p1])
    print(f"Canonical realization: inner_idx = {canonical_idx}, "
          f"(u1,u2,u3) = {canonical_u}")
    print()

    # One Phi step with diagnostics on the canonical realization
    print("Applying Phi step over all 125 inner cells...")
    P1, diag = phi_step(P_ext, verbose_at=canonical_idx)
    print("  done.\n")

    # ----- Canonical-realization detailed report -----
    print("-" * 78)
    print("DIAGNOSTICS at canonical realization (u1,u2,u3) = (+1,-1,+1)")
    print("-" * 78)
    u1, u2, u3 = diag["u"]
    p0 = diag["p0"]; p1 = diag["p1"]
    Tstar = TAU * (u1 + u2 + u3)
    print(f"  T* = tau*(u1+u2+u3) = {Tstar:.4f}")
    print(f"  Observed price p^(0) = {p0:.6f}  (no-learning at this realization)")
    print(f"  Updated  price p^(1) = {p1:.6f}  (after one Phi step)")
    print(f"  Delta p   = {p1 - p0:+.6f}")
    print()

    # Prior posteriors
    print("  PRIOR posteriors (own-signal only):")
    for k, u in enumerate([u1, u2, u3], start=1):
        print(f"    agent {k}: mu_prior = Lambda(tau * u_{k}) = "
              f"Lambda({TAU*u:+.2f}) = {Lam(TAU*u):.6f}")
    print()

    # Per-agent: contour data
    for label, u_own, cs, A0, A1, mu1 in diag["agent_data"]:
        u_ax_label = {"agent 1": ("u_2", "u_3"), "agent 2": ("u_1", "u_3"), "agent 3": ("u_1", "u_2")}[label]
        print(f"  {label.upper()}  (own signal u = {u_own:+.0f})")
        print(f"    contour {label}'s slice for p = {p0:.6f}")
        print(f"    crossings (axis  {u_ax_label[0]:>7}    {u_ax_label[1]:>7}    "
              f"f0*f0(crossing)   f1*f1(crossing)   T*-contrib):")
        # Sort crossings by which pass and grid coord
        for c in cs:
            ax, a, b, w0, w1 = c
            buf_a = "*" if abs(a) > UMAX_INNER + 1e-9 else " "
            buf_b = "*" if abs(b) > UMAX_INNER + 1e-9 else " "
            T_at = TAU * (u_own + a + b)
            print(f"      [{ax}]  {a:+.4f}{buf_a}  {b:+.4f}{buf_b}  "
                  f"{w0:.5e}     {w1:.5e}     T*={T_at:+.4f}")
        n_in = sum(1 for c in cs if abs(c[1]) <= UMAX_INNER + 1e-9
                   and abs(c[2]) <= UMAX_INNER + 1e-9)
        n_buf = len(cs) - n_in
        print(f"    crossings inside inner grid: {n_in},  in 2-SD buffer ring: {n_buf}")
        print(f"    A_0 = {A0:.6e}   A_1 = {A1:.6e}   ratio A_1/A_0 = {A1/A0:.4f}")
        u_own_lr = f_v(u_own, TAU, 1) / f_v(u_own, TAU, 0)
        print(f"    own-signal LR f1(u)/f0(u) = exp(tau*u) = {u_own_lr:.4f}")
        post_lr = u_own_lr * (A1 / A0)
        print(f"    posterior LR (own * contour) = {post_lr:.4f}")
        print(f"    mu^(1) = LR/(1+LR) = {mu1:.6f}   "
              f"(prior was {Lam(TAU*u_own):.6f})")
        print()

    # ----- Whole-grid summary -----
    R2_1, slope1, intc1, n1 = deficit(P1)
    print("-" * 78)
    print("WHOLE INNER GRID (125 cells)")
    print("-" * 78)
    print(f"  P^(0)  weighted regression: logit(p) = {intc0:+.4f} + {slope0:+.4f} * T*    "
          f"1-R^2 = {R2_0:.6f}")
    print(f"  P^(1)  weighted regression: logit(p) = {intc1:+.4f} + {slope1:+.4f} * T*    "
          f"1-R^2 = {R2_1:.6f}")
    print(f"  Delta(1-R^2)  = {R2_1 - R2_0:+.6f}")
    print(f"  Slope ratio   slope^(1) / slope^(0) = {slope1 / slope0:.4f}")
    print()

    # ----- u_1 = 0 slice of P^(1) for comparison with no-learning slice -----
    print("-" * 78)
    print("Slice P^(1)[i = 2 (u_1=0), :, :]")
    print("-" * 78)
    print(f"            u_3 =  {'   '.join(f'{u:+.0f}' for u in u_inner)}")
    for j in range(G_INNER):
        row = "  ".join(f"{P1[2, j, l]:.4f}" for l in range(G_INNER))
        print(f"   u_2 = {u_inner[j]:+.0f}  {row}")
    print()

    # ----- price change at the canonical row of all u_3, u_1 = +1, u_2 = -1 -----
    print("-" * 78)
    print(f"Price change p^(0) -> p^(1) along (u_1=+1, u_2=-1, u_3 varying)")
    print("-" * 78)
    print(f"  u_3       p^(0)        p^(1)        delta")
    for ll in range(G_INNER):
        u3v = u_inner[ll]
        p0v = P0[i_p1, i_m1, ll]
        p1v = P1[i_p1, i_m1, ll]
        print(f"  {u3v:+.0f}    {p0v:.6f}     {p1v:.6f}     {p1v - p0v:+.6f}")
    print()
    print("(Crossings marked '*' fall inside the 2-SD buffer ring outside the inner grid.)")
