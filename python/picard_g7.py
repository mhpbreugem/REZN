"""
Classic undamped Picard at G_inner = 7, with inner grid spanning ±2σ and a
2-σ buffer ring (so the extended grid spans ±4σ), τ=2, γ=0.5, CRRA.

Per-level plots saved to python/plots_g7/.

Inner grid:    7 points uniformly in [-2σ, +2σ] = [-1.4142, +1.4142]
Buffer:        2 points per side at u = ±3σ, ±4σ = ±2.1213, ±2.8284
Extended grid: 11 points

Buffer prices held fixed at analytic no-learning. Inner block updated
each level by undamped Phi.  No symmetrization, no damping, no Newton.
"""
import os, time
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- parameters ----------
TAU = 2.0
GAMMA = 0.5
W = 1.0
G_INNER = 7
SIGMA = 1.0 / np.sqrt(TAU)
INNER_HALFWIDTH = 2.0 * SIGMA          # 2σ
BUFFER = 2.0 * SIGMA                   # 2σ extra in the ring
N_PAD = 2

u_inner = np.linspace(-INNER_HALFWIDTH, +INNER_HALFWIDTH, G_INNER)
pad_pts_left = np.array([-INNER_HALFWIDTH - 2 * SIGMA, -INNER_HALFWIDTH - 1 * SIGMA])
pad_pts_right = np.array([+INNER_HALFWIDTH + 1 * SIGMA, +INNER_HALFWIDTH + 2 * SIGMA])
u_ext = np.concatenate([pad_pts_left, u_inner, pad_pts_right])
G_EXT = len(u_ext)
inner_idx = np.arange(N_PAD, N_PAD + G_INNER)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots_g7")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- model primitives ----------
def Lam(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
def logit(p): return np.log(p / (1.0 - p))
def f_v(u, tau, v):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2.0 * (u - mean) ** 2)
def crra_demand(mu, p):
    z = (logit(mu) - logit(p)) / GAMMA
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)
def crra_clear(mus):
    def Z(p): return sum(crra_demand(mu, p) for mu in mus)
    return brentq(Z, 1e-9, 1 - 1e-9, xtol=1e-14)


# ---------- build extended no-learning prices ----------
def build_P_ext():
    P = np.empty((G_EXT, G_EXT, G_EXT))
    for i in range(G_EXT):
        for j in range(G_EXT):
            for l in range(G_EXT):
                mus = [Lam(TAU * u_ext[i]), Lam(TAU * u_ext[j]), Lam(TAU * u_ext[l])]
                P[i, j, l] = crra_clear(mus)
    return P


# ---------- 2-pass contour ----------
def trace_contour(slice_2d, u_grid, target_p):
    n = len(u_grid)
    A0 = 0.0; A1 = 0.0
    for i in range(n):
        row = slice_2d[i, :]; diffs = row - target_p
        for j in range(n - 1):
            d0, d1 = diffs[j], diffs[j + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                b = u_grid[j] + t * (u_grid[j + 1] - u_grid[j])
                A0 += f_v(u_grid[i], TAU, 0) * f_v(b, TAU, 0)
                A1 += f_v(u_grid[i], TAU, 1) * f_v(b, TAU, 1)
            elif d0 == 0:
                A0 += f_v(u_grid[i], TAU, 0) * f_v(u_grid[j], TAU, 0)
                A1 += f_v(u_grid[i], TAU, 1) * f_v(u_grid[j], TAU, 1)
    for j in range(n):
        col = slice_2d[:, j]; diffs = col - target_p
        for i in range(n - 1):
            d0, d1 = diffs[i], diffs[i + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                a = u_grid[i] + t * (u_grid[i + 1] - u_grid[i])
                A0 += f_v(a, TAU, 0) * f_v(u_grid[j], TAU, 0)
                A1 += f_v(a, TAU, 1) * f_v(u_grid[j], TAU, 1)
            elif d0 == 0:
                A0 += f_v(u_grid[i], TAU, 0) * f_v(u_grid[j], TAU, 0)
                A1 += f_v(u_grid[i], TAU, 1) * f_v(u_grid[j], TAU, 1)
    return 0.5 * A0, 0.5 * A1


# ---------- one Phi step with posteriors ----------
def phi_with_post(P_ext):
    P_new = np.empty((G_INNER, G_INNER, G_INNER))
    Mu1 = np.empty_like(P_new); Mu2 = np.empty_like(P_new); Mu3 = np.empty_like(P_new)
    for ii in range(G_INNER):
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                i = inner_idx[ii]; j = inner_idx[jj]; l = inner_idx[ll]
                p0 = P_ext[i, j, l]
                u1 = u_ext[i]; u2 = u_ext[j]; u3 = u_ext[l]
                A0_1, A1_1 = trace_contour(P_ext[i, :, :], u_ext, p0)
                A0_2, A1_2 = trace_contour(P_ext[:, j, :], u_ext, p0)
                A0_3, A1_3 = trace_contour(P_ext[:, :, l], u_ext, p0)

                def post(u_own, A0, A1):
                    n1 = f_v(u_own, TAU, 1) * A1
                    n0 = f_v(u_own, TAU, 0) * A0
                    if n0 + n1 == 0: return Lam(TAU * u_own)
                    return n1 / (n0 + n1)

                mu1 = float(np.clip(post(u1, A0_1, A1_1), 1e-9, 1 - 1e-9))
                mu2 = float(np.clip(post(u2, A0_2, A1_2), 1e-9, 1 - 1e-9))
                mu3 = float(np.clip(post(u3, A0_3, A1_3), 1e-9, 1 - 1e-9))
                Mu1[ii, jj, ll] = mu1; Mu2[ii, jj, ll] = mu2; Mu3[ii, jj, ll] = mu3
                P_new[ii, jj, ll] = crra_clear([mu1, mu2, mu3])
    return P_new, Mu1, Mu2, Mu3


# ---------- 1-R^2 ----------
def deficit(P_inner):
    Y, X, Wts = [], [], []
    for i in range(G_INNER):
        for j in range(G_INNER):
            for l in range(G_INNER):
                p = P_inner[i, j, l]
                if not (1e-9 < p < 1 - 1e-9): continue
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
    return 1.0 - R2, slope, Ybar - slope * Xbar, len(Y)


# ---------- plots ----------
def plot_level(level, P_inner, Mu1, Mu2, Mu3, R2, slope, delta, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Level {level}  |  slope = {slope:.4f}  1-R² = {R2:.6f}  ||ΔP||∞ = {delta:.3e}  "
        f"(CRRA γ={GAMMA}, τ={TAU}, G={G_INNER}, ±2σ inner, ±4σ ext)",
        fontsize=11,
    )
    # Top row: price slices at u_1 indices spanning the inner range
    show_idx = [1, G_INNER // 2, G_INNER - 2]   # near both edges and center
    for col, ii in enumerate(show_idx):
        ax = axes[0, col]
        im = ax.imshow(P_inner[ii, :, :], origin="lower", vmin=0, vmax=1, cmap="RdBu_r",
                       extent=[u_inner[0], u_inner[-1], u_inner[0], u_inner[-1]],
                       aspect='equal')
        ax.set_xlabel("u_3"); ax.set_ylabel("u_2")
        ax.set_title(f"P[u_1={u_inner[ii]:+.3f}]")
        ax.set_xticks(u_inner); ax.set_yticks(u_inner)
        ax.tick_params(axis='both', labelsize=7)
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                color = "white" if abs(P_inner[ii, jj, ll] - 0.5) > 0.3 else "black"
                ax.text(u_inner[ll], u_inner[jj],
                        f"{P_inner[ii, jj, ll]:.2f}",
                        ha="center", va="center", fontsize=5, color=color)
        plt.colorbar(im, ax=ax, fraction=0.04)

    # Bottom row: posteriors vs T*
    Tstar = np.empty((G_INNER, G_INNER, G_INNER))
    for i in range(G_INNER):
        for j in range(G_INNER):
            for l in range(G_INNER):
                Tstar[i, j, l] = TAU * (u_inner[i] + u_inner[j] + u_inner[l])

    for col, (Mu, lbl, agent_axis) in enumerate(
        [(Mu1, "μ₁", 0), (Mu2, "μ₂", 1), (Mu3, "μ₃", 2)]
    ):
        ax = axes[1, col]
        own_signal = np.empty_like(Mu)
        for i in range(G_INNER):
            for j in range(G_INNER):
                for l in range(G_INNER):
                    if agent_axis == 0:   own_signal[i, j, l] = u_inner[i]
                    elif agent_axis == 1: own_signal[i, j, l] = u_inner[j]
                    else:                 own_signal[i, j, l] = u_inner[l]
        sc = ax.scatter(Tstar.flatten(), Mu.flatten(),
                        c=own_signal.flatten(), cmap="coolwarm",
                        vmin=u_inner[0], vmax=u_inner[-1],
                        s=18, alpha=0.7, edgecolors="none")
        T_grid = np.linspace(Tstar.min(), Tstar.max(), 200)
        ax.plot(T_grid, Lam(T_grid), "k--", linewidth=0.8, label="Λ(T*) (FR)", alpha=0.6)
        ax.axhline(0.5, color="grey", lw=0.4, ls=":", alpha=0.6)
        ax.axvline(0, color="grey", lw=0.4, ls=":", alpha=0.6)
        ax.set_xlabel("T*"); ax.set_ylabel(lbl)
        ax.set_title(f"{lbl} (color = own signal)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="own signal")

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(history, path):
    levels = [h[0] for h in history]
    deltas = [h[1] for h in history]
    R2s = [h[2] for h in history]
    slopes = [h[3] for h in history]
    p_cans = [h[4] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Picard trajectory (CRRA γ={GAMMA}, τ={TAU}, G={G_INNER}, ±2σ inner, ±4σ ext)",
        fontsize=11,
    )
    ax = axes[0, 0]
    ax.semilogy([k for k in levels if k > 0],
                [d for k, d in zip(levels, deltas) if k > 0], "o-", color="C0")
    ax.set_xlabel("level k"); ax.set_ylabel("||ΔP||∞")
    ax.set_title("Convergence proxy"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(levels, R2s, "o-", color="C1")
    ax.set_xlabel("level k"); ax.set_ylabel("1 - R²")
    ax.set_title("Revelation deficit"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(levels, slopes, "o-", color="C2")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", label="FR slope = 1")
    ax.axhline(1.0 / 3, color="grey", lw=0.7, ls=":", label="no-learn slope = 1/3")
    ax.set_xlabel("level k"); ax.set_ylabel("regression slope")
    ax.set_title("Slope of logit(p) vs T*")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    can_T = TAU * (u_inner[-2] + u_inner[1] + u_inner[-2])  # approx (+2σ-step, -2σ-step, +2σ-step)
    ax.plot(levels, p_cans, "o-", color="C3")
    ax.axhline(Lam(can_T), color="k", lw=0.7, ls="--",
               label=f"Λ(T*≈{can_T:.2f}) (FR)")
    ax.axhline(Lam(can_T / 3), color="grey", lw=0.7, ls=":",
               label=f"Λ(T*/3) (no-learn)")
    ax.set_xlabel("level k"); ax.set_ylabel("p at canonical")
    ax.set_title("Canonical price"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------
def run(maxlevel=20, tol=1e-7):
    print(f"=== Picard at G_in={G_INNER}, ±2σ inner, ±4σ ext (CRRA γ={GAMMA}, τ={TAU}) ===",
          flush=True)
    print(f"u_inner ({G_INNER}): {[f'{u:+.4f}' for u in u_inner]}", flush=True)
    print(f"u_ext   ({G_EXT}): {[f'{u:+.4f}' for u in u_ext]}", flush=True)

    t_build = time.time()
    P_ext = build_P_ext()
    print(f"build_P_ext: {time.time() - t_build:.1f}s   "
          f"(G_ext^3 = {G_EXT**3} cells)", flush=True)

    P_inner = P_ext[np.ix_(inner_idx, inner_idx, inner_idx)].copy()
    Mu_prior = np.array([[[Lam(TAU * u_inner[i]) for l in range(G_INNER)]
                          for j in range(G_INNER)] for i in range(G_INNER)])
    R2_0, slope_0, _, _ = deficit(P_inner)
    can_idx = (G_INNER - 2, 1, G_INNER - 2)   # (+2σ-step, -2σ-step, +2σ-step) approx
    p_can_0 = float(P_inner[can_idx])
    history = [(0, 0.0, R2_0, slope_0, p_can_0)]
    print(f"LEVEL 0  slope={slope_0:.4f}  1-R^2={R2_0:.6f}  p_canonical={p_can_0:.4f}",
          flush=True)
    plot_level(0, P_inner,
               Mu_prior, np.transpose(Mu_prior, (1, 0, 2)), np.transpose(Mu_prior, (2, 0, 1)),
               R2_0, slope_0, 0.0, os.path.join(PLOTS_DIR, "level_00.png"))

    P_prev = P_inner.copy()
    for k in range(1, maxlevel + 1):
        t0 = time.time()
        P_phi, Mu1, Mu2, Mu3 = phi_with_post(P_ext)
        delta = float(np.max(np.abs(P_phi - P_prev)))
        P_ext[np.ix_(inner_idx, inner_idx, inner_idx)] = P_phi
        R2, slope, _, _ = deficit(P_phi)
        p_can = float(P_phi[can_idx])
        history.append((k, delta, R2, slope, p_can))
        elapsed = time.time() - t0
        print(f"LEVEL {k:2d}  delta={delta:.4e}  slope={slope:.4f}  "
              f"1-R^2={R2:.6f}  p_canonical={p_can:.4f}  ({elapsed:.1f}s)",
              flush=True)
        plot_level(k, P_phi, Mu1, Mu2, Mu3, R2, slope, delta,
                   os.path.join(PLOTS_DIR, f"level_{k:02d}.png"))
        if delta < tol:
            print(f"converged at level {k}.", flush=True)
            break
        P_prev = P_phi

    plot_trajectory(history, os.path.join(PLOTS_DIR, "trajectory.png"))
    print(f"DONE.  saved {len(history)} level plots and trajectory.png to {PLOTS_DIR}",
          flush=True)
    return history


if __name__ == "__main__":
    run(maxlevel=20, tol=1e-7)
