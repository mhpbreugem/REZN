"""
Classic undamped Picard (alpha=1) on the level-k Phi map, with diagnostic
plots saved at every level.

For each level k:
  - level_<kk>.png: 6-panel figure
       Top row: price slices at u_1 = -1, 0, +1
       Bottom row: agent posteriors mu_1, mu_2, mu_3 vs T*
trajectory.png: 4-panel summary across all levels
                (slope, 1-R^2, ||delta||, canonical p) vs level

CRRA gamma=0.5, tau=2. G_inner=5. Buffer 2-SD ring held at no-learning.
"""

import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from level1_step import (
    TAU, GAMMA, G_INNER, u_inner, u_ext, inner_idx_in_ext, PAD,
    build_P_ext, deficit, Lam, logit, f_v,
    crra_clear, trace_contour,
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- Phi step that ALSO returns the posteriors ----------
def phi_step_with_posteriors(P_ext):
    """Returns (P_inner_new, mu1, mu2, mu3) all shape G_INNER^3."""
    P_new = np.empty((G_INNER, G_INNER, G_INNER))
    Mu1 = np.empty_like(P_new); Mu2 = np.empty_like(P_new); Mu3 = np.empty_like(P_new)
    for ii in range(G_INNER):
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                i = inner_idx_in_ext[ii]
                j = inner_idx_in_ext[jj]
                l = inner_idx_in_ext[ll]
                p0 = P_ext[i, j, l]
                u1, u2, u3 = u_ext[i], u_ext[j], u_ext[l]
                _, A0_1, A1_1 = trace_contour(P_ext[i, :, :], u_ext, p0)
                _, A0_2, A1_2 = trace_contour(P_ext[:, j, :], u_ext, p0)
                _, A0_3, A1_3 = trace_contour(P_ext[:, :, l], u_ext, p0)

                def post(u_own, A0, A1):
                    n1 = f_v(u_own, TAU, 1) * A1
                    n0 = f_v(u_own, TAU, 0) * A0
                    if n0 + n1 == 0: return Lam(TAU * u_own)
                    return n1 / (n0 + n1)

                mu1 = float(np.clip(post(u1, A0_1, A1_1), 1e-9, 1 - 1e-9))
                mu2 = float(np.clip(post(u2, A0_2, A1_2), 1e-9, 1 - 1e-9))
                mu3 = float(np.clip(post(u3, A0_3, A1_3), 1e-9, 1 - 1e-9))
                Mu1[ii, jj, ll] = mu1
                Mu2[ii, jj, ll] = mu2
                Mu3[ii, jj, ll] = mu3
                P_new[ii, jj, ll] = crra_clear([mu1, mu2, mu3])
    return P_new, Mu1, Mu2, Mu3


# ---------- Plotting ----------
def plot_level(level_idx, P_inner, Mu1, Mu2, Mu3, R2, slope, delta, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Level {level_idx}   |   slope = {slope:.4f}   1-R² = {R2:.6f}   "
        f"||ΔP||∞ = {delta:.3e}   (CRRA γ={GAMMA}, τ={TAU}, G={G_INNER})",
        fontsize=12,
    )

    # Top row: price heatmaps at u_1 = -1, 0, +1
    for col, own_idx in enumerate([1, 2, 3]):  # u_1 = -1, 0, +1
        ax = axes[0, col]
        im = ax.imshow(P_inner[own_idx, :, :], origin="lower", vmin=0, vmax=1,
                       cmap="RdBu_r",
                       extent=[u_inner[0] - 0.5, u_inner[-1] + 0.5,
                               u_inner[0] - 0.5, u_inner[-1] + 0.5])
        ax.set_xlabel("u_3"); ax.set_ylabel("u_2")
        ax.set_title(f"P[u_1={u_inner[own_idx]:+.0f}]")
        ax.set_xticks(u_inner); ax.set_yticks(u_inner)
        # Annotate cells with price values
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                color = "white" if abs(P_inner[own_idx, jj, ll] - 0.5) > 0.3 else "black"
                ax.text(u_inner[ll], u_inner[jj],
                        f"{P_inner[own_idx, jj, ll]:.2f}",
                        ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.04)

    # Bottom row: posteriors vs T*
    Tstar = np.empty((G_INNER, G_INNER, G_INNER))
    for i in range(G_INNER):
        for j in range(G_INNER):
            for l in range(G_INNER):
                Tstar[i, j, l] = TAU * (u_inner[i] + u_inner[j] + u_inner[l])

    for col, (Mu, lbl, agent_axis) in enumerate([
        (Mu1, "μ₁", 0), (Mu2, "μ₂", 1), (Mu3, "μ₃", 2),
    ]):
        ax = axes[1, col]
        # Color by own signal
        own_signal = np.empty_like(Mu)
        for i in range(G_INNER):
            for j in range(G_INNER):
                for l in range(G_INNER):
                    if agent_axis == 0:   own_signal[i, j, l] = u_inner[i]
                    elif agent_axis == 1: own_signal[i, j, l] = u_inner[j]
                    else:                 own_signal[i, j, l] = u_inner[l]
        sc = ax.scatter(Tstar.flatten(), Mu.flatten(),
                        c=own_signal.flatten(), cmap="coolwarm",
                        vmin=-2, vmax=2, s=20, alpha=0.7, edgecolors="none")
        ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        # Reference: full revelation -> mu = Lambda(T*)
        T_grid = np.linspace(Tstar.min(), Tstar.max(), 200)
        ax.plot(T_grid, Lam(T_grid), "k--", linewidth=0.8,
                label="Λ(T*) (FR)", alpha=0.6)
        # Reference: prior posterior -> mu = Lambda(tau * u_own)
        u_range = np.linspace(-2, 2, 200)
        for u in u_inner:
            ax.scatter([u * TAU * 3], [Lam(TAU * u)], marker="x",
                       color="black", s=20, alpha=0.0)  # invisible, just for limits
        ax.set_xlabel("T*"); ax.set_ylabel(lbl)
        ax.set_title(f"{lbl}  (color = own signal)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="own signal")

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(history, path):
    levels = [h[0] for h in history]
    deltas = [h[1] for h in history]
    R2s    = [h[2] for h in history]
    slopes = [h[3] for h in history]
    p_cans = [h[4] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Picard trajectory (CRRA γ={GAMMA}, τ={TAU}, G={G_INNER}, undamped)",
                 fontsize=12)

    ax = axes[0, 0]
    ax.semilogy([k for k in levels if k > 0], [d for k, d in zip(levels, deltas) if k > 0],
                "o-", color="C0")
    ax.set_xlabel("level k"); ax.set_ylabel("||ΔP||∞")
    ax.set_title("Convergence proxy")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(levels, R2s, "o-", color="C1")
    ax.set_xlabel("level k"); ax.set_ylabel("1 - R²")
    ax.set_title("Revelation deficit")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(levels, slopes, "o-", color="C2")
    ax.axhline(1.0, color="k", linewidth=0.7, linestyle="--", label="FR slope = 1")
    ax.axhline(1.0/3, color="grey", linewidth=0.7, linestyle=":", label="no-learning slope = 1/3")
    ax.set_xlabel("level k"); ax.set_ylabel("regression slope")
    ax.set_title("Slope of logit(p) vs T*")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(levels, p_cans, "o-", color="C3")
    Tstar_can = TAU * (1 + (-1) + 1)
    ax.axhline(Lam(Tstar_can), color="k", linewidth=0.7, linestyle="--",
               label=f"Λ(T*) = {Lam(Tstar_can):.4f} (FR)")
    ax.axhline(Lam(Tstar_can / 3), color="grey", linewidth=0.7, linestyle=":",
               label=f"Λ(T*/3) = {Lam(Tstar_can/3):.4f} (no-learn)")
    ax.set_xlabel("level k"); ax.set_ylabel("p at (+1,-1,+1)")
    ax.set_title("Canonical price")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Driver ----------
def run(maxlevel=20, tol=1e-7):
    P_ext = build_P_ext()
    P_inner = P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)].copy()

    # Level 0: no-learning. Compute posteriors at level 0 too (which are just priors).
    Mu1_0 = np.empty_like(P_inner); Mu2_0 = np.empty_like(P_inner); Mu3_0 = np.empty_like(P_inner)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for l in range(G_INNER):
                Mu1_0[i, j, l] = Lam(TAU * u_inner[i])
                Mu2_0[i, j, l] = Lam(TAU * u_inner[j])
                Mu3_0[i, j, l] = Lam(TAU * u_inner[l])

    R2_0, slope_0, _, _ = deficit(P_inner)
    i_p1 = int(np.argmin(np.abs(u_inner - 1.0)))
    i_m1 = int(np.argmin(np.abs(u_inner + 1.0)))
    p_can_0 = P_inner[i_p1, i_m1, i_p1]
    history = [(0, 0.0, R2_0, slope_0, p_can_0)]

    print(f"Level 0  slope={slope_0:.4f}  1-R^2={R2_0:.6f}  p_canonical={p_can_0:.4f}")
    plot_level(0, P_inner, Mu1_0, Mu2_0, Mu3_0,
               R2_0, slope_0, 0.0,
               os.path.join(PLOTS_DIR, "level_00.png"))

    P_inner_prev = P_inner.copy()
    for k in range(1, maxlevel + 1):
        P_phi, Mu1, Mu2, Mu3 = phi_step_with_posteriors(P_ext)
        delta = float(np.max(np.abs(P_phi - P_inner_prev)))
        # Update inner block of P_ext, ring untouched
        P_ext[np.ix_(inner_idx_in_ext, inner_idx_in_ext, inner_idx_in_ext)] = P_phi
        R2, slope, _, _ = deficit(P_phi)
        p_can = P_phi[i_p1, i_m1, i_p1]
        history.append((k, delta, R2, slope, p_can))
        print(f"Level {k:2d}  delta={delta:.4e}  slope={slope:.4f}  "
              f"1-R^2={R2:.6f}  p_canonical={p_can:.4f}")

        plot_level(k, P_phi, Mu1, Mu2, Mu3, R2, slope, delta,
                   os.path.join(PLOTS_DIR, f"level_{k:02d}.png"))

        if delta < tol:
            print(f"Converged at level {k}.")
            break
        P_inner_prev = P_phi

    plot_trajectory(history, os.path.join(PLOTS_DIR, "trajectory.png"))
    return history


if __name__ == "__main__":
    print("=" * 70)
    print(f"PICARD WITH PLOTS  -- saving to {PLOTS_DIR}")
    print(f"  CRRA gamma={GAMMA}, tau={TAU}, G_inner={G_INNER}, undamped (alpha=1)")
    print("=" * 70)
    history = run(maxlevel=20, tol=1e-7)
    print()
    print(f"Saved {len(history)} level plots and trajectory.png to {PLOTS_DIR}")
