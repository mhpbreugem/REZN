"""
Inner-domain sweep at G=13: inner halfwidth in {1, 2, 3, 4} sigma.
Buffer always 2 sigma extra each side. CRRA gamma=0.5, tau=2.

Each run: 15 undamped Picard iterations on the level-k Phi map, with
analytic-no-learning ghost ring held fixed.

Per-level contour plots saved to python/plots_inner_sweep/inner_<n>sd/.
Cross-run summary saved to python/plots_inner_sweep/summary.png.
"""

import os, time
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- shared parameters ----------
TAU = 2.0
GAMMA = 0.5
W = 1.0
G_INNER = 13
SIGMA = 1.0 / np.sqrt(TAU)
BUFFER_SIGMAS = 2.0     # buffer always 2σ extra each side
N_PAD = 2

PLOTS_ROOT = os.path.join(os.path.dirname(__file__), "plots_inner_sweep")
os.makedirs(PLOTS_ROOT, exist_ok=True)


# ---------- model ----------
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


def make_grids(inner_sigmas):
    inner_half = inner_sigmas * SIGMA
    u_inner = np.linspace(-inner_half, +inner_half, G_INNER)
    pad_left = np.array([-inner_half - 2 * SIGMA, -inner_half - 1 * SIGMA])
    pad_right = np.array([+inner_half + 1 * SIGMA, +inner_half + 2 * SIGMA])
    u_ext = np.concatenate([pad_left, u_inner, pad_right])
    inner_idx = np.arange(N_PAD, N_PAD + G_INNER)
    return u_inner, u_ext, len(u_ext), inner_idx


def build_P_ext(u_ext, G_ext):
    P = np.empty((G_ext, G_ext, G_ext))
    for i in range(G_ext):
        for j in range(G_ext):
            for l in range(G_ext):
                mus = [Lam(TAU * u_ext[i]), Lam(TAU * u_ext[j]), Lam(TAU * u_ext[l])]
                P[i, j, l] = crra_clear(mus)
    return P


def trace_contour(slice_2d, u_grid, target_p):
    n = len(u_grid)
    A0, A1 = 0.0, 0.0
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


def phi_step(P_ext, u_ext, inner_idx):
    P_new = np.empty((G_INNER, G_INNER, G_INNER))
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
                P_new[ii, jj, ll] = crra_clear([mu1, mu2, mu3])
    return P_new


def deficit(P_inner, u_inner):
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
    return 1.0 - R2, slope


def plot_contours(level, P_ext, u_inner, u_ext, R2, slope, delta, path,
                  inner_sigmas):
    G_ext = len(u_ext)
    i_mid = G_ext // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"inner=±{inner_sigmas}σ  Level={level}  slope={slope:.4f}  "
        f"1-R²={R2:.6f}  ||ΔP||∞={delta:.3e}  (G=13, CRRA γ={GAMMA}, τ={TAU}, +2σ buffer)",
        fontsize=11,
    )
    levels = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    panels = [
        ("Agent 1 (fixed u₁=0)",  P_ext[i_mid, :, :], "u₃", "u₂"),
        ("Agent 2 (fixed u₂=0)",  P_ext[:, i_mid, :], "u₃", "u₁"),
        ("Agent 3 (fixed u₃=0)",  P_ext[:, :, i_mid], "u₂", "u₁"),
    ]
    for ax, (title, slice_2d, xlabel, ylabel) in zip(axes, panels):
        X, Y = np.meshgrid(u_ext, u_ext)
        cs = ax.contour(X, Y, slice_2d, levels=levels, colors="C0", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.axhline(u_inner[0], color="red", lw=0.6, ls=":")
        ax.axhline(u_inner[-1], color="red", lw=0.6, ls=":")
        ax.axvline(u_inner[0], color="red", lw=0.6, ls=":")
        ax.axvline(u_inner[-1], color="red", lw=0.6, ls=":")
        for ux in u_ext:
            for uy in u_ext:
                inside = u_inner[0] <= ux <= u_inner[-1] and u_inner[0] <= uy <= u_inner[-1]
                ax.plot(ux, uy, "k.", markersize=2.0 if inside else 1.0,
                        alpha=0.6 if inside else 0.3)
        ax.set_xlim(u_ext[0], u_ext[-1])
        ax.set_ylim(u_ext[0], u_ext[-1])
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_for_inner(inner_sigmas, maxlevel=15, tol=1e-7):
    u_inner, u_ext, G_ext, inner_idx = make_grids(inner_sigmas)
    plots_dir = os.path.join(PLOTS_ROOT, f"inner_{inner_sigmas}sd")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n=== inner_halfwidth = {inner_sigmas}σ ({inner_sigmas * SIGMA:.4f} u-units)  "
          f"G_ext={G_ext}  spacing(inner)={u_inner[1]-u_inner[0]:.4f} ===", flush=True)

    t0 = time.time()
    P_ext = build_P_ext(u_ext, G_ext)
    print(f"inner={inner_sigmas}σ  build_P_ext: {time.time() - t0:.1f}s  "
          f"({G_ext**3} cells)", flush=True)

    P_inner = P_ext[np.ix_(inner_idx, inner_idx, inner_idx)].copy()
    R2_0, slope_0 = deficit(P_inner, u_inner)
    history = [(0, 0.0, R2_0, slope_0)]
    print(f"inner={inner_sigmas}σ  LEVEL  0  1-R²={R2_0:.6f}  slope={slope_0:.4f}  (no-learning)",
          flush=True)
    plot_contours(0, P_ext, u_inner, u_ext, R2_0, slope_0, 0.0,
                  os.path.join(plots_dir, "level_00.png"), inner_sigmas)

    P_prev = P_inner.copy()
    for k in range(1, maxlevel + 1):
        t = time.time()
        P_phi = phi_step(P_ext, u_ext, inner_idx)
        delta = float(np.max(np.abs(P_phi - P_prev)))
        P_ext[np.ix_(inner_idx, inner_idx, inner_idx)] = P_phi
        R2, slope = deficit(P_phi, u_inner)
        history.append((k, delta, R2, slope))
        elapsed = time.time() - t
        print(f"inner={inner_sigmas}σ  LEVEL {k:2d}  1-R²={R2:.6f}  slope={slope:.4f}  "
              f"||Δ||={delta:.3e}  ({elapsed:.1f}s)", flush=True)
        plot_contours(k, P_ext, u_inner, u_ext, R2, slope, delta,
                      os.path.join(plots_dir, f"level_{k:02d}.png"), inner_sigmas)
        if delta < tol:
            print(f"inner={inner_sigmas}σ  converged at level {k}", flush=True)
            break
        P_prev = P_phi
    return history


def plot_summary(all_histories, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Inner-halfwidth sweep at G=13: CRRA γ={GAMMA}, τ={TAU}, +2σ buffer",
        fontsize=11,
    )
    inner_keys = sorted(all_histories.keys())

    ax = axes[0, 0]
    for s in inner_keys:
        hist = all_histories[s]
        ks = [h[0] for h in hist]
        R2s = [h[2] for h in hist]
        ax.semilogy(ks, R2s, "o-", label=f"inner=±{s}σ")
    ax.set_xlabel("level"); ax.set_ylabel("1 - R²")
    ax.set_title("Revelation deficit per level"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for s in inner_keys:
        hist = all_histories[s]
        ks = [h[0] for h in hist]
        slopes = [h[3] for h in hist]
        ax.plot(ks, slopes, "o-", label=f"inner=±{s}σ")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR")
    ax.axhline(1.0/3, color="grey", lw=0.7, ls=":", alpha=0.6, label="no-learn")
    ax.set_xlabel("level"); ax.set_ylabel("regression slope")
    ax.set_title("Slope per level"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    final_R2 = [all_histories[s][-1][2] for s in inner_keys]
    final_slope = [all_histories[s][-1][3] for s in inner_keys]
    ax.plot(inner_keys, final_R2, "o-", color="C1")
    for s, y in zip(inner_keys, final_R2):
        ax.annotate(f"{s}σ", (s, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("inner halfwidth (σ)"); ax.set_ylabel("final 1 - R²")
    ax.set_title("Final 1-R² vs inner half-width")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(inner_keys, final_slope, "o-", color="C2")
    for s, y in zip(inner_keys, final_slope):
        ax.annotate(f"{s}σ", (s, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR")
    ax.set_xlabel("inner halfwidth (σ)"); ax.set_ylabel("final slope")
    ax.set_title("Final slope vs inner half-width")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print(f"INNER-HALFWIDTH SWEEP at G={G_INNER}  (CRRA γ={GAMMA}, τ={TAU}, +2σ buffer)",
          flush=True)
    print("=" * 70, flush=True)

    inner_sds = [1, 2, 3, 4]
    histories = {}
    for s in inner_sds:
        histories[s] = run_for_inner(s, maxlevel=15, tol=1e-7)

    summary_path = os.path.join(PLOTS_ROOT, "summary.png")
    plot_summary(histories, summary_path)
    print(f"\nDONE.  saved level plots and summary.png to {PLOTS_ROOT}", flush=True)

    print("\nFINAL TABLE:")
    print(f"{'inner':>6}  {'spacing':>8}  {'1-R²':>10}  {'slope':>8}")
    for s in inner_sds:
        u_inner = np.linspace(-s * SIGMA, +s * SIGMA, G_INNER)
        sp = u_inner[1] - u_inner[0]
        h = histories[s][-1]
        print(f"{s}σ    {sp:>8.4f}  {h[2]:>10.6f}  {h[3]:>8.4f}")
