"""
G refinement sweep at fixed domain: G_inner in {5, 9, 13}, all on
inner domain [-2sigma, +2sigma] with a 2-sigma buffer (so extended grid
is [-4sigma, +4sigma]). CRRA gamma=0.5, tau=2.

Per-level CONTOUR PLOTS for each G:
  3-panel figure showing contour lines of P_ext on each agent's middle slice,
  with the inner grid box marked.

Each level prints:  G | level | 1-R^2 | slope | ||delta||  to stdout.

Final cross-G summary plot saved to python/plots_sweep/summary.png.
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
SIGMA = 1.0 / np.sqrt(TAU)
INNER_HALFWIDTH = 2.0 * SIGMA          # 2 sigma
BUFFER_PER_SIDE = 2.0 * SIGMA          # 2 extra sigma in the ring
N_PAD = 2                              # 2 buffer points per side, evenly placed at +1sigma and +2sigma

PLOTS_ROOT = os.path.join(os.path.dirname(__file__), "plots_sweep")
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


# ---------- grid build ----------
def make_grids(G_inner):
    u_inner = np.linspace(-INNER_HALFWIDTH, +INNER_HALFWIDTH, G_inner)
    pad_left = np.array([-INNER_HALFWIDTH - 2 * SIGMA, -INNER_HALFWIDTH - 1 * SIGMA])
    pad_right = np.array([+INNER_HALFWIDTH + 1 * SIGMA, +INNER_HALFWIDTH + 2 * SIGMA])
    u_ext = np.concatenate([pad_left, u_inner, pad_right])
    G_ext = len(u_ext)
    inner_idx = np.arange(N_PAD, N_PAD + G_inner)
    return u_inner, u_ext, G_ext, inner_idx


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


def phi_step(P_ext, u_ext, inner_idx, G_inner):
    P_new = np.empty((G_inner, G_inner, G_inner))
    for ii in range(G_inner):
        for jj in range(G_inner):
            for ll in range(G_inner):
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


def deficit(P_inner, u_inner, G_inner):
    Y, X, Wts = [], [], []
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
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


# ---------- contour-line plot per level ----------
def plot_contours(level, P_ext, u_inner, u_ext, G_ext, R2, slope, delta, path, G_inner):
    """3-panel figure: contour lines of P_ext on each agent's middle (own=0) slice."""
    i_mid = G_ext // 2  # corresponds to u=0 (since grid is symmetric and odd-length)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"G={G_inner}  Level={level}  slope={slope:.4f}  1-R²={R2:.6f}  ||ΔP||∞={delta:.3e}  "
        f"(CRRA γ={GAMMA}, τ={TAU}, ±2σ inner / ±4σ ext)",
        fontsize=11,
    )

    levels = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])

    panels = [
        ("Agent 1 (fixed u₁=0)",  P_ext[i_mid, :, :], "u₃", "u₂"),
        ("Agent 2 (fixed u₂=0)",  P_ext[:, i_mid, :], "u₃", "u₁"),
        ("Agent 3 (fixed u₃=0)",  P_ext[:, :, i_mid], "u₂", "u₁"),
    ]
    for ax, (title, slice_2d, xlabel, ylabel) in zip(axes, panels):
        # contour lines in u_ext x u_ext (so we see how contours behave in the buffer)
        X, Y = np.meshgrid(u_ext, u_ext)  # x along axis-1 of slice (cols), y along axis-0
        cs = ax.contour(X, Y, slice_2d, levels=levels, colors="C0", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        # Reference: FR contour at p=0.5 -> straight line u_a + u_b = 0
        # Reference: any FR level p has contour u_a + u_b = (logit(p) - tau*0)/tau = logit(p)/tau
        # since FR for own=0 means logit(p) = tau(u_a + u_b)/3 ... no wait that's wrong, we should
        # plot what no-learning would look like: logit(p_NL) = (1/3)(tau*0 + tau u_a + tau u_b)/... etc
        # Skip for clarity; just show CRRA contours.
        # Mark inner grid box
        ax.axhline(u_inner[0], color="red", lw=0.6, ls=":")
        ax.axhline(u_inner[-1], color="red", lw=0.6, ls=":")
        ax.axvline(u_inner[0], color="red", lw=0.6, ls=":")
        ax.axvline(u_inner[-1], color="red", lw=0.6, ls=":")
        # Mark grid points
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


# ---------- run a single G ----------
def run_for_G(G_inner, maxlevel=15, tol=1e-7):
    u_inner, u_ext, G_ext, inner_idx = make_grids(G_inner)
    plots_dir = os.path.join(PLOTS_ROOT, f"g{G_inner}")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n=== G_inner={G_inner}  G_ext={G_ext}  spacing(inner)={(u_inner[1]-u_inner[0]):.4f}  "
          f"plots={plots_dir} ===", flush=True)

    t0 = time.time()
    P_ext = build_P_ext(u_ext, G_ext)
    print(f"G={G_inner}  build_P_ext: {time.time() - t0:.1f}s  ({G_ext**3} cells)", flush=True)

    P_inner = P_ext[np.ix_(inner_idx, inner_idx, inner_idx)].copy()
    R2_0, slope_0 = deficit(P_inner, u_inner, G_inner)
    history = [(0, 0.0, R2_0, slope_0)]
    print(f"G={G_inner}  LEVEL  0  1-R²={R2_0:.6f}  slope={slope_0:.4f}  (no-learning)", flush=True)
    plot_contours(0, P_ext, u_inner, u_ext, G_ext, R2_0, slope_0, 0.0,
                  os.path.join(plots_dir, "level_00.png"), G_inner)

    P_prev = P_inner.copy()
    for k in range(1, maxlevel + 1):
        t = time.time()
        P_phi = phi_step(P_ext, u_ext, inner_idx, G_inner)
        delta = float(np.max(np.abs(P_phi - P_prev)))
        P_ext[np.ix_(inner_idx, inner_idx, inner_idx)] = P_phi
        R2, slope = deficit(P_phi, u_inner, G_inner)
        history.append((k, delta, R2, slope))
        elapsed = time.time() - t
        print(f"G={G_inner}  LEVEL {k:2d}  1-R²={R2:.6f}  slope={slope:.4f}  "
              f"||Δ||={delta:.3e}  ({elapsed:.1f}s)", flush=True)
        plot_contours(k, P_ext, u_inner, u_ext, G_ext, R2, slope, delta,
                      os.path.join(plots_dir, f"level_{k:02d}.png"), G_inner)
        if delta < tol:
            print(f"G={G_inner}  converged at level {k}", flush=True)
            break
        P_prev = P_phi

    return history


def plot_summary(all_histories, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"G refinement sweep: CRRA γ={GAMMA}, τ={TAU}, ±2σ inner / ±4σ ext", fontsize=11)

    # Panel A: 1-R² vs level for each G
    ax = axes[0, 0]
    for G, hist in all_histories.items():
        ks = [h[0] for h in hist]
        R2s = [h[2] for h in hist]
        ax.semilogy(ks, R2s, "o-", label=f"G={G}")
    ax.set_xlabel("level"); ax.set_ylabel("1 - R²")
    ax.set_title("Revelation deficit per level")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel B: slope vs level for each G
    ax = axes[0, 1]
    for G, hist in all_histories.items():
        ks = [h[0] for h in hist]
        slopes = [h[3] for h in hist]
        ax.plot(ks, slopes, "o-", label=f"G={G}")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR (slope=1)")
    ax.set_xlabel("level"); ax.set_ylabel("regression slope")
    ax.set_title("Slope per level"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel C: 1-R² vs 1/G at final level
    ax = axes[1, 0]
    Gs = sorted(all_histories.keys())
    final_R2 = [all_histories[G][-1][2] for G in Gs]
    final_slope = [all_histories[G][-1][3] for G in Gs]
    inv_G = [1.0 / G for G in Gs]
    ax.plot(inv_G, final_R2, "o-", color="C1")
    for G, x, y in zip(Gs, inv_G, final_R2):
        ax.annotate(f"G={G}", (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("1 / G"); ax.set_ylabel("final 1 - R²")
    ax.set_title("Convergence in G")
    ax.grid(True, alpha=0.3)

    # Panel D: slope vs 1/G
    ax = axes[1, 1]
    ax.plot(inv_G, final_slope, "o-", color="C2")
    for G, x, y in zip(Gs, inv_G, final_slope):
        ax.annotate(f"G={G}", (x, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR")
    ax.set_xlabel("1 / G"); ax.set_ylabel("final slope")
    ax.set_title("Slope convergence in G")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print(f"G REFINEMENT SWEEP  (CRRA γ={GAMMA}, τ={TAU}, ±2σ inner / ±4σ ext)", flush=True)
    print("=" * 70, flush=True)

    Gs = [5, 9, 13]
    histories = {}
    for G in Gs:
        histories[G] = run_for_G(G, maxlevel=15, tol=1e-7)

    summary_path = os.path.join(PLOTS_ROOT, "summary.png")
    plot_summary(histories, summary_path)
    print(f"\nDONE.  saved per-G level plots and summary.png to {PLOTS_ROOT}", flush=True)

    # final compact table
    print("\nFINAL TABLE:")
    print(f"{'G':>4}  {'spacing':>8}  {'last level':>10}  {'1-R²':>10}  {'slope':>8}")
    for G in Gs:
        u_inner = np.linspace(-INNER_HALFWIDTH, +INNER_HALFWIDTH, G)
        sp = u_inner[1] - u_inner[0]
        h = histories[G][-1]
        print(f"{G:>4}  {sp:>8.4f}  {h[0]:>10}  {h[2]:>10.6f}  {h[3]:>8.4f}")
