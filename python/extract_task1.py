"""Task 1: Extract figures from seed posterior_v3_G20_umax5_notrim_mp300.json.

Per origin/main:FIGURES_TODO.md (UPDATED 2026-05-02):
- 1a: Fig 3B CRRA contour lines (300x300, σ=1.5, contours at p=0.2,0.3,0.5,0.7,0.8)
- 1b: Fig 5 price vs T* (50 symmetric triples, 3 curves: FR, NL, REE)
- 1c: Fig 6B CRRA posteriors vs T* (50 triples u1=+1, u2=-1, u3=T*/τ swap)
- 1d: 1-R² from seed (regress logit(p_REE) on T* across all G³ triples)

All in float64. Outputs to results/full_ree/.
"""
import json
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = "results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json"
OUTDIR = "results/full_ree"


def load_seed():
    with open(SEED) as f:
        d = json.load(f)
    G = d["G"]
    UMAX = d["UMAX"]
    tau = d["tau"]
    gamma = d["gamma"]
    u_grid = np.array([float(s) for s in d["u_grid"]])
    p_grid = np.array([[float(s) for s in row] for row in d["p_grid"]])
    mu = np.array([[float(s) for s in row] for row in d["mu_strings"]])
    return G, UMAX, tau, gamma, u_grid, p_grid, mu


def Lam(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def crra_demand(mu, p, gamma, W=1.0):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(z)
    return W * (R - 1.0) / ((1.0 - p) + R * p)


def mu_interp_one(u, p, u_grid, p_grid, mu):
    """Bilinear interp of μ*(u, p). p_grid is per-row (G x G).

    For fixed u: find two adjacent rows ia, ib in u_grid; interp μ along
    each row's p-grid to the requested p, then linearly blend in u.
    """
    G = len(u_grid)
    if u <= u_grid[0]:
        ia = ib = 0; w_u = 0.0
    elif u >= u_grid[-1]:
        ia = ib = G - 1; w_u = 0.0
    else:
        ib = int(np.searchsorted(u_grid, u))
        ia = ib - 1
        w_u = (u - u_grid[ia]) / (u_grid[ib] - u_grid[ia])

    def row_mu(i):
        p_row = p_grid[i]
        if p <= p_row[0]: return mu[i, 0]
        if p >= p_row[-1]: return mu[i, -1]
        return float(np.interp(p, p_row, mu[i]))

    if ia == ib:
        return row_mu(ia)
    return (1 - w_u) * row_mu(ia) + w_u * row_mu(ib)


def market_clear(u_triple, u_grid, p_grid, mu, gamma, p_lo=1e-4, p_hi=1-1e-4):
    """Solve Σ d(μ*(u_k, p), p) = 0 for p ∈ [p_lo, p_hi]."""
    def F(p):
        s = 0.0
        for uk in u_triple:
            mk = mu_interp_one(uk, p, u_grid, p_grid, mu)
            s += crra_demand(mk, p, gamma)
        return s
    f_lo = F(p_lo); f_hi = F(p_hi)
    if f_lo * f_hi > 0:
        # Outside bracket — return whichever endpoint has smaller |F|
        return p_lo if abs(f_lo) < abs(f_hi) else p_hi
    return brentq(F, p_lo, p_hi, xtol=1e-12)


def market_clear_no_learning(u_triple, tau, gamma, p_lo=1e-4, p_hi=1-1e-4):
    """No-learning: prior μ_k = Λ(τ u_k), solve market clearing."""
    priors = [Lam(tau * uk) for uk in u_triple]

    def F(p):
        return sum(crra_demand(mk, p, gamma) for mk in priors)
    f_lo = F(p_lo); f_hi = F(p_hi)
    if f_lo * f_hi > 0:
        return p_lo if abs(f_lo) < abs(f_hi) else p_hi
    return brentq(F, p_lo, p_hi, xtol=1e-12)


# ---------------- Task 1a: Fig 3B contours ----------------

def task_1a(u_grid, p_grid, mu, gamma):
    print("\n=== Task 1a: Fig 3B contour lines ===")
    i1 = int(np.argmin(np.abs(u_grid - 1.0)))
    u1 = float(u_grid[i1])
    print(f"  u1 = u_grid[{i1}] = {u1:.4f} (closest to 1.0)")

    N = 300
    u23 = np.linspace(-3.5, 3.5, N)
    P = np.empty((N, N))
    print(f"  Building {N}x{N} price surface...", flush=True)
    for i in range(N):
        if i % 30 == 0:
            print(f"    row {i}/{N}", flush=True)
        for j in range(N):
            u2 = float(u23[i]); u3 = float(u23[j])
            P[i, j] = market_clear((u1, u2, u3), u_grid, p_grid, mu, gamma)
    print(f"  Price surface: min={P.min():.4f}, max={P.max():.4f}")

    # Smooth
    P_smooth = gaussian_filter(P, sigma=1.5)

    # Extract contours
    levels = [0.2, 0.3, 0.5, 0.7, 0.8]
    fig, ax = plt.subplots()
    cs = ax.contour(u23, u23, P_smooth, levels=levels)
    plt.close(fig)

    out_path = f"{OUTDIR}/fig3B_G20_pgfplots.tex"
    lines = ["% Fig 3B CRRA contour lines",
             f"% G=20 UMAX=5 γ={gamma} τ=2 u1={u1:.4f}",
             f"% Smoothing: σ=1.5 px", ""]
    # In matplotlib >=3.8 use cs.allsegs, in older use collections.
    try:
        all_segs = cs.allsegs
    except AttributeError:
        all_segs = [[p.vertices for p in c.get_paths()]
                    for c in cs.collections]

    for level, segs in zip(levels, all_segs):
        # Pick longest segment (the dominant contour at this level)
        if not segs: continue
        seg = max(segs, key=lambda s: len(s))
        # Arc-length resample to 50 points
        d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(seg, axis=0), axis=1)])
        if d[-1] == 0:
            xy = seg[:1]
        else:
            t = np.linspace(0, d[-1], 50)
            xs = np.interp(t, d, seg[:, 0])
            ys = np.interp(t, d, seg[:, 1])
            xy = np.column_stack([xs, ys])
        coords = "".join(f"({x:.4f},{y:.4f})" for x, y in xy)
        lines.append(f"% p = {level}")
        lines.append(f"\\addplot coordinates {{{coords}}};")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")
    return P, P_smooth


# ---------------- Task 1b: Fig 5 ----------------

def task_1b(u_grid, p_grid, mu, gamma, tau):
    print("\n=== Task 1b: Fig 5 price vs T* ===")
    Ts = np.linspace(-8.0, 8.0, 50)
    p_FR = Lam(Ts / 3.0)
    p_NL = np.empty_like(Ts)
    p_REE = np.empty_like(Ts)
    for k, T in enumerate(Ts):
        u = float(T) / (3.0 * tau)
        p_NL[k] = market_clear_no_learning((u, u, u), tau, gamma)
        p_REE[k] = market_clear((u, u, u), u_grid, p_grid, mu, gamma)

    out_path = f"{OUTDIR}/fig5_G20_pgfplots.tex"
    lines = ["% Fig 5 price vs T* (symmetric triples)",
             f"% G=20 UMAX=5 γ={gamma} τ={tau}", ""]

    def fmt(xs, ys):
        return "".join(f"({x:.4f},{y:.6f})" for x, y in zip(xs, ys))

    lines += ["% FR (analytical)",
              f"\\addplot coordinates {{{fmt(Ts, p_FR)}}};", ""]
    lines += ["% NL (no learning)",
              f"\\addplot coordinates {{{fmt(Ts, p_NL)}}};", ""]
    lines += ["% REE (using μ*)",
              f"\\addplot coordinates {{{fmt(Ts, p_REE)}}};", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")
    print(f"  Sample: T*=0 → FR={p_FR[len(Ts)//2]:.4f}, NL={p_NL[len(Ts)//2]:.4f}, "
          f"REE={p_REE[len(Ts)//2]:.4f}")


# ---------------- Task 1c: Fig 6B ----------------

def task_1c(u_grid, p_grid, mu, gamma, tau):
    print("\n=== Task 1c: Fig 6B CRRA posteriors vs T* ===")
    Ts = np.linspace(-3.0, 4.0, 50)
    # u1=+1 fixed, u2=-1 fixed, u3 chosen so that T* = τ(u1+u2+u3) → u3 = T*/τ
    u1 = 1.0; u2 = -1.0
    mu1 = np.empty_like(Ts)
    mu2 = np.empty_like(Ts)
    p_REE = np.empty_like(Ts)
    for k, T in enumerate(Ts):
        u3 = float(T) / tau
        p = market_clear((u1, u2, u3), u_grid, p_grid, mu, gamma)
        mu1[k] = mu_interp_one(u1, p, u_grid, p_grid, mu)
        mu2[k] = mu_interp_one(u2, p, u_grid, p_grid, mu)
        p_REE[k] = p

    out_path = f"{OUTDIR}/fig6B_G20_pgfplots.tex"
    lines = ["% Fig 6B CRRA posteriors vs T*",
             f"% G=20 UMAX=5 γ={gamma} τ={tau}",
             f"% u1=+1 u2=-1 u3=T*/τ", ""]

    def fmt(xs, ys):
        return "".join(f"({x:.4f},{y:.6f})" for x, y in zip(xs, ys))

    lines += ["% μ1 (agent 1, u=+1)",
              f"\\addplot coordinates {{{fmt(Ts, mu1)}}};", ""]
    lines += ["% μ2 (agent 2, u=-1)",
              f"\\addplot coordinates {{{fmt(Ts, mu2)}}};", ""]
    lines += ["% price (REE)",
              f"\\addplot coordinates {{{fmt(Ts, p_REE)}}};", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


# ---------------- Task 1d: 1-R² ----------------

def task_1d(u_grid, p_grid, mu, gamma, tau, G):
    print("\n=== Task 1d: 1-R² from seed ===")
    n = G * G * G
    Tstar = np.empty(n)
    logit_p = np.empty(n)
    valid = np.zeros(n, dtype=bool)
    k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                u = (float(u_grid[i]), float(u_grid[j]), float(u_grid[l]))
                p = market_clear(u, u_grid, p_grid, mu, gamma)
                T = tau * (u[0] + u[1] + u[2])
                Tstar[k] = T
                if 1e-10 < p < 1 - 1e-10:
                    logit_p[k] = logit(p)
                    valid[k] = True
                k += 1
        if i % 5 == 0:
            print(f"  row {i+1}/{G}", flush=True)

    T_v = Tstar[valid]; lp_v = logit_p[valid]
    # Regress lp = a + b T
    b, a = np.polyfit(T_v, lp_v, 1)
    pred = a + b * T_v
    ss_res = float(np.sum((lp_v - pred) ** 2))
    ss_tot = float(np.sum((lp_v - lp_v.mean()) ** 2))
    R2 = 1 - ss_res / ss_tot
    one_minus_R2 = 1 - R2

    out_path = f"{OUTDIR}/G20_umax5_R2.json"
    rec = {
        "params": {"G": G, "UMAX": 5.0, "gamma": gamma, "tau": tau},
        "1-R2": one_minus_R2,
        "slope": float(b),
        "intercept": float(a),
        "n_triples": int(valid.sum()),
        "n_total": int(n),
    }
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  1-R² = {one_minus_R2:.6e}")
    print(f"  slope = {b:.6f}")
    print(f"  n = {valid.sum()} / {n}")
    print(f"  Saved {out_path}")


# ---------------- Main ----------------
if __name__ == "__main__":
    G, UMAX, tau, gamma, u_grid, p_grid, mu = load_seed()
    print(f"Loaded seed: G={G} UMAX={UMAX} γ={gamma} τ={tau}")
    print(f"  μ shape: {mu.shape}, range [{mu.min():.4e}, {mu.max():.4e}]")

    task_1a(u_grid, p_grid, mu, gamma)
    task_1b(u_grid, p_grid, mu, gamma, tau)
    task_1c(u_grid, p_grid, mu, gamma, tau)
    task_1d(u_grid, p_grid, mu, gamma, tau, G)
    print("\nAll Task 1 extractions complete.")
