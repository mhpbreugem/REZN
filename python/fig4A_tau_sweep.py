"""Fig 4A: REE 1-R² vs τ at γ ∈ {0.25, 1, 4}, G=15.

τ ∈ [0.2, 8.0], 14 log-spaced values.
Warm-start chain along τ: solve at τ=2 (paper baseline) first,
then sweep down to 0.2 and up to 8.
Strict NK with trim=0.05.
"""
import json, time, warnings, os
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from posterior_method_v3 import (
    Lam, init_p_grid, phi_step, measure_R2, EPS,
)
from gap_reparam import pava_p_only, pava_u_only

UMAX = 4.0; G = 15; TOL_MAX = 1e-14; TRIM = 0.05
RESULTS_DIR = "results/full_ree"

GAMMAS = [0.25, 1.0, 4.0]
TAUS = list(np.exp(np.linspace(np.log(0.2), np.log(8.0), 14)))
# Sort so τ=2 is first (paper baseline), then sweep down then up
def sort_taus(taus, anchor=2.0):
    sorted_taus = sorted(taus, key=lambda t: abs(t - anchor))
    return sorted_taus


def pava_2d(mu): return pava_u_only(pava_p_only(mu))


def interp_mu(mu_old, u_old, p_old, u_new, p_new):
    G_new = len(u_new)
    out = np.empty((G_new, p_new.shape[1]))
    for i in range(G_new):
        u = u_new[i]; u_c = np.clip(u, u_old[0], u_old[-1])
        ra = np.searchsorted(u_old, u_c)
        rb = max(ra - 1, 0); ra = min(ra, len(u_old) - 1)
        w = (u_c - u_old[rb]) / (u_old[ra] - u_old[rb]) if ra != rb else 1.0
        for j in range(p_new.shape[1]):
            p = p_new[i, j]
            p_c1 = np.clip(p, p_old[rb, 0], p_old[rb, -1])
            mb = np.interp(p_c1, p_old[rb, :], mu_old[rb, :])
            p_c2 = np.clip(p, p_old[ra, 0], p_old[ra, -1])
            ma = np.interp(p_c2, p_old[ra, :], mu_old[ra, :])
            out[i, j] = (1 - w) * mb + w * ma
    return np.clip(out, EPS, 1 - EPS)


def picard_round(mu, u_grid, p_grid, p_lo, p_hi, n, na, alpha, tau, gamma):
    mu_sum = np.zeros_like(mu); n_collected = 0
    for it in range(n):
        cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
        cand = pava_2d(cand)
        mu = alpha * cand + (1 - alpha) * mu
        mu = np.clip(mu, EPS, 1 - EPS)
        if it >= n - na:
            mu_sum += mu; n_collected += 1
    return pava_2d(mu_sum / n_collected)


def F_phi(x, shape, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    mu = np.clip(x.reshape(shape), EPS, 1 - EPS)
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = (cand - mu); F[~active] = 0.0
    return F.ravel()


def measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma):
    cand, active, _ = phi_step(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    F = np.abs(cand - mu)
    return {
        "max": float(F[active].max()) if active.any() else float("nan"),
        "med": float(np.median(F[active])) if active.any() else float("nan"),
        "u_viol": int((np.diff(mu, axis=0) < 0).sum()),
        "p_viol": int((np.diff(mu, axis=1) < 0).sum()),
    }


def solve(tau, gamma, mu_warm, u_warm, p_warm, label=""):
    u_grid = np.linspace(-UMAX, UMAX, G)
    p_lo, p_hi, p_grid = init_p_grid(u_grid, tau, gamma, G, trim=TRIM)
    if mu_warm is None:
        mu = np.zeros((G, G))
        for i, u in enumerate(u_grid):
            mu[i, :] = Lam(tau * u)
    else:
        mu = interp_mu(mu_warm, u_warm, p_warm, u_grid, p_grid)
    mu = pava_2d(mu)
    t0 = time.time()
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 3000, 1500, 0.05, tau, gamma)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 3000, 1500, 0.01, tau, gamma)
    mu = picard_round(mu, u_grid, p_grid, p_lo, p_hi, 5000, 2500, 0.003, tau, gamma)
    d = measure(mu, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    if d["max"] < TOL_MAX and d["u_viol"] == 0 and d["p_viol"] == 0:
        return mu, d, u_grid, p_grid, p_lo, p_hi, "strict_picard", time.time()-t0
    try:
        sol = newton_krylov(
            lambda x: F_phi(x, mu.shape, u_grid, p_grid, p_lo, p_hi, tau, gamma),
            mu.ravel(), f_tol=TOL_MAX, maxiter=200, method="lgmres",
            verbose=False)
        mu_nk = np.clip(sol.reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "ok"
    except NoConvergence as e:
        mu_nk = np.clip(e.args[0].reshape(mu.shape), EPS, 1 - EPS)
        nk_status = "noconv"
    except (ValueError, RuntimeError) as exc:
        mu_nk = mu; nk_status = f"err"
    d_nk = measure(mu_nk, u_grid, p_grid, p_lo, p_hi, tau, gamma)
    if d_nk["max"] < TOL_MAX and d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "strict_NK", time.time()-t0
    if d_nk["u_viol"] == 0 and d_nk["p_viol"] == 0:
        return mu_nk, d_nk, u_grid, p_grid, p_lo, p_hi, "non_strict_monotone", time.time()-t0
    return mu, d, u_grid, p_grid, p_lo, p_hi, "fallback_picard", time.time()-t0


warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"=== Fig 4A: REE τ-sweep at γ={GAMMAS}, G={G} ===\n", flush=True)
all_curves = {}

for gamma in GAMMAS:
    print(f"\n--- γ = {gamma} ---", flush=True)
    # Try existing G=15 ckpt for this γ at τ=2
    ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15_gamma{gamma:g}.npz"
    if not os.path.exists(ck_path) and abs(gamma - 0.5) < 1e-6:
        ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15.npz"
    # Otherwise try G=14 strict at this γ
    if not os.path.exists(ck_path):
        g14 = f"{RESULTS_DIR}/posterior_v3_strict_G14_gamma{gamma:g}.npz"
        if os.path.exists(g14):
            ck_path = g14
        else:
            ck_path = f"{RESULTS_DIR}/posterior_v3_strict_G15.npz"
            print(f"  (warm from G=15 γ=0.5 — nearest)", flush=True)

    ck = np.load(ck_path)
    mu_w = ck["mu"]; u_w = ck["u_grid"]; p_w = ck["p_grid"]
    print(f"  Seeded from {ck_path}", flush=True)

    # Sort τ values: start near τ=2, sweep outward
    taus_sorted = sort_taus(TAUS, anchor=2.0)
    points = []
    mu_anchor = mu_w; u_anchor = u_w; p_anchor = p_w
    for tau in taus_sorted:
        mu_run, d, ug, pg, plo, phi_, status, elapsed = solve(
            tau, gamma, mu_anchor, u_anchor, p_anchor,
            label=f"γ={gamma}, τ={tau:.3g}")
        r2, slope, _ = measure_R2(mu_run, ug, pg, plo, phi_, tau, gamma)
        points.append({"tau": float(tau), "1-R2": float(r2),
                        "slope": float(slope), "max": d["max"],
                        "status": status, "t": elapsed})
        print(f"    τ={tau:6.3g}: 1-R²={r2:.4e}, slope={slope:.4f}, "
              f"max={d['max']:.2e}, {status}, t={elapsed:.0f}s",
              flush=True)
        # Update anchor if good
        if status.startswith("strict") or status == "non_strict_monotone":
            mu_anchor = mu_run; u_anchor = ug; p_anchor = pg
        # Save partial
        all_curves[f"{gamma:g}"] = {"gamma": gamma, "points": points}
        with open(f"{RESULTS_DIR}/fig_4A_data.json", "w") as f:
            json.dump({"figure": "fig_4A_REE_vs_tau",
                        "params": {"G": G, "gammas": GAMMAS,
                                    "tau_grid": list(TAUS)},
                        "curves": list(all_curves.values())}, f, indent=2)

print("\n=== DONE Fig 4A ===")

# Render
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
colors = [GREEN, RED, BLUE]
styles = ["-", "--", ":"]
for c, color, style in zip(all_curves.values(), colors, styles):
    pts = sorted(c["points"], key=lambda p: p["tau"])
    ts = [p["tau"] for p in pts]
    rs = [p["1-R2"] for p in pts]
    ax.plot(ts, rs, color=color, ls=style, lw=2, marker="o", ms=4,
              label=f"$\\gamma={c['gamma']}$")
ax.set_xscale("log")
ax.set_xlabel("$\\tau$"); ax.set_ylabel("$1-R^2$")
ax.set_title(f"Fig 4A: REE 1-R² vs τ\nG={G}", fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_4A_preview.pdf", bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_4A_preview.png", dpi=150,
              bbox_inches="tight")
plt.close(fig)
print(f"Saved fig_4A_preview.{{pdf,png}}", flush=True)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_4A_pgfplots.tex", "w") as f:
    for c in all_curves.values():
        pts = sorted(c["points"], key=lambda p: p["tau"])
        f.write(f"% gamma={c['gamma']}\n")
        f.write(f"\\addplot coordinates {{"
                f"{pgf([(p['tau'], p['1-R2']) for p in pts])}}};\n\n")
print(f"Saved fig_4A_pgfplots.tex", flush=True)
