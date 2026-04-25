"""Paper-ready plots for REZN without noise traders.

Produces three figures:
  A. plot_two_branches.png — two co-existing CRRA equilibria at γ=(3,3,3):
     pre-jump (near-CARA-FR) and post-jump (strong PR).
  B. plot_asymmetric.png — 1-R² vs τ for asymmetric γ=(5,3,1).
  C. plot_overview.png — single figure combining both panels for the paper.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FWD  = "/home/user/REZN/python/pchip_continuation_results.csv"
BWD  = "/home/user/REZN/python/pchip_backward_results.csv"
ASYM = "/home/user/REZN/python/pchip_asymmetric_results.csv"

OUT_A = "/home/user/REZN/python/plot_two_branches.png"
OUT_B = "/home/user/REZN/python/plot_asymmetric.png"
OUT_C = "/home/user/REZN/python/plot_overview.png"


def load_homo3(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                # Accept rows that solved to reasonable precision; conv=1 is
                # F_TOL-strict but post-jump rows often plateau at 1e-5.
                try:
                    if float(r["Finf"]) > 1e-3: continue
                except Exception:
                    if int(r["converged"]) != 1: continue
                if not (abs(float(r["gamma_1"])-3.0)<1e-9
                        and abs(float(r["gamma_2"])-3.0)<1e-9
                        and abs(float(r["gamma_3"])-3.0)<1e-9): continue
                out.append((float(r["tau_1"]), float(r["oneR2_het"]),
                            float(r["pr_gap"]), float(r["p_star"])))
            except Exception:
                continue
    out.sort()
    return out


def load_asym(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                # Accept rows that solved to reasonable precision; conv=1 is
                # F_TOL-strict but post-jump rows often plateau at 1e-5.
                try:
                    if float(r["Finf"]) > 1e-3: continue
                except Exception:
                    if int(r["converged"]) != 1: continue
                out.append((float(r["tau_1"]), float(r["oneR2_het"]),
                            float(r["pr_gap"]), float(r["p_star"])))
            except Exception:
                continue
    out.sort()
    return out


def split_pre_post(rows, thr=5e-3):
    return [r for r in rows if r[1] < thr], [r for r in rows if r[1] >= thr]


# ============================================================
# Figure A: two-branch coexistence for γ=(3,3,3)
# ============================================================
fwd  = load_homo3(FWD)
bwd  = load_homo3(BWD)
asym = load_asym(ASYM)

fwd_pre, fwd_post = split_pre_post(fwd)
bwd_pre, bwd_post = split_pre_post(bwd)

print(f"Forward pre-jump: {len(fwd_pre)}  post-jump: {len(fwd_post)}")
print(f"Backward post-jump: {len(bwd_post)}")
print(f"Asymmetric: {len(asym)}")

fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

ax = axs[0]
if fwd_pre:
    x, y = zip(*[(r[0], r[1]) for r in fwd_pre])
    ax.semilogy(x, y, "o", color="#1f77b4", ms=4,
                label=f"near-CARA-FR branch (n={len(fwd_pre)})")
post_all = sorted(set(fwd_post + bwd_post))
if post_all:
    x, y = zip(*[(r[0], r[1]) for r in post_all])
    ax.semilogy(x, y, "s", color="#d62728", ms=4,
                label=f"strong-PR branch (n={len(post_all)})")
ax.set_ylabel("1 − R²$_\\mathrm{het}$")
ax.set_title("Two co-existing CRRA equilibria at γ=(3,3,3), G=11 PCHIP")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")

ax = axs[1]
if fwd_pre:
    x, y = zip(*[(r[0], r[2]) for r in fwd_pre])
    ax.plot(x, y, "o", color="#1f77b4", ms=4)
if post_all:
    x, y = zip(*[(r[0], r[2]) for r in post_all])
    ax.plot(x, y, "s", color="#d62728", ms=4)
ax.axhline(0.0, color="k", lw=0.5)
ax.set_xlabel("τ (homogeneous)")
ax.set_ylabel("PR gap  μ₁ − μ₂  at (u1,u2,u3)=(1,−1,1)")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_A, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_A}")


# ============================================================
# Figure B: asymmetric γ=(5,3,1)
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

ax = axs[0]
if asym:
    x = [r[0] for r in asym]
    y = [r[1] for r in asym]
    ax.plot(x, y, "o-", color="#2ca02c", ms=5,
            label=f"CRRA equilibrium (n={len(asym)})")
    # reference: 1-R² = 0 under CARA
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5,
               label="CARA-FR reference (1−R² = 0)")
ax.set_ylabel("1 − R²$_\\mathrm{het}$")
ax.set_ylim(0, 0.4)
ax.set_title("Asymmetric preferences drive strong partial revelation\n"
             "γ=(5, 3, 1), G=11 PCHIP")
ax.grid(True, alpha=0.3)
ax.legend(loc="center right")

ax = axs[1]
if asym:
    x = [r[0] for r in asym]
    y = [r[2] for r in asym]
    ax.plot(x, y, "s-", color="#9467bd", ms=5)
ax.axhline(0.0, color="k", lw=0.5)
ax.set_xlabel("τ (homogeneous across signals)")
ax.set_ylabel("PR gap  μ₁ − μ₂  at (u1,u2,u3)=(1,−1,1)")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_B, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_B}")


# ============================================================
# Figure C: overview combining both phenomena for the paper
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# left: two branches at γ=(3,3,3)
ax = axs[0]
if fwd_pre:
    x, y = zip(*[(r[0], r[1]) for r in fwd_pre])
    ax.semilogy(x, y, "o", color="#1f77b4", ms=4,
                label=f"near-CARA branch (n={len(fwd_pre)})")
if post_all:
    x, y = zip(*[(r[0], r[1]) for r in post_all])
    ax.semilogy(x, y, "s", color="#d62728", ms=4,
                label=f"strong-PR branch (n={len(post_all)})")
ax.set_xlabel("τ (homogeneous)")
ax.set_ylabel("1 − R²$_\\mathrm{het}$")
ax.set_title("(a) Two equilibria at γ=(3,3,3)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best", fontsize=9)

# right: asymmetric
ax = axs[1]
if asym:
    x = [r[0] for r in asym]
    y = [r[1] for r in asym]
    ax.plot(x, y, "o-", color="#2ca02c", ms=5,
            label=f"γ=(5, 3, 1)  (n={len(asym)})")
    # overlay the γ=(3,3,3) pre-jump branch for comparison
    if fwd_pre:
        x2, y2 = zip(*[(r[0], r[1]) for r in fwd_pre])
        ax.plot(x2, y2, "s-", color="#1f77b4", ms=3, alpha=0.6,
                label=f"γ=(3,3,3) pre-jump branch")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5,
               label="CARA-FR (1−R² = 0)")
ax.set_xlabel("τ")
ax.set_ylabel("1 − R²$_\\mathrm{het}$")
ax.set_title("(b) Heterogeneity drives strong PR")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9)

fig.suptitle("REZN without noise traders: CRRA equilibria at G=11 (PCHIP)",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_C, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_C}")
