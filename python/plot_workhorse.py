"""Paper-quality figures for the REZN-without-NT workhorse model.

G=11 logit-PCHIP, UMAX=2.5. Sparse 10-point grids.

Figures:
  Fig 1: 1-R² vs τ at γ=(3,3,3) — two co-existing CRRA equilibria.
  Fig 2: 1-R² vs γ at τ=(3,3,3) — homogeneous γ sweep.
  Fig 3: 1-R² for γ=(5,3,1) vs τ — heterogeneity = strong PR.
  Fig 4: composite for the paper.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FWD = "/home/user/REZN/python/pchip_G11u25_forward.csv"
ASYM = "/home/user/REZN/python/pchip_asymmetric_results.csv"

OUT_FIG1 = "/home/user/REZN/python/fig1_branches.png"
OUT_FIG2 = "/home/user/REZN/python/fig2_gamma_sweep.png"
OUT_FIG3 = "/home/user/REZN/python/fig3_asymmetric.png"
OUT_FIG4 = "/home/user/REZN/python/fig4_composite.png"


def load(path, finf_max=1e-3):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                if float(r["Finf"]) > finf_max:
                    continue
            except Exception:
                continue
            try:
                rows.append({
                    "tau": float(r["tau_1"]),
                    "gamma": float(r["gamma_1"]),
                    "oneR2": float(r["oneR2_het"]),
                    "pr_gap": float(r["pr_gap"]),
                    "p_star": float(r["p_star"]),
                    "Finf": float(r["Finf"]),
                })
            except Exception:
                continue
    return rows


fwd = load(FWD)
asym = load(ASYM)

# Forward homogeneous γ=(3,3,3) τ-sweep
fwd_tau = sorted([r for r in fwd if abs(r["gamma"] - 3.0) < 1e-9],
                 key=lambda r: r["tau"])
# Forward homogeneous τ=(3,3,3) γ-sweep
fwd_gam = sorted([r for r in fwd if abs(r["tau"] - 3.0) < 1e-9],
                 key=lambda r: r["gamma"])

print(f"τ-sweep at γ=3: {len(fwd_tau)} pts")
print(f"γ-sweep at τ=3: {len(fwd_gam)} pts")
print(f"Asymmetric γ=(5,3,1): {len(asym)} pts")


# Branch separation in τ-sweep
tau_pre = [r for r in fwd_tau if r["oneR2"] < 5e-3]
tau_post = [r for r in fwd_tau if r["oneR2"] >= 5e-3]


# ============== FIG 1: τ-sweep two branches ==============
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))

ax = axs[0]
if tau_pre:
    x = [r["tau"] for r in tau_pre]; y = [r["oneR2"] for r in tau_pre]
    ax.semilogy(x, y, "o-", color="#1f77b4", ms=6, lw=1.5,
                label=f"near-CARA branch (n={len(tau_pre)})")
if tau_post:
    x = [r["tau"] for r in tau_post]; y = [r["oneR2"] for r in tau_post]
    ax.semilogy(x, y, "s-", color="#d62728", ms=6, lw=1.5,
                label=f"strong-PR branch (n={len(tau_post)})")
ax.set_xlabel("τ (homogeneous)", fontsize=12)
ax.set_ylabel("1 − R²", fontsize=12)
ax.set_title("(a) Two co-existing CRRA equilibria  γ=(3,3,3)",
             fontsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=10)

ax = axs[1]
if tau_pre:
    x = [r["tau"] for r in tau_pre]; y = [r["pr_gap"] for r in tau_pre]
    ax.plot(x, y, "o-", color="#1f77b4", ms=6, lw=1.5)
if tau_post:
    x = [r["tau"] for r in tau_post]; y = [r["pr_gap"] for r in tau_post]
    ax.plot(x, y, "s-", color="#d62728", ms=6, lw=1.5)
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("τ (homogeneous)", fontsize=12)
ax.set_ylabel("μ₁ − μ₂  at  (1,−1,1)", fontsize=12)
ax.set_title("(b) PR gap (single-cell)", fontsize=12)
ax.grid(True, alpha=0.3)

fig.suptitle("Two CRRA equilibria at γ=(3,3,3), G=11 PCHIP, UMAX=2.5",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_FIG1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG1}")


# ============== FIG 2: γ-sweep at τ=3 ==============
fig, ax = plt.subplots(figsize=(7.5, 4.5))
if fwd_gam:
    x = [r["gamma"] for r in fwd_gam]; y = [r["oneR2"] for r in fwd_gam]
    ax.loglog(x, y, "o-", color="#2ca02c", ms=7, lw=1.5,
              label=f"CRRA equilibrium (n={len(fwd_gam)})")
    # CARA-FR reference: 1-R²=0
ax.set_xlabel("γ (homogeneous risk aversion)", fontsize=12)
ax.set_ylabel("1 − R²", fontsize=12)
ax.set_title(f"PR strength vs γ at τ=(3,3,3), G=11 PCHIP",
             fontsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_FIG2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG2}")


# ============== FIG 3: asymmetric γ=(5,3,1) ==============
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))

ax = axs[0]
if asym:
    x = [r["tau"] for r in asym]; y = [r["oneR2"] for r in asym]
    ax.plot(x, y, "o-", color="#9467bd", ms=7, lw=1.5,
            label=f"γ=(5, 3, 1)  (n={len(asym)})")
    # overlay homogeneous γ=3 pre-jump branch for comparison
    if tau_pre:
        x2 = [r["tau"] for r in tau_pre]; y2 = [r["oneR2"] for r in tau_pre]
        ax.plot(x2, y2, "s-", color="#1f77b4", ms=4, lw=1, alpha=0.6,
                label="γ=(3,3,3) near-CARA")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4,
               label="CARA-FR (1−R² = 0)")
ax.set_xlabel("τ (homogeneous across signals)", fontsize=12)
ax.set_ylabel("1 − R²", fontsize=12)
ax.set_title("(a) Asymmetric γ produces strong PR",
             fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axs[1]
if asym:
    x = [r["tau"] for r in asym]; y = [r["pr_gap"] for r in asym]
    ax.plot(x, y, "s-", color="#9467bd", ms=7, lw=1.5)
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("τ", fontsize=12)
ax.set_ylabel("μ₁ − μ₂  at  (1,−1,1)", fontsize=12)
ax.set_title("(b) PR gap with γ=(5,3,1)", fontsize=12)
ax.grid(True, alpha=0.3)

fig.suptitle("Heterogeneity in risk aversion drives strong CRRA-PR",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_FIG3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG3}")


# ============== FIG 4: composite for paper ==============
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axs[0]
if tau_pre:
    x = [r["tau"] for r in tau_pre]; y = [r["oneR2"] for r in tau_pre]
    ax.semilogy(x, y, "o-", color="#1f77b4", ms=5, lw=1.5,
                label=f"near-CARA (n={len(tau_pre)})")
if tau_post:
    x = [r["tau"] for r in tau_post]; y = [r["oneR2"] for r in tau_post]
    ax.semilogy(x, y, "s-", color="#d62728", ms=5, lw=1.5,
                label=f"strong-PR (n={len(tau_post)})")
ax.set_xlabel("τ", fontsize=11)
ax.set_ylabel("1 − R²", fontsize=11)
ax.set_title("(a) Two equilibria at γ=(3,3,3)", fontsize=11)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="best")

ax = axs[1]
if fwd_gam:
    x = [r["gamma"] for r in fwd_gam]; y = [r["oneR2"] for r in fwd_gam]
    ax.loglog(x, y, "o-", color="#2ca02c", ms=5, lw=1.5)
ax.set_xlabel("γ", fontsize=11)
ax.set_ylabel("1 − R²", fontsize=11)
ax.set_title("(b) PR vs γ at τ=3", fontsize=11)
ax.grid(True, which="both", alpha=0.3)

ax = axs[2]
if asym:
    x = [r["tau"] for r in asym]; y = [r["oneR2"] for r in asym]
    ax.plot(x, y, "o-", color="#9467bd", ms=5, lw=1.5,
            label="γ=(5,3,1)")
if tau_pre:
    x = [r["tau"] for r in tau_pre]; y = [r["oneR2"] for r in tau_pre]
    ax.plot(x, y, "s-", color="#1f77b4", ms=4, lw=1, alpha=0.5,
            label="γ=(3,3,3) homo")
ax.set_xlabel("τ", fontsize=11)
ax.set_ylabel("1 − R²", fontsize=11)
ax.set_title("(c) Heterogeneity drives PR", fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("REZN without noise traders — CRRA partial revelation  "
             "(G=11 PCHIP, UMAX=2.5)", y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_FIG4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG4}")
