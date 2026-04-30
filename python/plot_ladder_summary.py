"""Summary plot of all three ladders (G, γ, τ)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = (0.7, 0.11, 0.11)
BLUE = (0.0, 0.20, 0.42)
GREEN = (0.11, 0.35, 0.02)

with open("results/full_ree/posterior_v3_pava_full_ladder.json") as f:
    R = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

def plot_ladder(ax_top, ax_bot, ladder, key, label, ref_value=None):
    xs = [r[key] for r in ladder]
    r2s = [r["1-R^2"] for r in ladder]
    sl = [r["slope"] for r in ladder]
    meds = [r["phi_resid_med"] for r in ladder]
    ax_top.plot(xs, r2s, "o-", color=RED, lw=2, ms=8, label="1-R²")
    ax_top.set_xlabel(label, fontsize=11)
    ax_top.set_ylabel("1-R²", fontsize=11, color=RED)
    ax_top.tick_params(axis="y", labelcolor=RED)
    ax_top.grid(alpha=0.3)
    if ref_value is not None:
        ax_top.axhline(0.097, color=GREEN, ls=":", alpha=0.5,
                        label="G=14 PAVA-Cesaro reference")
    # Slope on right axis
    ax_top2 = ax_top.twinx()
    ax_top2.plot(xs, sl, "s--", color=BLUE, lw=1.5, ms=6, label="slope")
    ax_top2.set_ylabel("slope of logit(p) on T*", fontsize=11, color=BLUE)
    ax_top2.tick_params(axis="y", labelcolor=BLUE)
    ax_top2.axhline(1.0, color="black", ls=":", alpha=0.4, label="FR (slope=1)")
    ax_top2.axhline(1/3, color="gray", ls=":", alpha=0.4, label="no-learning ~ 1/3")

    # Bottom: median residual on log scale
    ax_bot.semilogy(xs, meds, "v-", color=GREEN, lw=2, ms=7)
    ax_bot.set_xlabel(label, fontsize=11)
    ax_bot.set_ylabel("median Φ-residual", fontsize=11)
    ax_bot.set_title(f"convergence quality vs {label}")
    ax_bot.grid(alpha=0.3, which="both")
    ax_bot.axhline(1e-3, color="black", ls=":", alpha=0.5, label="threshold 1e-3")
    ax_bot.legend(fontsize=8)


plot_ladder(axes[0, 0], axes[1, 0], R["G_ladder"], "G",
             "grid size G", ref_value=0.097)
axes[0, 0].set_title("(a) G-ladder at γ=0.5, τ=2", fontsize=11)
plot_ladder(axes[0, 1], axes[1, 1], R["gamma_ladder"], "gamma",
             "γ (CRRA)", ref_value=0.097)
axes[0, 1].set_xscale("log")
axes[1, 1].set_xscale("log")
axes[0, 1].set_title("(b) γ-ladder at G=14, τ=2", fontsize=11)
plot_ladder(axes[0, 2], axes[1, 2], R["tau_ladder"], "tau",
             "τ (signal precision)", ref_value=0.097)
axes[0, 2].set_xscale("log")
axes[1, 2].set_xscale("log")
axes[0, 2].set_title("(c) τ-ladder at G=14, γ=0.5", fontsize=11)

plt.suptitle("Posterior-method v3 PAVA-2D-Cesaro: 1-R², slope, residual",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("results/full_ree/ladder_summary.png", dpi=150, bbox_inches="tight")
plt.savefig("results/full_ree/ladder_summary.pdf", bbox_inches="tight")
print("Saved: results/full_ree/ladder_summary.png")
print("Saved: results/full_ree/ladder_summary.pdf")
