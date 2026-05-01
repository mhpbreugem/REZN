"""Redo Fig 6A: CARA posteriors at high resolution.

CARA + no-learning (analytical, no solver):
  μ_high = Λ(τ·u_high)  (constant for fixed u_high)
  μ_low  = Λ(τ·u_low)
  p_NL   = Λ(T*/3)      (CARA aggregates in logit space, K=3)

Same format as Fig 6B for visual comparison.
"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from posterior_method_v3 import Lam

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS_DIR = "results/full_ree"
TAU = 2.0
RED = (0.7, 0.11, 0.11); BLUE = (0.0, 0.20, 0.42); GREEN = (0.11, 0.35, 0.02)

T_grid = np.linspace(-12, 12, 200)
u_high = +1.0; u_low = -1.0
mu_high = Lam(TAU * u_high)
mu_low = Lam(TAU * u_low)
p_NL = Lam(T_grid / 3.0)

# JSON
fig6A_data = {
    "figure": "fig_posteriors_CARA_no_learning",
    "params": {"tau": TAU, "u_high": u_high, "u_low": u_low,
                "n_points": len(T_grid)},
    "agent_high_constant": float(mu_high),
    "agent_low_constant": float(mu_low),
    "price_no_learning": [{"T_star": float(t), "p": float(p)}
                           for t, p in zip(T_grid, p_NL)],
}
with open(f"{RESULTS_DIR}/fig_posteriors_CARA_data.json", "w") as f:
    json.dump(fig6A_data, f, indent=2)


def pgf(pts, fmt="{:.6g}"):
    return "".join(f"({fmt.format(x)},{fmt.format(y)})" for x, y in pts)


with open(f"{RESULTS_DIR}/fig_posteriors_CARA_pgfplots.tex", "w") as f:
    f.write("% mu_high = Λ(τ·u_high) = constant (no learning)\n")
    f.write(f"\\addplot coordinates {{({T_grid[0]:.3f},{mu_high:.6g})"
            f"({T_grid[-1]:.3f},{mu_high:.6g})}};\n\n")
    f.write("% mu_low = Λ(τ·u_low) = constant (no learning)\n")
    f.write(f"\\addplot coordinates {{({T_grid[0]:.3f},{mu_low:.6g})"
            f"({T_grid[-1]:.3f},{mu_low:.6g})}};\n\n")
    f.write("% p_NL = Λ(T*/3) — no-learning CARA price\n")
    pts = list(zip(T_grid.tolist(), p_NL.tolist()))
    f.write(f"\\addplot coordinates {{{pgf(pts)}}};\n")
print("Saved fig_posteriors_CARA_*.{json,tex}", flush=True)

# Render preview
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
ax.axhline(mu_high, color=RED, lw=2,
             label=f"$\\mu_1 = \\Lambda(\\tau\\cdot{u_high:+g})={mu_high:.3f}$")
ax.axhline(mu_low, color=GREEN, lw=2,
             label=f"$\\mu_2 = \\Lambda(\\tau\\cdot{u_low:+g})={mu_low:.3f}$")
ax.plot(T_grid, p_NL, color=BLUE, ls="--", lw=2,
          label="price $p = \\Lambda(T^*/3)$")
ax.set_xlabel("$T^*$"); ax.set_ylabel("$\\mu$, $p$")
ax.set_xlim(-12, 12); ax.set_ylim(0, 1)
ax.set_title(f"Fig 6A: CARA posteriors (no learning)\n"
              f"$\\tau={TAU}$, analytical, $n={len(T_grid)}$",
              fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CARA_preview.pdf",
              bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/fig_posteriors_CARA_preview.png",
              dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved preview", flush=True)
