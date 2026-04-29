"""Plot G=6 vs G=12 γ-ladder comparison."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
PLOTS = REPO / "python" / "plots_gscan"

with open(RESULTS / "smooth_gamma_ladder_G6_h0.005.json") as f:
    d6 = json.load(f)
with open(RESULTS / "G12_smooth_gamma_ladder_h0.005.json") as f:
    d12 = json.load(f)

# G=6 has separate down/up; merge and dedup on γ
g6 = []
for r in d6["down"] + d6["up"]:
    if r["gamma"] not in [x["gamma"] for x in g6]:
        g6.append(r)
g6.sort(key=lambda r: r["gamma"])

g12 = sorted(d12["rows"], key=lambda r: r["gamma"])

g6_g = np.array([r["gamma"] for r in g6])
g6_R2 = np.array([r["1-R2"] for r in g6])
g6_sl = np.array([r["slope"] for r in g6])

g12_g = np.array([r["gamma"] for r in g12])
g12_R2 = np.array([r["revelation_deficit"] for r in g12])
g12_sl = np.array([r["slope"] for r in g12])

# baseline = γ=20
g6_base = g6_R2[g6_g == 20.0][0]
g12_base = g12_R2[g12_g == 20.0][0]
g6_NET = g6_R2 - g6_base
g12_NET = g12_R2 - g12_base

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("γ ladder: G=6 vs G=12 with smooth Φ (h=0.005, τ=2). All converged to ~1e-13.",
             fontsize=12)

ax = axes[0]
ax.semilogx(g6_g, g6_R2, "o-", color="C0", label="G=6", linewidth=2)
ax.semilogx(g12_g, g12_R2, "s-", color="C1", label="G=12", linewidth=2)
ax.axhline(g6_base, color="C0", ls=":", alpha=0.5)
ax.axhline(g12_base, color="C1", ls=":", alpha=0.5)
ax.set_xlabel("γ"); ax.set_ylabel("1 − R²")
ax.set_title("Revelation deficit vs γ")
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[1]
ax.semilogx(g6_g, g6_sl, "o-", color="C0", label="G=6", linewidth=2)
ax.semilogx(g12_g, g12_sl, "s-", color="C1", label="G=12", linewidth=2)
ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR (slope=1)")
ax.set_xlabel("γ"); ax.set_ylabel("regression slope")
ax.set_title("Slope vs γ")
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[2]
ax.semilogx(g6_g, g6_NET, "o-", color="C0", label="G=6", linewidth=2)
ax.semilogx(g12_g, g12_NET, "s-", color="C1", label="G=12", linewidth=2)
ax.axhline(0, color="k", lw=0.7, alpha=0.6)
ax.set_xlabel("γ"); ax.set_ylabel("NET 1 − R² (γ minus γ=20)")
ax.set_title("Genuine PR signal (artifact-subtracted)")
ax.grid(True, alpha=0.3); ax.legend()

fig.tight_layout()
out = PLOTS / "gamma_ladder_G6_vs_G12.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# Summary
print("\nSummary:")
print(f"{'γ':>6} | {'G=6 NET':>10} | {'G=12 NET':>10} | {'ratio':>6}")
for gam in [0.1, 0.25, 0.5, 1.0, 2.0]:
    g6r = g6_NET[g6_g == gam]
    g12r = g12_NET[g12_g == gam]
    if len(g6r) and len(g12r):
        ratio = g12r[0] / g6r[0] if g6r[0] != 0 else float("inf")
        print(f"{gam:>6g} | {g6r[0]:>+10.6f} | {g12r[0]:>+10.6f} | {ratio:>6.2f}")
