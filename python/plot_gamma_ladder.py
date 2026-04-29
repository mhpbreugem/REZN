"""Plot the γ-ladder at G=6, h=0.005 (smooth method)."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "python" / "plots_gscan"
PLOTS.mkdir(exist_ok=True)

with open(REPO / "results/full_ree/smooth_gamma_ladder_G6_h0.005.json") as f:
    data = json.load(f)

down = data["down"]; up = data["up"]
# Combine, dedup on gamma
seen = set()
rows = []
for r in down + up:
    if r["gamma"] in seen:
        continue
    seen.add(r["gamma"])
    rows.append(r)
rows.sort(key=lambda r: r["gamma"])

g = np.array([r["gamma"] for r in rows])
R2 = np.array([r["1-R2"] for r in rows])
sl = np.array([r["slope"] for r in rows])

# CARA baseline ≈ value at largest γ (γ=20)
R2_base = R2[np.argmax(g)]
sl_base = sl[np.argmax(g)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"γ-ladder at G=6, τ=2, smooth Φ (h=0.005). All residuals ≤ 1e-13.",
             fontsize=12)

ax = axes[0]
ax.semilogx(g, R2, "o-", color="C0")
ax.axhline(R2_base, color="grey", lw=0.8, ls="--", alpha=0.6,
           label=f"γ=20 baseline (artifact floor)\n= {R2_base:.5f}")
ax.set_xlabel("γ"); ax.set_ylabel("1 - R²")
ax.set_title("Revelation deficit vs γ")
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[1]
ax.semilogx(g, sl, "o-", color="C2")
ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="FR (slope=1)")
ax.axhline(sl_base, color="grey", lw=0.8, ls=":", alpha=0.6,
           label=f"γ=20 baseline\n= {sl_base:.4f}")
ax.set_xlabel("γ"); ax.set_ylabel("regression slope")
ax.set_title("Slope vs γ")
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[2]
NET = R2 - R2_base
ax.semilogx(g, NET, "o-", color="C1")
ax.axhline(0, color="k", lw=0.7, alpha=0.6)
ax.set_xlabel("γ"); ax.set_ylabel("NET 1 - R² (γ minus γ=20)")
ax.set_title("Genuine PR signal (artifact-subtracted)")
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = PLOTS / "gamma_ladder_G6_h0.005.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

print("\nNET PR table:")
print(f"{'γ':>6} | {'1-R²':>11} | {'NET (1-R²)':>12} | {'slope':>7} | {'(1-slope)':>10}")
for gam, r2_v, sl_v, n in zip(g, R2, sl, NET):
    print(f"{gam:>6g} | {r2_v:>11.6f} | {n:>+12.6f} | {sl_v:>7.4f} | {1-sl_v:>10.4f}")
