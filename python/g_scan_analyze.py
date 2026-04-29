"""Analyze the smooth G-scan: tabulate, fit h→0 extrapolation per G,
fit G→∞ extrapolation at small h. Saves plots to python/plots_gscan/."""

import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_ree"
PLOTS = REPO / "python" / "plots_gscan"
PLOTS.mkdir(exist_ok=True)


def load():
    """Load every G_*_smoothh*_het2_2_2_gscan_G*_h*_summary.json and
    organize by (G, h)."""
    rows = []
    for f in RESULTS.glob("G*_tau2_smoothh*_het2_2_2_gscan_*_summary.json"):
        with open(f) as fh:
            s = json.load(fh)
        # parse G and h from the filename: G6_tau2_smoothh0.05_het2_2_2_gscan_G6_h0.05_summary.json
        rows.append({
            "G": s["G"], "h": s["h"],
            "iters": s["iterations"],
            "resid": s["residual_inf"],
            "1-R2": s["revelation_deficit"],
            "slope": s["slope"],
            "max_FR": s["max_fr_error"],
            "price": s["representative_realization"]["price"],
            "fr_price": s["representative_realization"]["fr_price"],
        })
    return rows


def main():
    rows = load()
    if not rows:
        print("No G-scan summaries found")
        return

    # tabulate
    print("Loaded results:")
    rows.sort(key=lambda r: (r["G"], -r["h"]))
    print(f"{'G':>3} | {'h':>7} | {'1-R²':>11} | {'slope':>7} | {'max_FR':>8} | {'iters':>5}")
    print("-" * 65)
    for r in rows:
        print(f"{r['G']:>3} | {r['h']:>7g} | {r['1-R2']:>11.4e} | "
              f"{r['slope']:>7.4f} | {r['max_FR']:>8.4f} | {r['iters']:>5}")

    # h→0 extrapolation per G (linear in h^2)
    Gs = sorted({r["G"] for r in rows})
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Smooth Φ G-scan: γ=(2,2,2), τ=2, no-symm artifact baseline", fontsize=12)

    cmap = plt.get_cmap("viridis")
    G_extrap_R2 = {}
    G_extrap_slope = {}
    for i, G in enumerate(Gs):
        r_G = sorted([r for r in rows if r["G"] == G], key=lambda r: r["h"])
        h_arr = np.array([r["h"] for r in r_G])
        R2_arr = np.array([r["1-R2"] for r in r_G])
        slope_arr = np.array([r["slope"] for r in r_G])
        col = cmap(i / max(1, len(Gs) - 1))

        ax[0, 0].loglog(h_arr, R2_arr, "o-", label=f"G={G}", color=col)
        ax[0, 1].semilogx(h_arr, slope_arr, "o-", label=f"G={G}", color=col)

        # h→0 fit on smallest h points (linear in h²)
        if len(r_G) >= 3:
            h2 = h_arr[:3] ** 2
            a_R2 = np.polyfit(h2, R2_arr[:3], 1)
            a_sl = np.polyfit(h2, slope_arr[:3], 1)
            G_extrap_R2[G] = float(a_R2[1])
            G_extrap_slope[G] = float(a_sl[1])

    ax[0, 0].set_xlabel("h"); ax[0, 0].set_ylabel("1 - R²")
    ax[0, 0].set_title("Revelation deficit vs h (per G)")
    ax[0, 0].grid(True, alpha=0.3); ax[0, 0].legend()

    ax[0, 1].set_xlabel("h"); ax[0, 1].set_ylabel("slope")
    ax[0, 1].axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5, label="FR (slope=1)")
    ax[0, 1].set_title("Slope vs h (per G)")
    ax[0, 1].grid(True, alpha=0.3); ax[0, 1].legend()

    # G→∞ extrapolation at h→0 limits
    if G_extrap_R2:
        Gx = sorted(G_extrap_R2.keys())
        invG = np.array([1.0 / G for G in Gx])
        R2_x = np.array([G_extrap_R2[G] for G in Gx])
        sl_x = np.array([G_extrap_slope[G] for G in Gx])

        ax[1, 0].plot(invG, R2_x, "o-", color="C1")
        for G, ig, y in zip(Gx, invG, R2_x):
            ax[1, 0].annotate(f"G={G}", (ig, y), xytext=(5, 5), textcoords="offset points")
        ax[1, 0].set_xlabel("1/G"); ax[1, 0].set_ylabel("1 - R² @ h→0")
        ax[1, 0].set_title("Extrapolated 1-R² vs 1/G")
        ax[1, 0].grid(True, alpha=0.3)

        ax[1, 1].plot(invG, sl_x, "o-", color="C2")
        for G, ig, y in zip(Gx, invG, sl_x):
            ax[1, 1].annotate(f"G={G}", (ig, y), xytext=(5, 5), textcoords="offset points")
        ax[1, 1].axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5, label="FR")
        ax[1, 1].set_xlabel("1/G"); ax[1, 1].set_ylabel("slope @ h→0")
        ax[1, 1].set_title("Extrapolated slope vs 1/G")
        ax[1, 1].grid(True, alpha=0.3); ax[1, 1].legend()

    fig.tight_layout()
    out = PLOTS / "summary.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out}")

    # JSON dump
    with open(PLOTS / "summary.json", "w") as f:
        json.dump({
            "rows": rows,
            "h_to_zero_extrap": {
                str(G): {"1-R2": G_extrap_R2.get(G), "slope": G_extrap_slope.get(G)}
                for G in Gs
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
