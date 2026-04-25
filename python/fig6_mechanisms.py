"""Figure 6 — Three Mechanisms (§6).

Six configurations isolate the distinct sources of partial revelation
that emerge once preferences leave the CARA knife-edge:

  1. baseline      γ=(0.5,0.5,0.5), τ=(2,2,2)        pure Jensen gap
  2. het γ          γ=(1,3,10),     τ=(2,2,2)        + heterogeneous γ
  3. het τ          γ=(2,2,2),      τ=(1,3,10)       + heterogeneous τ
  4. aligned       γ=(10,3,1),     τ=(1,3,10)       low-γ matched with low-τ
  5. opposed       γ=(1,3,10),     τ=(1,3,10)       low-γ matched with high-τ
  6. extreme       γ=(0.3,3,30),   τ=(0.3,3,30)     aggressive uninformed

For each row we solve the REE with the production PCHIP+contour kernel
(Anderson, m=6, abstol=1e-7, G=11, UMAX=2.5) and report 1−R² of
logit(P) against the nominal CARA sufficient statistic
T*_nominal = Σ τ̄ u_k where τ̄ is the mean signal precision.

Caveat: production-method bias (~1e-3, see diag_gh_contour_results.md)
means values at the small-γ end may shift slightly under the GH-contour
upgrade; the ordering of mechanisms is preserved.

Outputs:
  figures/fig6_mechanisms.csv  — table
  figures/fig6_mechanisms.{png,pdf,tex}  — horizontal bar chart, B&W
"""
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import rezn_het as rh
import rezn_pchip as rp
from fig_export import save_png_pdf_tex


G        = 11
UMAX     = 2.0
ABSTOL   = 1e-7
PICARD_ITERS  = 600
ANDERSON_ITERS = 800
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


# Six configurations, mild ranges (γ ≤ 3, τ ≤ 3) so the production
# kernel converges cleanly at G=11. Qualitative ordering of mechanisms
# matches the wider-range table from the paper (project_summary.txt §6).
# (label, taus, gammas, short-id)
CONFIGS = [
    ("baseline (equal γ=0.5, equal τ=2)",
        np.array([2.0, 2.0, 2.0]),
        np.array([0.5, 0.5, 0.5]),
        "baseline"),
    ("het γ=(0.5,1,3), equal τ=2",
        np.array([2.0, 2.0, 2.0]),
        np.array([0.5, 1.0, 3.0]),
        "het_gamma"),
    ("equal γ=2, het τ=(1,2,3)",
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 2.0, 2.0]),
        "het_tau"),
    ("het γ + het τ aligned   (γ=(3,1,0.5), τ=(1,2,3))",
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 1.0, 0.5]),
        "aligned"),
    ("het γ + het τ opposed   (γ=(0.5,1,3), τ=(1,2,3))",
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.0, 3.0]),
        "opposed"),
    ("extreme opposed         (γ=(0.3,1,3), τ=(0.5,2,3))",
        np.array([0.5, 2.0, 3.0]),
        np.array([0.3, 1.0, 3.0]),
        "extreme"),
]


def _solve(taus, gammas):
    """Stable solver chain for stiff heterogeneous configurations:
    Picard with α=0.3 first (handles wild iterates), then Anderson
    polish from the Picard best iterate."""
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=PICARD_ITERS, abstol=ABSTOL,
                                    alpha=0.3)
    if not np.isfinite(res_p["history"][-1]):
        return res_p
    res_a = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                      maxiters=ANDERSON_ITERS, abstol=ABSTOL,
                                      m_window=8, P_init=res_p["P_star"])
    # Return whichever produced the smaller residual
    finf_p = float(np.abs(res_p["residual"]).max())
    finf_a = float(np.abs(res_a["residual"]).max())
    return res_a if finf_a <= finf_p else res_p


def main():
    os.makedirs(OUT, exist_ok=True)
    u = np.linspace(-UMAX, UMAX, G)

    rows = []
    for label, taus, gammas, sid in CONFIGS:
        print(f"solving {sid}: τ={tuple(taus)} γ={tuple(gammas)}",
              flush=True)
        res = _solve(taus, gammas)
        P_star = res["P_star"]
        finf = float(np.abs(res["residual"]).max())
        # 1-R² against the per-agent T* = Σ τ_k u_k (CARA sufficient statistic)
        oneR2 = rh.one_minus_R2(P_star, u, taus)
        rows.append({
            "id": sid, "label": label,
            "tau": ",".join(f"{t:g}" for t in taus),
            "gamma": ",".join(f"{g:g}" for g in gammas),
            "Finf": finf,
            "oneR2": oneR2,
        })
        print(f"  iters={len(res['history'])}  Finf={finf:.2e}  "
              f"1−R²={oneR2:.4f}", flush=True)

    csv_path = os.path.join(OUT, "fig6_mechanisms.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [r["label"] for r in rows]
    vals   = [r["oneR2"] for r in rows]
    y_pos  = np.arange(len(rows))[::-1]
    # B&W: shade ∝ 1−R², lightest at top (baseline), darkest at bottom (extreme)
    norm = (np.array(vals) - min(vals)) / max(1e-12, max(vals) - min(vals))
    shades = 0.93 - 0.78 * norm
    bars = ax.barh(y_pos, vals, color=[(s, s, s) for s in shades],
                    edgecolor="black", linewidth=0.8, height=0.72)
    for y, v in zip(y_pos, vals):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center",
                 fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$1 - R^2$ of logit$(p)$ vs $T^*$")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title(r"Three mechanisms for partial revelation"
                  f"  (CRRA, $G={G}$, $u\\in[-{UMAX},{UMAX}]$)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_png_pdf_tex(fig, os.path.join(OUT, "fig6_mechanisms"))


if __name__ == "__main__":
    main()
