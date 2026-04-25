"""Figure 5 — Convergence Paths (Appendix B).

‖P − Φ(P)‖∞ vs iteration for Picard (damped, α=0.3) and Anderson (m=6)
on the same configuration. CRRA γ=0.5, τ=2, G=15. Both runs warm-start
from the no-learning equilibrium.

Key visual: Anderson reaches the precision floor in ~6 iterations;
Picard takes 30+. The floor itself is set by the production
PCHIP-grid-edge contour discretisation (~1e-7 at G=15).

Output:
  plots/fig5_convergence.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import rezn_pchip as rp


G       = 11
TAU     = 2.0
GAMMA   = 0.5
UMAX    = 2.0
N_ITERS = 300
OUT     = os.path.join(os.path.dirname(__file__), "plots")


def main():
    os.makedirs(OUT, exist_ok=True)
    taus   = np.array([TAU, TAU, TAU])
    gammas = np.array([GAMMA, GAMMA, GAMMA])

    print("Picard (α=0.3)…", flush=True)
    res_p = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                    maxiters=N_ITERS, abstol=1e-15,
                                    alpha=0.3)
    print(f"  iters={len(res_p['history'])}  "
          f"final ‖Φ-I‖∞ = {res_p['history'][-1]:.3e}", flush=True)

    print("Anderson (m=6)…", flush=True)
    res_a = rp.solve_anderson_pchip(G, taus, gammas, umax=UMAX,
                                      maxiters=N_ITERS, abstol=1e-15,
                                      m_window=6)
    print(f"  iters={len(res_a['history'])}  "
          f"final ‖Φ-I‖∞ = {res_a['history'][-1]:.3e}", flush=True)

    h_p = np.asarray(res_p["history"])
    h_a = np.asarray(res_a["history"])
    env_p = np.minimum.accumulate(h_p)
    env_a = np.minimum.accumulate(h_a)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Raw histories, faint grey
    ax.semilogy(h_p, lw=0.6, color="0.5", alpha=0.5)
    ax.semilogy(h_a, lw=0.6, color="0.5", alpha=0.5)
    # Best-so-far envelopes: solid (Anderson) vs dashed (Picard); both black.
    ax.semilogy(env_p, lw=1.8, color="black", linestyle="--",
                 label=r"Picard, $\alpha=0.3$")
    ax.semilogy(env_a, lw=1.8, color="black", linestyle="-",
                 label=r"Anderson, $m=6$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|P - \Phi(P)\|_\infty$")
    ax.set_title(rf"Convergence at $\gamma={GAMMA}$, $\tau={TAU}$, $G={G}$"
                  "\n(faint grey: raw residual; bold: best-so-far envelope)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    png = os.path.join(OUT, "fig5_convergence.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
