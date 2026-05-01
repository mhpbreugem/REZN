"""Render PDF/PNG previews of the completed figure data files.

BC20 style: red (0.7, 0.11, 0.11), blue (0.0, 0.20, 0.42),
green (0.11, 0.35, 0.02). 8cm × 8cm axes.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/full_ree"
RED = (0.7, 0.11, 0.11)
BLUE = (0.0, 0.20, 0.42)
GREEN = (0.11, 0.35, 0.02)
BLACK = "black"


def save_fig(fig, name):
    fig.savefig(f"{RESULTS_DIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{RESULTS_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/{name}.{{pdf,png}}", flush=True)


# ===== Fig 1: knife-edge =====
def render_knife_edge():
    with open(f"{RESULTS_DIR}/fig_knife_edge_data.json") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    colors = [GREEN, RED, BLUE]
    styles = ["-", "--", ":"]
    for c, color, style in zip(d["curves"], colors, styles):
        taus = [p["tau"] for p in c["points"]]
        r2s = [p["1-R2"] for p in c["points"]]
        ax.plot(taus, r2s, color=color, ls=style, lw=2,
                  label=f"$\\gamma={c['gamma']}$")
    ax.set_xscale("log")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$1-R^2$")
    ax.set_title("Fig 1: knife-edge (no-learning)\n"
                  f"G={d['params']['G']}", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "fig_knife_edge_preview")


# ===== Fig 5: REE vs NL =====
def render_fig5():
    with open(f"{RESULTS_DIR}/fig_ree_vs_nolearning_data.json") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    Ts = [p["T_star"] for p in d["REE"]]
    p_REE = [p["p"] for p in d["REE"]]
    p_NL = [p["p"] for p in d["no_learning"]]
    p_FR = [p["p"] for p in d["FR"]]
    ax.plot(Ts, p_REE, color=RED, lw=2, label="REE (CRRA)")
    ax.plot(Ts, p_NL, color=BLUE, ls="--", lw=2, label="no learning")
    ax.plot(Ts, p_FR, color=BLACK, ls=":", lw=1.5, label="$p=\\Lambda(T^*)$ (FR)")
    ax.set_xlabel("$T^* = \\tau \\sum_k u_k$")
    ax.set_ylabel("price $p$")
    ax.set_xlim(min(Ts), max(Ts)); ax.set_ylim(0, 1)
    ax.set_title(f"Fig 5: REE vs no-learning\n"
                  f"$G={d['params']['G']}$, $\\tau={d['params']['tau']}$, "
                  f"$\\gamma={d['params']['gamma']}$", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "fig_ree_vs_nolearning_preview")


# ===== Fig 6A: CARA analytical =====
def render_fig6A():
    with open(f"{RESULTS_DIR}/fig_posteriors_CARA_data.json") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    Ts = [p["T_star"] for p in d["points"]]
    mus = [p["mu"] for p in d["points"]]
    ax.plot(Ts, mus, color=BLACK, lw=2, label="$\\mu = \\Lambda(T^*/3)$")
    ax.set_xlabel("$T^*$"); ax.set_ylabel("$\\mu$")
    ax.set_xlim(min(Ts), max(Ts)); ax.set_ylim(0, 1)
    ax.set_title("Fig 6A: CARA posterior (analytical)", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "fig_posteriors_CARA_preview")


# ===== Fig 6B: CRRA fan-out =====
def render_fig6B():
    with open(f"{RESULTS_DIR}/fig_posteriors_CRRA_data.json") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    Ts = [p["T_star"] for p in d["agent_high"]]
    mu_high = [p["mu"] for p in d["agent_high"]]
    mu_low = [p["mu"] for p in d["agent_low"]]
    Ts_p = [p["T_star"] for p in d["price"]]
    prices = [p["p"] for p in d["price"]]
    ax.plot(Ts, mu_high, color=RED, lw=2, label=f"$\\mu_1$ ($u=+1$)")
    ax.plot(Ts, mu_low, color=GREEN, lw=2, label=f"$\\mu_2$ ($u=-1$)")
    ax.plot(Ts_p, prices, color=BLUE, ls="--", lw=2, label="price $p$")
    ax.set_xlabel("$T^*$"); ax.set_ylabel("$\\mu$, $p$")
    ax.set_xlim(min(Ts), max(Ts)); ax.set_ylim(0, 1)
    ax.set_title(f"Fig 6B: CRRA posteriors (REE)\n"
                  f"$G={d['params']['G']}$, $\\tau={d['params']['tau']}$, "
                  f"$\\gamma={d['params']['gamma']}$", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "fig_posteriors_CRRA_preview")


# ===== Fig 10: convergence =====
def render_fig10():
    with open(f"{RESULTS_DIR}/fig_convergence_data.json") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    p1 = [(h["iter"], max(h["residual_inf"], 1e-20))
           for h in d["history"] if h["phase"] == "picard1"]
    p2 = [(h["iter"], max(h["residual_inf"], 1e-20))
           for h in d["history"] if h["phase"] == "picard2"]
    nk = [(h["iter"], max(h["residual_inf"], 1e-20))
           for h in d["history"] if h["phase"] == "NK"]
    if p1:
        xs, ys = zip(*p1)
        ax.semilogy(xs, ys, color=GREEN, lw=1.5,
                      label=f"Picard ($\\alpha=0.05$)")
    if p2:
        xs, ys = zip(*p2)
        ax.semilogy(xs, ys, color=RED, lw=1.5,
                      label=f"Picard ($\\alpha=0.01$, Cesaro)")
    if nk:
        xs, ys = zip(*nk)
        ax.semilogy(xs, ys, color=BLUE, marker="o", ls="-", lw=1.5,
                      ms=4, label="Newton-Krylov")
    ax.axhline(1e-14, color=BLACK, ls=":", alpha=0.5,
                 label="strict tol $10^{-14}$")
    ax.set_xlabel("iteration"); ax.set_ylabel("$\\|\\Phi(\\mu)-\\mu\\|_\\infty$")
    ax.set_title(f"Fig 10: convergence\n"
                  f"$G={d['params']['G']}$, $\\tau={d['params']['tau']}$, "
                  f"$\\gamma={d['params']['gamma']}$", fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    save_fig(fig, "fig_convergence_preview")


for name, fn in [("Fig 1", render_knife_edge),
                  ("Fig 5", render_fig5),
                  ("Fig 6A", render_fig6A),
                  ("Fig 6B", render_fig6B),
                  ("Fig 10", render_fig10)]:
    try:
        print(f"Rendering {name}...", flush=True)
        fn()
    except FileNotFoundError as e:
        print(f"  Skipping {name}: {e}", flush=True)
