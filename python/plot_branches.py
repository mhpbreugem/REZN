"""Plot both τ-sweep branches at γ=(3,3,3): pre-jump + post-jump.

Reads pchip_continuation_results.csv (forward sweep — contains both
branches, distinguishable by 1-R² magnitude) and
pchip_backward_results.csv (backward sweep along post-jump branch
starting from τ=3.39-ish and walking down).

Produces a figure with:
  A. 1-R² vs τ, both branches overlaid.
  B. PR gap (μ1 − μ2) vs τ, both branches.
  C. p* at the reference cell (1,-1,1) vs τ, both branches.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FWD_CSV = "/home/user/REZN/python/pchip_continuation_results.csv"
BWD_CSV = "/home/user/REZN/python/pchip_backward_results.csv"
OUT_PNG = "/home/user/REZN/python/plot_branches.png"

JUMP_TAU    = 3.39       # empirical boundary where NK first lands on the other branch
JUMP_R2     = 5e-3       # 1-R² threshold: above this = post-jump, below = pre-jump


def load(csv_path):
    out = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                if int(r["converged"]) != 1:
                    continue
                if not (abs(float(r["gamma_1"])-3.0)<1e-9
                        and abs(float(r["gamma_2"])-3.0)<1e-9
                        and abs(float(r["gamma_3"])-3.0)<1e-9):
                    continue
                tau = float(r["tau_1"])
                oneR2 = float(r["oneR2_het"])
                pr = float(r["pr_gap"])
                p = float(r["p_star"])
                out.append((tau, oneR2, pr, p))
            except Exception:
                continue
    return out


def split(rows):
    pre = [r for r in rows if r[1] < JUMP_R2]
    post = [r for r in rows if r[1] >= JUMP_R2]
    pre.sort(); post.sort()
    return pre, post


def main():
    fwd = load(FWD_CSV)
    bwd = load(BWD_CSV)
    print(f"Forward rows (γ=3, converged): {len(fwd)}")
    print(f"Backward rows (γ=3, converged): {len(bwd)}")

    fwd_pre, fwd_post = split(fwd)
    bwd_pre, bwd_post = split(bwd)

    print(f"  forward pre-jump: {len(fwd_pre)}  post-jump: {len(fwd_post)}")
    print(f"  backward pre-jump: {len(bwd_pre)}  post-jump: {len(bwd_post)}")

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    def xy(rows, i):
        return [r[0] for r in rows], [r[i] for r in rows]

    # 1-R²
    ax = axs[0]
    if fwd_pre:
        x, y = xy(fwd_pre, 1)
        ax.semilogy(x, y, "o-", color="#1f77b4", ms=3, lw=1,
                    label=f"forward pre-jump (n={len(fwd_pre)})")
    if fwd_post:
        x, y = xy(fwd_post, 1)
        ax.semilogy(x, y, "s-", color="#d62728", ms=3, lw=1,
                    label=f"forward post-jump (n={len(fwd_post)})")
    if bwd_post:
        x, y = xy(bwd_post, 1)
        ax.semilogy(x, y, "^-", color="#ff7f0e", ms=3, lw=1,
                    label=f"backward post-jump (n={len(bwd_post)})")
    if bwd_pre:
        x, y = xy(bwd_pre, 1)
        ax.semilogy(x, y, "v-", color="#2ca02c", ms=3, lw=1,
                    label=f"backward onto pre-jump (n={len(bwd_pre)})")
    ax.axvline(JUMP_TAU, color="grey", ls=":", alpha=0.6,
               label=f"empirical jump τ={JUMP_TAU}")
    ax.set_ylabel("1 − R²_het  (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Two branches at γ=(3,3,3), G=11 PCHIP")

    # PR gap
    ax = axs[1]
    if fwd_pre:
        x, y = xy(fwd_pre, 2);  ax.plot(x, y, "o-", color="#1f77b4", ms=3, lw=1)
    if fwd_post:
        x, y = xy(fwd_post, 2); ax.plot(x, y, "s-", color="#d62728", ms=3, lw=1)
    if bwd_post:
        x, y = xy(bwd_post, 2); ax.plot(x, y, "^-", color="#ff7f0e", ms=3, lw=1)
    if bwd_pre:
        x, y = xy(bwd_pre, 2);  ax.plot(x, y, "v-", color="#2ca02c", ms=3, lw=1)
    ax.axvline(JUMP_TAU, color="grey", ls=":", alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylabel("PR gap  μ1 − μ2  at (1,-1,1)")
    ax.grid(True, alpha=0.3)

    # p*
    ax = axs[2]
    if fwd_pre:
        x, y = xy(fwd_pre, 3);  ax.plot(x, y, "o-", color="#1f77b4", ms=3, lw=1)
    if fwd_post:
        x, y = xy(fwd_post, 3); ax.plot(x, y, "s-", color="#d62728", ms=3, lw=1)
    if bwd_post:
        x, y = xy(bwd_post, 3); ax.plot(x, y, "^-", color="#ff7f0e", ms=3, lw=1)
    if bwd_pre:
        x, y = xy(bwd_pre, 3);  ax.plot(x, y, "v-", color="#2ca02c", ms=3, lw=1)
    ax.axvline(JUMP_TAU, color="grey", ls=":", alpha=0.6)
    ax.set_ylabel("p*  at (1,-1,1)")
    ax.set_xlabel("τ (homogeneous)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
