"""Plot 1-R² as a function of γ for the homogeneous γ sweep."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/user/REZN/python/pchip_continuation_results.csv"
OUT_GAMMA = "/home/user/REZN/python/plot_1mR2_vs_gamma.png"
OUT_TAU   = "/home/user/REZN/python/plot_1mR2_vs_tau.png"
OUT_BOTH  = "/home/user/REZN/python/plot_1mR2_both.png"

def collect():
    """Return dict: 'gamma' → sorted (γ, 1-R²) homo γ at τ=3;
                    'tau'   → sorted (τ, 1-R²) homo τ at γ=3."""
    gamma_rows = []
    tau_rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["converged"] != "1":
                continue
            tau = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
            gam = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
            if not (tau[0] == tau[1] == tau[2]
                    and gam[0] == gam[1] == gam[2]):
                continue
            one_het = float(r["oneR2_het"])
            if abs(tau[0] - 3.0) < 1e-9:
                gamma_rows.append((gam[0], one_het))
            if abs(gam[0] - 3.0) < 1e-9:
                tau_rows.append((tau[0], one_het))
    gamma_rows.sort(key=lambda x: x[0])
    tau_rows.sort(key=lambda x: x[0])
    return gamma_rows, tau_rows


def main():
    gamma_rows, tau_rows = collect()
    print(f"homo γ sweep: {len(gamma_rows)} pts, γ ∈ [{gamma_rows[0][0]}, {gamma_rows[-1][0]}]")
    print(f"homo τ sweep: {len(tau_rows)} pts, τ ∈ [{tau_rows[0][0]}, {tau_rows[-1][0]}]")

    # γ sweep solo
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    xs, ys = zip(*gamma_rows)
    axs[0].plot(xs, ys, "o-", color="#1f77b4")
    axs[0].set_xlabel("γ (homogeneous)")
    axs[0].set_ylabel("1 − R²_het")
    axs[0].set_title(f"PR vs γ  (τ=3, G=7, PCHIP)  n={len(xs)}")
    axs[0].grid(True, alpha=0.4)
    axs[1].loglog(xs, ys, "o-", color="#d62728")
    axs[1].set_xlabel("γ")
    axs[1].set_ylabel("1 − R²_het")
    axs[1].set_title("Same, log-log")
    axs[1].grid(True, which="both", alpha=0.4)
    fig.suptitle("Homogeneous γ sweep, τ=(3,3,3), G=7", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_GAMMA, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_GAMMA}")

    # τ sweep solo
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    xs, ys = zip(*tau_rows)
    axs[0].plot(xs, ys, "s-", color="#2ca02c")
    axs[0].set_xlabel("τ (homogeneous)")
    axs[0].set_ylabel("1 − R²_het")
    axs[0].set_title(f"PR vs τ  (γ=3, G=7, PCHIP)  n={len(xs)}")
    axs[0].grid(True, alpha=0.4)
    axs[1].loglog(xs, ys, "s-", color="#9467bd")
    axs[1].set_xlabel("τ")
    axs[1].set_ylabel("1 − R²_het")
    axs[1].set_title("Same, log-log")
    axs[1].grid(True, which="both", alpha=0.4)
    fig.suptitle("Homogeneous τ sweep, γ=(3,3,3), G=7", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_TAU, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_TAU}")

    # Combined log-log
    fig, ax = plt.subplots(figsize=(7, 5))
    g_x, g_y = zip(*gamma_rows)
    t_x, t_y = zip(*tau_rows)
    ax.loglog(g_x, g_y, "o-", color="#1f77b4",
              label=f"γ sweep (τ=3), n={len(g_x)}")
    ax.loglog(t_x, t_y, "s-", color="#2ca02c",
              label=f"τ sweep (γ=3), n={len(t_x)}")
    ax.set_xlabel("swept parameter value")
    ax.set_ylabel("1 − R²_het")
    ax.set_title("PR strength: homogeneous sweeps at G=7 (PCHIP)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_BOTH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_BOTH}")


if __name__ == "__main__":
    main()
