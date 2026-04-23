"""Plot 1-R² as a function of γ for the homogeneous γ sweep."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/user/REZN/python/pchip_continuation_results.csv"
OUT = "/home/user/REZN/python/plot_1mR2_vs_gamma.png"

def main():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["converged"] != "1":
                continue
            tau = (float(r["tau_1"]), float(r["tau_2"]), float(r["tau_3"]))
            gam = (float(r["gamma_1"]), float(r["gamma_2"]), float(r["gamma_3"]))
            # homogeneous: all three equal
            if not (tau[0] == tau[1] == tau[2]
                    and gam[0] == gam[1] == gam[2]):
                continue
            # γ sweep: τ == 3
            if abs(tau[0] - 3.0) > 1e-9:
                continue
            rows.append((gam[0], float(r["oneR2_het"])))

    rows.sort(key=lambda x: x[0])
    if not rows:
        print("No converged homogeneous γ-sweep rows in CSV.")
        return
    gammas = [r[0] for r in rows]
    oneR2  = [r[1] for r in rows]

    print(f"{len(rows)} converged points plotted.")
    print("γ range:", min(gammas), "..", max(gammas))
    print("1-R² range:", min(oneR2), "..", max(oneR2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(gammas, oneR2, "o-", color="#1f77b4")
    axs[0].set_xlabel("γ (homogeneous)")
    axs[0].set_ylabel("1 − R²_het")
    axs[0].set_title("PR strength vs γ  (τ=3 homogeneous, G=7, PCHIP)")
    axs[0].grid(True, alpha=0.4)

    axs[1].loglog(gammas, oneR2, "o-", color="#d62728")
    axs[1].set_xlabel("γ (log scale)")
    axs[1].set_ylabel("1 − R²_het (log scale)")
    axs[1].set_title("Same, log-log")
    axs[1].grid(True, which="both", alpha=0.4)

    fig.suptitle("Homogeneous γ sweep at τ=(3,3,3), G=7", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
