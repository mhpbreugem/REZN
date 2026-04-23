"""Driver for Python sweep at G in {5, 9, 13, 17}, CRRA gamma=2, tau=2.

Runs BOTH the pure-numpy implementation (rezn) and the numba-accelerated
one (rezn_numba). Reports Picard + Newton-LU timing, allocations,
peak RSS, and comparison-ready columns (p*, mu, ||F||, PR gap) at the
closest grid cell to (u1,u2,u3) = (1,-1,1).

Usage:
    python3 run_sweep.py numpy       # slow, skip G>=13
    python3 run_sweep.py numba       # fast, full sweep
"""
import sys
import time
import numpy as np
import resource
import gc

TAU = 2.0
GAMMA = 2.0
UMAX = 2.0
U_REPORT = (1.0, -1.0, 1.0)

mode = sys.argv[1] if len(sys.argv) > 1 else "numba"
if mode == "numba":
    import rezn_numba as rezn
elif mode == "numpy":
    import rezn
else:
    raise SystemExit(f"unknown mode: {mode}")

print(f"Python sweep mode: {mode}")
print(f"numpy version:     {np.__version__}")
print(f"CRRA γ={GAMMA}, τ={TAU}, target signal (u1,u2,u3)={U_REPORT}")


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_G(G, picard_iters=1200, newton_iters=5, do_newton=True):
    N = G ** 3
    jac_mb = N * N * 8 / (1024 ** 2)
    print(f"\n=== G={G:2d}  (N={N}, J ~ {jac_mb:.1f} MB dense)  mode={mode} ===")
    sys.stdout.flush()

    # Picard
    gc.collect()
    t0 = time.time()
    pic = rezn.solve_picard(G, TAU, GAMMA, umax=UMAX,
                            maxiters=picard_iters, abstol=1e-13)
    dt_pic = time.time() - t0
    u = pic["u"]
    ij = int(np.argmin(np.abs(u - U_REPORT[0])))
    jj = int(np.argmin(np.abs(u - U_REPORT[1])))
    lj = int(np.argmin(np.abs(u - U_REPORT[2])))
    p_pic = float(pic["P_star"][ij, jj, lj])
    mu_pic = rezn.posteriors_at(ij, jj, lj, p_pic, pic["P_star"], u, TAU)
    Finf_pic = float(np.abs(pic["residual"]).max())
    PhiI = pic["history"][-1] if pic["history"] else float("nan")
    rss_pic = peak_rss_mb()
    oneR2_pic = rezn.one_minus_R2(pic["P_star"], u, TAU) if hasattr(rezn, "one_minus_R2") else float("nan")
    print(f"  Picard     : iters={len(pic['history']):5d}   t={dt_pic:7.2f}s   peakRSS={rss_pic:6.0f} MB   ‖Φ-I‖∞={PhiI:.2e}   ‖F‖∞={Finf_pic:.2e}   1-R²={oneR2_pic:.4e}")
    print(f"              p*={p_pic:.10f}   μ=({mu_pic[0]:.6f}, {mu_pic[1]:.6f}, {mu_pic[2]:.6f})   PR_gap={mu_pic[0]-mu_pic[1]:.5f}")
    sys.stdout.flush()
    pic_row = (G, "Picard", len(pic["history"]), dt_pic, rss_pic, Finf_pic, p_pic, mu_pic, oneR2_pic)

    new_row = None
    if do_newton:
        gc.collect()
        t0 = time.time()
        new = rezn.solve_newton_lu(G, TAU, GAMMA, umax=UMAX,
                                   x0=pic["P_star"].reshape(-1).copy(),
                                   maxiters=newton_iters, abstol=1e-11)
        dt_new = time.time() - t0
        p_new = float(new["P_star"][ij, jj, lj])
        mu_new = rezn.posteriors_at(ij, jj, lj, p_new, new["P_star"], u, TAU)
        Finf_new = float(np.abs(new["residual"]).max())
        rss_new = peak_rss_mb()
        tm = new["timings"]
        oneR2_new = rezn.one_minus_R2(new["P_star"], u, TAU) if hasattr(rezn, "one_minus_R2") else float("nan")
        print(f"  Newton-LU  : iters={len(new['history'])-1:5d}   t={dt_new:7.2f}s   peakRSS={rss_new:6.0f} MB   ‖F‖∞={Finf_new:.2e}   1-R²={oneR2_new:.4e}")
        print(f"              breakdown: jac={tm['jac']:.2f}s  lu={tm['lu']:.2f}s  solve={tm['solve']:.2f}s  line-search={tm['ls']:.2f}s")
        print(f"              p*={p_new:.10f}   μ=({mu_new[0]:.6f}, {mu_new[1]:.6f}, {mu_new[2]:.6f})   PR_gap={mu_new[0]-mu_new[1]:.5f}")
        sys.stdout.flush()
        new_row = (G, "Newton-LU", len(new["history"]) - 1, dt_new, rss_new, Finf_new, p_new, mu_new, oneR2_new)

    return pic_row, new_row


def main():
    # Feasibility budget
    # numpy: skip Newton at G>=13, even Picard at G=17 may be too slow
    # numba: full sweep feasible
    if mode == "numpy":
        plan = [
            (5,  True),
            (9,  True),
            (13, False),     # numpy Newton at G=13 would be hours
        ]
    else:
        plan = [
            (5,  True),
            (9,  True),
            (13, True),
            (17, False),     # skip Newton at G=17
        ]

    # Warm-up numba first (JIT compile small case; don't count in timing)
    if mode == "numba":
        print("\n[numba warmup — JIT compiling at G=5, not timed]")
        sys.stdout.flush()
        _ = rezn.solve_picard(5, TAU, GAMMA, umax=UMAX, maxiters=3)
        # also compile newton path
        _ = rezn._fd_jacobian(np.zeros(125), rezn.build_grid(5), TAU, GAMMA, 1.0, 5, 1e-6)
        print("[warmup done]\n")
        sys.stdout.flush()

    rows = []
    for G, do_newton in plan:
        pic_row, new_row = run_G(G, do_newton=do_newton)
        rows.append(pic_row)
        if new_row is not None:
            rows.append(new_row)

    print("\n" + "=" * 127)
    print(f"  PYTHON / {mode.upper()}   CRRA γ={GAMMA}, τ={TAU}, target (1,-1,1)")
    print("=" * 127)
    print(f"{'G':>3} | {'method':<10} | {'iters':>6} | {'time(s)':>8} | {'peakRSS(MB)':>12} | {'‖F‖∞':>10} | {'1-R²':>10} | {'p*':>14} | {'μ₁':>11} | {'PR gap':>8}")
    print("-" * 140)
    for row in rows:
        G, method, iters, dt, rss, Finf, p, mu, oneR2 = row
        print(f"{G:>3} | {method:<10} | {iters:>6} | {dt:8.2f} | {rss:>12.0f} | {Finf:10.2e} | {oneR2:10.3e} | {p:14.10f} | {mu[0]:11.6f} | {mu[0]-mu[1]:8.5f}")
    print("-" * 127)


if __name__ == "__main__":
    main()
