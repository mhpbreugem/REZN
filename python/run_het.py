"""Heterogeneous-preferences CRRA sweep (numba).

Three configurations at G=9, matching the 'Three Mechanisms for PR'
row of project_summary.txt:

  (1) Homogeneous γ=0.5, τ=2  (pure Jensen gap baseline)
  (2) Het γ=(1,3,10), τ=2 for all  (+ risk-aversion heterogeneity)
  (3) Het γ=(1,3,10) and het τ=(1,3,10)  (aligned: low-γ = high-τ)

Runs Picard with damped α and enough iterations so ‖Φ-I‖∞ < 1e-10 or
we clearly see the floor, reports p*, μ per agent, 1-R², ‖F‖∞, PR gap.
Comments on when PR coexists with FR.
"""
import sys, time
import numpy as np
import rezn_het as rh

G = 9
UMAX = 2.0
U_REPORT = (1.0, -1.0, 1.0)


def run_case(tag, taus, gammas, maxiters=8000,
             alphas=(1.0, 0.3, 0.1)):
    """Run Picard at a ladder of damping values; keep the most converged."""
    print("\n" + "=" * 90)
    print(f"  {tag}")
    print(f"  taus   = {taus}")
    print(f"  gammas = {gammas}")
    print("=" * 90)

    best = None
    for alpha in alphas:
        t0 = time.time()
        res = rh.solve_picard(G, taus, gammas, umax=UMAX,
                              maxiters=maxiters, abstol=1e-13, alpha=alpha)
        dt = time.time() - t0
        Finf = float(np.abs(res["residual"]).max())
        PhiI = res["history"][-1] if res["history"] else float("nan")
        print(f"  [α={alpha:.2f}]  iters={len(res['history'])}   t={dt:.1f}s   ‖Φ-I‖∞={PhiI:.2e}   ‖F‖∞={Finf:.2e}   converged={res['converged']}")
        sys.stdout.flush()
        cand = (Finf, PhiI, dt, res, alpha)
        if best is None or cand[0] < best[0]:
            best = cand
        # if this alpha already converged fully, stop
        if PhiI < 1e-10:
            break
    Finf, PhiI, dt, res, alpha = best
    print(f"  chosen α={alpha:.2f} (lowest ‖F‖∞)")

    u = res["u"]
    taus_vec = res["taus"]
    gammas_vec = res["gammas"]

    idx = lambda x: int(np.argmin(np.abs(u - x)))
    i, j, l = idx(U_REPORT[0]), idx(U_REPORT[1]), idx(U_REPORT[2])
    p_star = float(res["P_star"][i, j, l])
    mus = rh.posteriors_at(i, j, l, p_star, res["P_star"], u, taus_vec)
    Finf = float(np.abs(res["residual"]).max())
    PhiI = res["history"][-1] if res["history"] else float("nan")
    oneR2 = rh.one_minus_R2(res["P_star"], u, taus_vec)

    print(f"  Picard: iters={len(res['history'])}   time={dt:.2f}s   ‖Φ-I‖∞={PhiI:.2e}   ‖F‖∞={Finf:.2e}")
    print(f"          converged={res['converged']}")
    print(f"          1-R² (using T* = Σ τ_k u_k) = {oneR2:.4e}")
    print(f"          p* at (1,-1,1) = {p_star:.10f}")
    print(f"          μ_1 (u=+1, τ_1={taus_vec[0]:.2g}, γ_1={gammas_vec[0]:.2g}) = {mus[0]:.6f}")
    print(f"          μ_2 (u=-1, τ_2={taus_vec[1]:.2g}, γ_2={gammas_vec[1]:.2g}) = {mus[1]:.6f}")
    print(f"          μ_3 (u=+1, τ_3={taus_vec[2]:.2g}, γ_3={gammas_vec[2]:.2g}) = {mus[2]:.6f}")
    print(f"          PR gap μ_1 - μ_2 = {mus[0]-mus[1]:.5f}")
    print(f"          max|μ - p|       = {max(abs(mus[k]-p_star) for k in range(3)):.5f}")
    sys.stdout.flush()
    return dict(tag=tag, dt=dt, iters=len(res["history"]),
                PhiI=PhiI, Finf=Finf, p_star=p_star, mus=mus,
                oneR2=oneR2, converged=res["converged"],
                taus=tuple(taus_vec), gammas=tuple(gammas_vec))


def main():
    # Warmup
    print("[numba warmup]")
    sys.stdout.flush()
    _ = rh.solve_picard(5, 2.0, 0.5, umax=UMAX, maxiters=3)
    print("[warmup done]")
    sys.stdout.flush()

    results = []
    # (1) Homogeneous baseline, γ=0.5
    results.append(run_case("(1) Homogeneous γ=0.5, τ=2",
                             taus=2.0, gammas=0.5))
    # (2) Het γ=(1,3,10), equal τ
    results.append(run_case("(2) Het γ=(1,3,10), τ=2 for all",
                             taus=2.0, gammas=(1.0, 3.0, 10.0)))
    # (3) Het γ and het τ aligned (low-γ = high-τ): γ=(1,3,10), τ=(10,3,1)
    results.append(run_case("(3) Het γ=(1,3,10), het τ=(10,3,1) — aligned",
                             taus=(10.0, 3.0, 1.0),
                             gammas=(1.0, 3.0, 10.0)))

    print("\n" + "=" * 127)
    print(f"  Heterogeneous preferences — Picard, G={G}, numba")
    print("=" * 127)
    hdr = ("config", "iters", "t(s)", "‖Φ-I‖", "‖F‖", "converged", "1-R²", "p*", "μ₁ μ₂ μ₃", "PR gap")
    print(f"{hdr[0]:<45} | {hdr[1]:>5} | {hdr[2]:>6} | {hdr[3]:>8} | {hdr[4]:>8} | {hdr[5]:>9} | {hdr[6]:>9} | {hdr[7]:>12} | {hdr[9]:>7}")
    print("-" * 127)
    for r in results:
        gap = r["mus"][0] - r["mus"][1]
        print(f"{r['tag']:<45} | {r['iters']:>5d} | {r['dt']:>6.2f} | {r['PhiI']:>8.2e} | {r['Finf']:>8.2e} | {str(r['converged']):>9} | {r['oneR2']:>9.3e} | {r['p_star']:>12.8f} | {gap:>7.4f}")
    print("-" * 127)

    # Commentary: when does PR coexist with FR?
    print("""
Commentary on PR vs FR:
───────────────────────
• Under CARA (γ → ∞) with ANY τ: logit(p) = Σ (τ_k/γ_k)·u_k / Σ (1/γ_k),
  an exact linear function of signals. Every agent's demand depends
  only on (logit μ − logit p)/γ, and the agent extracts the log-odds
  sufficient statistic perfectly from the price — full revelation.
  That is Prop 1 of project_summary.txt.

• PR "coexists" with FR in a KNIFE-EDGE sense: CARA is the unique
  preference class that gives FR. Any deviation (γ finite, even large)
  breaks the exact aggregation because the market clears in PROBABILITY
  space (Σ x_k(μ,p) = 0 is nonlinear in logit p), not log-odds space.
  The price function is then a curved surface in (u_1,u_2,u_3) and
  agents cannot invert it to T* exactly.

• Heterogeneity AMPLIFIES PR:
   - Pure Jensen gap alone (homo γ, homo τ): 1-R² ~ 10^-3 at G=9.
   - Heterogeneous γ keeps demand aggregation nonlinear AND weights
     agents unevenly: 1-R² grows to ~10^-1 (two orders larger).
   - Heterogeneous τ + aligned het γ: low-γ agents are also high-precision
     (stabilising). Misalignment (low-γ/low-τ) makes uninformed agents
     trade aggressively — the "endogenous noise trader" regime with the
     largest 1-R² in project_summary.txt.

• A true FR fixed point STILL EXISTS formally (you can always impose
  μ_k = p for all k), but:
    1. It is unstable under Picard iteration whenever γ is finite —
       any perturbation pushes Picard onto the PR fixed point that the
       contour iteration actually converges to.
    2. Under heterogeneity the "FR price" depends on which weighting
       one picks; the pointwise contour fixed point is PR.

So in the numerical sense: FR is a measure-zero idealisation reached
only in the CARA limit. For any ε > 0 away from CARA, Picard converges
to PR and 1-R² is strictly positive.
""")


if __name__ == "__main__":
    main()
