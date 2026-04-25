# REZN-without-NT — Handoff for next session

## Paper

**Title (working)**: *Noise Traders Are Not a Primitive*
**Author**: Matthijs Breugem (Politecnico di Milano)
**Target**: Econometrica
**Branch**: `claude/rarar-without-nt-I8tiz`
**Repo**: `mhpbreugem/REZN`

### Core claim
CRRA preferences produce **partial revelation (PR) of information through prices WITHOUT noise traders**. CARA is a knife-edge case giving full revelation (FR). Noise traders are not a primitive — they are a substitute for the heterogeneity-induced PR that CRRA already provides.

### Key empirical findings (G=11/G=9 logit-PCHIP runs)
1. **Two co-existing CRRA equilibria** at homogeneous γ=(3,3,3):
   - **Near-CARA branch**: 1-R² ~ 1e-7 to 1e-3 depending on grid precision. Originally interpreted as a ~1e-3 PR signal (raw PCHIP), but with logit-space PCHIP it collapses to ~1e-7 — i.e., the homogeneous CRRA pre-jump branch is **essentially CARA-FR within numerical precision**. The earlier "1-R²=1e-3" was discretisation noise.
   - **Strong-PR branch (post-jump)**: 1-R² ≈ 0.04 – 0.18, real partial revelation. Found by Newton-Krylov from warm starts that drift across the basin boundary at τ ≈ 3.39. This branch persists down to τ ≈ 2.87 via the backward sweep.
2. **Asymmetric γ=(5,3,1)** at τ ∈ [0.5, 5.0]:
   - 1-R² ≈ **0.32, essentially flat in τ** (300× the homogeneous pre-jump value).
   - This is the cleanest result: **heterogeneity in γ alone drives strong PR**, independent of signal precision.
3. **PR gap (μ₁−μ₂ at (1,−1,1))** on post-jump branch is small (~1e-4 at G=9 logit-PCHIP) despite large 1-R². Interpretation: post-jump distortion is symmetric in trader indices (e.g., depends on Σu²), so global nonlinearity is large but per-trader information asymmetry at this single cell is small. Multi-cell PR diagnostics would clarify.

## Numerical methodology — current state

### Model
3 traders, each with CRRA utility `w^(1-γ_k)/(1-γ_k)`, signal `s_k = v + η_k` with precision `τ_k`, Bernoulli payoff `v ∈ {0,1}`, equal initial wealth `W = 1`. Walrasian market clearing in 1 risky + 1 riskless asset.

State: tensor `P[i,j,l]` of equilibrium prices on a `G³` signal grid with each axis `u ∈ [-UMAX, UMAX]`. Fixed-point map `Φ(P)`:
1. For each cell `(i,j,l)`: posterior `μ_k(u_k, p_obs, P)` for each of 3 traders via contour integration over the level set `{P_slice_k = p_obs}`, where `slice_k` fixes trader k's signal index.
2. Market clearing at `(i,j,l)`: solve `Σ_k x_k(μ_k, p) = 0` for `p`, where `x_k` is CRRA demand.
3. Output: `Φ(P)[i,j,l] = p`.

### Code map
All in `python/`:
- `rezn_het.py` — heterogeneous (τ, γ) primitives. `@njit` numba. Contains `_phi_map`, `_residual_array`, `_clear_price`, `_nolearning_price`, `_as_vec3`, posterior helpers. **3-pass contour for CARA reference; piecewise-linear**.
- `rezn_pchip.py` — PCHIP version of the contour integration. **The version we use for results.** `@njit`. Contains:
  - `_pchip_derivs` — Fritsch-Carlson derivatives.
  - `_hermite_val`, `_hermite_deriv` — cubic Hermite evaluation.
  - `_pchip_root_in_segment` — Newton+bisection for level-set crossings.
  - `_contour_sum_pchip` — **2-pass contour, now interpolating in LOGIT(P) space** (not raw P) — this is the major precision improvement that dropped Φ-noise from ~1e-4 to ~1e-13.
  - `_phi_map_pchip` — builds Φ tensor.
  - `solve_picard_pchip(G, taus, gammas, ...)` — fixed-point iteration with min-iterate tracking.
  - `solve_anderson_pchip(...)` — Anderson acceleration with min-iterate tracking.
- `pchip_continuation.py` — main driver. Configurable `G`, `UMAX`, `F_TOL`, `ABSTOL`. γ-chain seed → γ-chain → τ-sweep. Solver ladder: NK → Anderson m=6,10,15 → Picard. Writes `CSV_OUT` and pickles `CACHE_PKL`. Contains `_solve_nk` (scipy newton_krylov with rdiff=1e-8, lgmres, callback for live status). **CACHE accepts Finf < 1e-6 even if conv=0** so warm-start chain doesn't break.
- `pchip_backward_sweep.py` — walks τ DOWNWARD from a post-jump seed (τ=3.39+), warm-starting only from the previous backward solution. Used to demonstrate post-jump branch persists below τ=3.39.
- `pchip_asymmetric_sweep.py` — γ=(5,3,1) τ-sweep.
- `plot_workhorse.py` — paper-quality figures: fig1_branches, fig2_gamma_sweep, fig3_asymmetric, fig4_composite.
- `plot_paper.py`, `plot_branches.py` — older plots, mostly superseded.
- `run_overnight.sh` — pipeline: forward → backward → asymmetric → plots → push.
- `analyze_weighted_R2.py` — value-of-information computation `V_k^full`, `V_k^pub`, ΔV_k. Posterior in public-only signal space via kernel density in logit-price.
- `newton_krylov_test.py` — early standalone NK test on the PCHIP map.
- `push_1em12_v3.py` — Aitken Δ² + central-difference Newton-Krylov experiments.
- `pchip_G11_*_snap.csv` — snapshots from various runs (G=11 forward and backward).
- `pchip_continuation_results.snap_20260424.csv` — snapshot of an earlier 130-row G=11 run (had the τ=3.46 fake-converged jump).

### Solver behaviour summary
- **Picard**: linearly convergent at rate ρ = spectral radius of the Φ Jacobian. At γ=500 ρ≈0.001 (fast), at γ=3 ρ≈0.999 (tens of thousands of iters needed). Hits machine precision at G=5 cleanly; at G=11 plateaus at ~1e-7 because of accumulated FP noise in PCHIP+contour.
- **Anderson**: superlinear when ρ < 1, polishes 1 order beyond Picard's floor in good cases.
- **Newton-Krylov (scipy `newton_krylov` lgmres)**: quadratic convergence given a warm start, but FD-Jacobian noise floor is fundamental: optimal `rdiff = sqrt(eps_F)`, giving Jacobian accuracy `√eps_machine · ||F''||` ~ 1e-8. **This is the wall.**
- **Critical fix**: interpolating PCHIP in **logit(P) space** instead of raw P. Logit is linear in τ·u under CARA-FR, so cubic interpolation is essentially exact and Φ-noise drops from ~1e-4 to ~1e-13. **Already implemented in `rezn_pchip.py`'s `_contour_sum_pchip`.**

### Tolerance conventions
Constants in `pchip_continuation.py`:
- `ABSTOL` = Picard-step tolerance the inner solver iterates to (typically 1e-11).
- `F_TOL` = acceptance threshold on the true fixed-point residual `||F(P)||_∞` for `converged=1` flag (depends on what we can reach: 1e-7 for paper-clean, 1e-5 for sweep speed, 1e-3 for capturing post-jump rows).
- Solvers internally track the **minimum residual iterate seen** during iteration (not just the last one) — this is critical because Picard near-fixed-point oscillates.

### Workhorse parameters (current target)
- G=11, UMAX=2.5, F_TOL=1e-7, sparse 10-point τ-sweep, 15-point γ-sweep.

## What's been done

### Sweeps in CSV form (in `python/`)
- `pchip_G11_forward_snap.csv` — old G=11 forward sweep, 248 rows.
- `pchip_G11_backward_snap.csv` — backward sweep along post-jump branch.
- `pchip_G11logit_forward.csv` — G=11 logit-PCHIP forward.
- `pchip_G9_forward.csv` — G=9 logit-PCHIP forward (the run that revealed pre-jump 1-R² ~ 1e-7).
- `pchip_asymmetric_results.csv` — γ=(5,3,1) sweep at G=11 UMAX=2.0.
- `pchip_G11u25_forward.csv` — workhorse G=11 UMAX=2.5 (in progress at handoff time).

### Plots committed
- `plot_overview.png`, `plot_two_branches.png`, `plot_asymmetric.png`, `plot_branches.png` — older versions.
- `fig1_branches.png`, `fig2_gamma_sweep.png`, `fig3_asymmetric.png`, `fig4_composite.png` — workhorse paper figures (auto-generated by `plot_workhorse.py`).

## Open / pending

### High priority
1. **Analytic Jacobian for the Φ map** — the only way to push F-residual below 1e-8 at G=11 (3-4 hour implementation). Sketch:
   - At cell `(i,j,l)`: `Φ[i,j,l] = p*` solving `Σ_k x_k(μ_k, p*) = 0`.
   - `μ_k = g_{k1} A_{k1} / (g_{k0} A_{k0} + g_{k1} A_{k1})` where `A_{kc}` is sum over level-set crossings.
   - `x_k(μ_k, p, γ_k)` has closed form for CRRA: `x = W·(A − B)/(A·p + B·(1−p))` with `A = (μ(1−p))^(1/γ)`, `B = ((1−μ)p)^(1/γ)`. Differentiable analytically.
   - Total derivative `dΦ[i,j,l]/dP[i',j',l']` requires:
     - `∂x_k/∂μ_k`, `∂x_k/∂p` (analytic).
     - `∂μ_k/∂(slice cell)` and `∂μ_k/∂p_obs` via `_pchip_root_in_segment` implicit derivative: at a root `t*` with `H(t*;y0,y1,m0,m1) = p_obs`, we have `dt*/dy_i = -∂H/∂y_i / ∂H/∂t`. This propagates to derivatives of contour weights `f0/f1` evaluated at the crossing.
     - Implicit-function chain through market clearing: `dp/dμ_k = -(∂h/∂μ_k)/(∂h/∂p)` where `h = Σ x_k`.
   - **Sparsity**: `dΦ[i,j,l]/dP[i',j',l'] = 0` unless `i=i'` OR `j=j'` OR `l=l'` (each trader's slice fixes one index). For `G=11` the Jacobian is ~30% dense in N×N where N=G³.
   - Implement as `J_dot_v(P, v) -> J·v` so we can use scipy's `lgmres` with a `LinearOperator` — no need to materialise the full matrix. **Same code works for any G** — derivation is grid-agnostic.
2. **Resume τ-sweep with analytic Jacobian** — once Jacobian is in, F-residual will reach machine precision, all conv=0 rows currently flagging the post-jump branch will become conv=1.
3. **Multi-cell PR diagnostics** — currently we report `pr_gap = μ_1 − μ_2` only at the cell `(u_i,u_j,u_l) = (1,-1,1)`. Add μ-gap at multiple reference cells to clarify whether the post-jump branch has zero PR gap globally or only at this specific cell.
4. **Value-of-information sweep** — `analyze_weighted_R2.py` computes `V_k^full − V_k^pub`. Run across the workhorse grid, plot ΔV vs (γ, τ).

### Lower priority
5. **Heterogeneous γ surface** — sweep γ=(γ_1, γ_2, γ_3) over the upper triangle (γ_1 ≤ γ_2 ≤ γ_3) and plot 1-R² as a function of γ-spread.
6. **Heterogeneous τ** — symmetric question on signal-precision side.
7. **G=15 / G=21 confirmation** — once analytic Jacobian works, confirm the post-jump branch and its 1-R² value at finer grids. Check whether 1-R² ≈ 0.04-0.18 is grid-converged.

### Done but worth re-running with analytic Jacobian
- All τ-sweeps and γ-sweeps. Current results have Finf ~1e-7 at best (limited by FD-Jacobian floor); analytic Jacobian gets to ~1e-13. The qualitative picture won't change but plots will be smooth.

## Operational notes

### Restart pattern
```bash
# Kill anything running
pkill -f 'pchip_continuation\|run_overnight\|pchip_backward\|pchip_asymmetric'
# Remove stale status
rm -f python/sweep_status.txt
# Launch the pipeline
nohup /home/user/REZN/python/run_overnight.sh > python/run_overnight.out 2>&1 &
disown
```

### Live status
- `python/sweep_status.txt` is updated every 25 iterations by Picard/Anderson and every NK outer iteration.
- `python/sweep_G15.log` (or named log) accumulates per-config completion lines.

### Branch / push
- Working branch: `claude/rarar-without-nt-I8tiz`. Push with `-u origin <branch>`. Local proxy git remote at `127.0.0.1:<port>/git/mhpbreugem/REZN`. Initial pushes sometimes fail with "could not read Password" — wait 4-8s and retry; exponential backoff up to 16s recommended.
- Don't force-push. Don't push to other branches.

## Suggested first message in new chat

> Read `HANDOFF.md` and the file map. Then implement the analytic Jacobian for the PCHIP+contour Φ map as `J_dot_v(P, v) -> J·v` in `rezn_pchip.py` (or a new module), wire it as a custom Newton solver in `pchip_continuation.py` (replacing or complementing scipy's `newton_krylov`), and re-run the workhorse sweep at G=11 UMAX=2.5. Target: Finf < 1e-12 across all converged rows, smooth plots from `plot_workhorse.py`. Continue the heterogeneous-γ exploration after that.

## Files in repo to read first
1. `HANDOFF.md` (this file)
2. `python/rezn_pchip.py` (PCHIP map, ~370 lines)
3. `python/rezn_het.py` (primitives, contour, demand)
4. `python/pchip_continuation.py` (driver and solver ladder)
5. `python/plot_workhorse.py` (figures)
