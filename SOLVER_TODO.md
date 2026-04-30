# SOLVER TODO — Posterior Method v3
# Priority-ordered tasks for Claude Code
# Read POSTERIOR_METHOD_V2.md (sections A-E) for method details
# Current best results: G=14 strict (max≤1e-14) at γ=0.3,0.5,1,2 and τ=0.5,1,2,4,8

---

## STATUS SUMMARY

### What works (strict convergence, max≤1e-14)
- G=10, 12, 14 at baseline (γ=0.5, τ=2)
- G=14 γ-ladder: γ=0.3, 0.5, 1.0, 2.0
- G=14 τ-ladder: τ=0.5, 1.0, 2.0, 4.0, 8.0

### What doesn't work yet
- G≥16: NK-monotone trade-off. Picard+PAVA converges median cells but
  max residual stuck at ~0.08-0.20. NK polish breaks p-monotonicity.
- γ=0.1: NK breaks p-monotonicity. Fallback 1-R²=0.154 but max=0.05.
- γ≥5: not attempted at strict level.

---

## P0: CRITICAL FOR PAPER

### 1. Run at paper figure gammas: γ = 0.25, 1.0, 4.0
The paper figures use γ=0.25, 1, 4 (not 0.3, 0.5, 2). Need strict
convergence at G=14 for all three at τ=2.
- γ=0.25: between 0.3 (done, 1-R²=0.119) and 0.1 (fallback). Should converge.
- γ=1.0: already done (1-R²=0.100). Verify.
- γ=4.0: between 2.0 (done, 1-R²=0.079) and 10 (not done). Should converge.

### 2. CARA baseline at each parameter point
Run explicit CARA (γ=100 or γ=50) at G=14, τ=2. Must give 1-R²≈0
(machine precision). This is the ground truth — if CARA doesn't give
FR, the method has a bug. Currently only verified informally.
Report: converged μ*(u,p) = p (or close), residual, 1-R².

### 3. REE survival ratios
For each strictly converged (γ, τ):
- Compute no-learning 1-R² at the same G=14 (this is fast, no iteration)
- Report: REE 1-R² / no-learning 1-R² = survival ratio
- Current puzzle: REE 1-R² (0.108) > no-learning 1-R² (0.062) at γ=0.5, τ=2.
  This needs verification. Is the no-learning number computed at G=14 or G=20?
  Recompute both at G=14 with identical 1-R² methodology.

### 4. Posteriors table for Fig 4
At the converged μ* (G=14, γ=0.5, τ=2), reconstruct P at realization
(u₁,u₂,u₃) = (1, -1, 1):
- Solve market clearing: F(p) = Σ d(u_k; p) = 0 for p
- Extract posteriors: μ*(u_k, p) for k=1,2,3
- Report table:

| | Prior μ_k | CARA posterior | CRRA posterior |
|---|---|---|---|
| Agent 1 (u=+1) | | | |
| Agent 2 (u=-1) | | | |
| Agent 3 (u=+1) | | | |
| Price | | | |

---

## P1: HIGH PRIORITY

### 5. Break through G≥16
The NK-monotone trade-off is the main blocker. Three approaches to try:

a) **Gap reparametrization** (POSTERIOR_METHOD_V2.md §E): Instead of
   PAVA projection, reparametrize as logit(μ_k) = logit(μ_{k-1}) + exp(c_k).
   Monotonicity by construction, smooth, nonsingular Jacobian. Run NK
   in (base, c) space. May avoid the NK-vs-monotone conflict entirely.

b) **Warmer warm-start**: Interpolate converged G=14 onto G=16 grid
   with high-order interpolation (cubic, not linear). The G=14
   solution is strict — should be a good starting point if interpolated
   carefully.

c) **Hybrid**: Picard-PAVA to get close (max~0.01), then gap-reparam
   NK to polish.

### 6. Knife-edge figure data at full REE
At G=14, sweep τ = 20-30 log-spaced points from 0.1 to 10.
Three curves: γ = 0.25, 1.0, 4.0.
Output as pgfplots coordinates.
This is the REE version of the no-learning knife-edge figure.
Compare: are the curves similar? Does learning change the shape?

### 7. Contour figure data (Fig 3)
From the converged μ* at G=14, γ=0.5, τ=2:
- Reconstruct the price surface P(u₁, u₂, u₃) at u₁=1
- Extract the level set at p₀ = P(1, -1, 1)
- Report: list of (u₂, u₃, T*) crossing points
- Also do the same for CARA (should be a straight line)
Side-by-side: straight (CARA) vs curved (CRRA)

### 8. Convergence figure data (Fig 5)
From the iteration history at G=14, γ=0.5, τ=2:
- Report: ||Φ-I||∞ at each iteration
- If Picard was used first then NK, report both stages
- Output as pgfplots coordinates for a log-y plot

---

## P2: MEDIUM PRIORITY

### 9. Extended γ-ladder
Fill in: γ = 0.1 (strict, not fallback), γ = 3, 5, 10, 20.
The γ→∞ limit should give 1-R²→0. Currently only γ≤2 strictly converged.
The full curve γ ∈ [0.1, 20] at τ=2 would be a powerful figure.

### 10. Trade volume (Fig 7)
At each converged μ*:
- E[|x_k|] = Σ w(u₁,u₂,u₃) · |x_k(μ*(u_k, P(u₁,u₂,u₃)), P)|
- Requires reconstructing P at each grid triple (expensive but doable)
- Report one number per (γ, τ) pair

### 11. Value of information (Fig 8)
- E[U_informed] = Σ_{v,i,j,l} 0.5·f_v(i)f_v(j)f_v(l) · U(W + x_k·(v-P))
- E[U_uninformed] = U(W) (since x=0 when μ=1/2 and p=1/2)
- V(τ) = E[U_informed] - E[U_uninformed]
- Report V(τ) for each converged (γ, τ)

### 12. Two-branch investigation
At G=14, seed from FR array. Does it stay at FR? (It should.)
Report the basin structure: how far from FR can you perturb before
falling to the PR branch?

---

## P3: LOW PRIORITY

### 13. K=4 extension
The side-projects branch has K=4 code. Can the posterior method handle K=4?
μ(u, p) is still 2D (same for each symmetric agent). The contour becomes
3D: d(u₁)+d(u₂)+d(u₃)+d(u₄)=0. Sweep u₂,u₃, root-find u₄. Harder but
the posterior method still applies. Try at G_u=10, G_p=10.

### 14. Heterogeneous γ at full REE
The REZN branch showed het γ=(5,3,1) gives 1-R²≈0.32 at no-learning.
What happens at full REE? Need separate μ_k(u, p) per agent type.
State: K × G_u × G_p. At K=3, G=14: 3×14×14 ≈ 600 unknowns. Feasible.

### 15. Gauss-Hermite quadrature
Replace equally-spaced u₂ sweep with Gauss-Hermite nodes weighted by
the signal density. Should improve accuracy at extreme signals without
increasing sweep count. Compare 1-R² at N_GH=20 vs N_uniform=50.

### 16. Publication table
Once all P0+P1 tasks are done, produce a single table:

| γ | τ | No-learning 1-R² | REE 1-R² | Survival | Slope | E[|x|] | V(τ) |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.5 | | | | | | |
| 0.25 | 1.0 | | | | | | |
| ... | ... | | | | | | |
| CARA | all | 0 | 0 | — | 1.0 | 0 | 0 |

---

## TECHNICAL NOTES

### The survival ratio puzzle
REE 1-R² (0.108) appears LARGER than no-learning 1-R² (0.062) at
γ=0.5, τ=2. This could be:
a) Different G used (no-learning was G=20, REE is G=14)
b) Different UMAX (no-learning uses [-4,4], posterior method may differ)
c) Real: learning from curved contours amplifies Jensen gap
d) The posterior method's 1-R² includes the contour-integration
   approximation error in both directions

Resolution: recompute no-learning at G=14 with IDENTICAL methodology
(same u-grid, same weight computation, same regression). If still
REE > no-learning, it's finding (c) and should be investigated.

### Strict convergence definition
max(|Φ(μ) - μ|) ≤ 1e-14 over ALL active cells.
Zero u-monotonicity violations. Zero p-monotonicity violations.
This is machine precision — the answer cannot improve further at
float64.

### File naming convention
Save converged μ arrays as:
  posterior_v3_strict_G{G}_gamma{γ}_tau{τ}.npz
Save summary JSON as:
  posterior_v3_strict_results.json (append to existing)
Push to results/full_ree/ on main branch.

---

## P0.5: URGENT DIAGNOSTIC — CARA FLOOR

The CARA baseline at γ=50 gives 1-R²=0.037 (status: no_strict).
This could be:
  (a) γ=50 is not large enough — the CRRA demand at γ=50 still has
      a tiny Jensen gap that gets amplified by the REE loop
  (b) Convergence artifact — the solver didn't reach the true fixed point
  (c) Method artifact — the posterior-function discretization introduces
      nonlinearity that even true CARA can't escape

### Tests to discriminate:

**Test 1: Higher γ sweep.**
Run γ = 50, 100, 200, 500 at G=14, τ=2. All with same solver settings.
- If 1-R² keeps falling toward zero: (a) confirmed. Report the decay rate.
- If 1-R² plateaus at ~0.037: (b) or (c).

**Test 2: Explicit CARA demands.**
Replace the CRRA demand formula x = W(R-1)/((1-p)+Rp) with the exact
CARA formula x = (logit(μ)-logit(p))/γ_CARA (with some fixed γ_CARA
for scaling). Everything else identical: same posterior method, same
PAVA, same grid. Run at G=14, τ=2.
- If 1-R² = 0 (machine precision): the method works perfectly for true
  CARA, and the 0.037 at γ=50 is genuine CRRA residual → finding (a).
  This would mean: the REE amplifies even the tiniest Jensen gap by
  1000×, making CARA an even sharper knife-edge than no-learning suggests.
- If 1-R² > 0: the posterior-function discretization introduces artifacts
  even for true CARA → finding (c). Must fix the method.

**Test 3: Convergence check at γ=50.**
Run γ=50 with more iterations (double), tighter damping, Anderson with
larger window. Check if 1-R² is still falling when stopped, or truly flat.
Report: 1-R² at iteration 100, 200, 500. Is it still decreasing?

### Why this matters

If finding (a): the paper's story is even stronger. The no-learning
knife-edge table shows γ=50 as "effectively CARA" (1-R²=0.0000). But
the REE amplifies that invisible gap to 0.037 — detectable and
economically significant. The knife-edge is SHARPER at REE than at
no-learning. Add a paragraph to the paper about this.

If finding (b) or (c): need to subtract the floor from all CRRA numbers.
The NET deficits from CHAT_MEMORY are still valid but the raw numbers
in the paper's tables need a baseline correction column.

### Expected result

Most likely (a). Reason: the no-learning 1-R² at γ=50 is 0.00003.
The survival ratio at γ=2 is 3.8×. If the ratio continues growing
(which the data suggests — it's monotone in γ), then at γ=50 the
ratio could be 100-1000×, giving 0.003-0.03 for the REE — consistent
with the observed 0.037. The high-γ CRRA demand is nearly linear but
not exactly linear, and the REE loop amplifies the residual nonlinearity.

Test 2 (explicit CARA) is the definitive discriminator.
