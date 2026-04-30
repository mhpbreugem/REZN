# Survival-ratio puzzle resolved (SOLVER_TODO.md §3 / Technical Notes)

Branch: `claude/add-claude-documentation-AC2lV`. Run during 2026-04-30
session.

## The puzzle

`CHAT_MEMORY.md` § "LATEST REE RESULTS" reports posterior method v3
strict (max ≤ 1e-14) at G=14, γ=0.5, τ=2: **1−R² = 0.108**. The
no-learning entry in `theory.md` § 6.1 at the same (γ, τ) is **0.062**
— but at G=20.

`SOLVER_TODO.md` § "TECHNICAL NOTES — The survival ratio puzzle" lists
four candidate explanations and asks for "no-learning at G=14 with
IDENTICAL methodology" to discriminate.

## Resolution

I ran the exact no-learning algorithm from `FIGURES_SPEC.md` SHARED
ALGORITHM at **G=14, umax=4** (same grid as v3) for the full γ ladder:

| γ \\ τ  | 0.5    | 1.0    | 2.0    |
|---------|--------|--------|--------|
| 0.10    | 0.1472 | 0.1471 | 0.1485 |
| 0.25    | 0.0604 | 0.0840 | 0.0980 |
| 0.30    | 0.0450 | 0.0706 | 0.0889 |
| 0.50    | 0.0166 | 0.0383 | **0.0619** |
| 1.00    | 0.0039 | 0.0134 | 0.0295 |
| 2.00    | 0.0010 | 0.0041 | 0.0110 |
| 3.00    | 0.0005 | 0.0020 | 0.0057 |
| 4.00    | 0.0003 | 0.0012 | 0.0035 |
| 10.00   | 0.0000 | 0.0002 | 0.0006 |
| 100.00  | 0.0000 | 0.0000 | 0.0000 |

CARA explicit: identically 0 across τ.

**The G=14 no-learning value at γ=0.5, τ=2 is 0.0619 — essentially
identical to the G=20 number (0.062).** Grid resolution from G=14 to
G=20 is not the source of the discrepancy.

## The survival ratio

REE 1−R² (v3 strict, G=14) divided by no-learning 1−R² (this file,
G=14), at τ=2:

| γ    | REE    | no-learning | survival ratio |
|------|--------|-------------|----------------|
| 0.30 | 0.119  | 0.0889      | **1.34** |
| 0.50 | 0.108  | 0.0619      | **1.74** |
| 1.00 | 0.100  | 0.0295      | **3.39** |
| 2.00 | 0.079  | 0.0110      | **7.18** |

The ratio is **monotonically increasing in γ**. At γ=2.0, REE 1−R² is
**seven times** the no-learning baseline. At every paper γ the REE
deficit strictly exceeds the static Jensen-gap baseline.

## Interpretation

Of the four candidates in the SOLVER_TODO Technical Notes:

- (a) Different G — **ruled out**: my G=14 no-learning matches G=20.
- (b) Different umax — **ruled out**: I used umax=4, same as v3.
- (c) "Real: learning from curved contours amplifies Jensen gap" —
  **this is the correct explanation**. REE > no-learning at every γ in
  {0.3, 0.5, 1.0, 2.0}, with the gap widening as γ rises.
- (d) Posterior method approximation noise — **inconsistent with the
  monotone γ pattern**. A noise floor would not increase with γ; the
  observed ratio doing so is a structural feature of REE.

## Implication for the paper

Proposition 4 (PR survives at REE) currently states only that the
deficit is positive. The numerical result is stronger: **REE
amplifies the deficit relative to no-learning, more so at higher γ**.
Two consequences:

1. The paper's no-learning figures *understate* the size of the PR
   distortion at REE. The knife-edge result is sharper than the
   no-learning curves suggest.
2. The mechanism deserves explicit mention in §4.4 or §6: agents who
   condition on a curved-contour price aggregate even less efficiently
   than agents who ignore the price entirely. Learning from a
   misaligned aggregator is worse than not learning at all (in
   the revelation-deficit metric).

## Files

- `no_learning_K3_G14.json` — full table + survival ratios in JSON.

## Open follow-ups

- The survival ratio at γ=0.1 (currently fallback in v3, max=0.05)
  cannot be computed here; would complete the row.
- At γ=4 (a paper figure value), v3 hasn't been run. No-learning gives
  0.0035; if survival ratio remains ≥ 5x, REE 1−R² should be ≈ 0.02 —
  worth running v3 at this corner of the ladder.
- The result depends on the v3 strict numbers being reproducible. The
  v3 solver itself is not committed (per repo convention); the strict
  numbers in CHAT_MEMORY rely on the recipe in
  `POSTERIOR_METHOD_V2.md` §A–E. A re-run would be a useful sanity
  check.
