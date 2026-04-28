# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a single-paper LaTeX research project, not a software codebase. It contains the manuscript, figures, and supporting theory/numerics notes for:

> **"Noise Traders Are Not a Primitive"** — Matthijs Breugem, Nyenrode Business University. Target: *Econometrica*.

Core claim: CRRA preferences produce partial revelation (PR) of information through prices **without noise traders**. CARA is a knife-edge — only exponential utility gives full revelation (FR). The mechanism is the **aggregation space** mismatch: CARA aggregates demands in log-odds (matching the Bayesian sufficient statistic $T^\star = \sum_k \tau_k u_k$); non-CARA CRRA aggregates in probability space, and the Jensen gap is the size of the revelation failure.

## Two-agent workflow (important context)

This repo is shared between two Claude surfaces with different roles. Reading the wrong file for the wrong role wastes context:

- **Paper chat (claude.ai)** — owns paper *content*: `main.tex`, narrative, propositions, figure `.tex` files. See `LATEX_CHAT_INSTRUCTIONS.md`.
- **Claude Code (here)** — owns *heavy computation*: contour-method REE solves at large $G$, figure data CSVs, and any Python/Julia solver scripts that produce them.

`CHAT_MEMORY.md` is the living handoff between sessions — read it first if context was compacted; it has the latest decisions, open TODOs, and the "round N polish" history of `main.tex`. `theory.md` is the single source of truth for the model and method.

## Repository layout

```
main.tex / main.pdf       The Econometrica manuscript (~28 pages)
references.bib            Bibliography
figures/                  All figures: <name>.tex (pgfplots standalone) + <name>.pdf + <name>.png
                          + optional <name>_data.csv for the underlying numbers
figures/NOTES_full_eq_status.md   Why fig4/6/7/8/9/10 are still yellow placeholders

theory.md                 Single source of truth — model, propositions, contour method, parameter defaults
contour.md                Plain-language walkthrough of the 2-pass contour method
project_summary.txt       Original project spec (frozen)

CHAT_MEMORY.md            Living handoff between sessions; latest decisions and open TODOs
SESSION_SUMMARY.md        Earlier session log (Python solver experiments, lessons learned)
LATEX_CHAT_INSTRUCTIONS.md System prompt for spinning up a new paper chat
FIGURES_SPEC.md           Per-figure algorithms, parameters, expected reference numbers
PAPER_ADDITIONS.md        LaTeX snippets queued for insertion into main.tex
```

There is no Python/Julia source tree in this repo. `figures/NOTES_full_eq_status.md` references `python/probe_*.py` and `python/ree_solver.py` — those scripts live outside the repo (previous Claude Code workspaces). Recreate them locally if you need to rerun a solve.

## Build / compile commands

LaTeX is the only "build". Standard latexmk pipeline:

```
latexmk -pdf main.tex          # full build (bibtex + pdflatex passes)
latexmk -c                     # clean .aux/.log/.bbl/.blg (also matches .gitignore)
pdflatex main.tex              # quick single-pass; rerun to settle refs
```

Each figure under `figures/` is a **standalone** pgfplots document — compile in isolation when iterating:

```
cd figures && pdflatex fig_knife_edge.tex
```

`main.tex` includes figures via `\includegraphics{figures/<name>.pdf}`, so rebuild the figure PDF before rebuilding the paper.

There are no tests, linters, or CI. The only "test" is: paper compiles with 0 warnings and 0 undefined references (current state per `CHAT_MEMORY.md`).

## Figure conventions (BC20 pgfplots) — must match exactly

All plot figures use this style. New figures that diverge will look wrong next to the existing ones.

```latex
\definecolor{red}{rgb}{0.7, 0.11, 0.11}
\definecolor{blue}{rgb}{0.0, 0.20, 0.42}
\definecolor{green}{rgb}{0.11, 0.35, 0.02}
```

- 8cm × 8cm axes, `legend style={draw=none, legend pos=north west, font=\footnotesize}`
- Line order: green solid → red dashed → blue dotted → black dashdotted (CARA baseline)
- `very thick` for CRRA curves, `ultra thick` for CARA, all `smooth`, `ymin=-0.001` so CARA sits visibly above the x-axis
- Six figures (4, 6, 7, 8, 9, 10) currently render with a **yellow background** to flag them as placeholders awaiting a converged REE solve. Do not strip the yellow until the underlying numerics actually converge — see `figures/NOTES_full_eq_status.md` for what's blocking each one.

`FIGURES_SPEC.md` has the per-figure algorithm, exact parameters ($G$, $\tau$, $\gamma$ grids), and reference numbers to validate against.

## The contour method (the only nontrivial algorithm)

The full-REE solve is a fixed point $P = \Phi(P)$ on a $G \times G \times G$ price array $P[i,j,l]$. **$P$ is the only unknown**; the root-found off-grid signals, contour integrals, posteriors, and market-clearing prices are all intermediate computations inside one $\Phi$ evaluation. Do not introduce them as additional unknowns.

One $\Phi$ call, given $P$:

1. For each $(i,j,l)$, read $p = P[i,j,l]$.
2. For each agent $k$, take her **own** 2D slice (agent 1: `P[i,:,:]`, agent 2: `P[:,j,:]`, agent 3: `P[:,:,l]` — no transposition trick).
3. Two-pass contour: sweep one axis on grid, root-find the other off grid via linear interpolation; do it again with axes swapped; average. Each crossing contributes $f_v(u_a)\,f_v(u_b)$ to $A_v$ — these are **prior** signal densities, not posteriors. No Jacobian (it cancels in the posterior ratio).
4. Bayes with own signal: $\mu_k = f_1(u_\text{own})\,A_1 / (f_0(u_\text{own})\,A_0 + f_1(u_\text{own})\,A_1)$.
5. Market clearing $\sum_k x_k(\mu_k, p_\text{new}) = 0$ → write $P_\text{new}[i,j,l]$.

CARA → straight contours → uniform $T^\star$ along the level set → FR. CRRA → curved contours (Jensen gap) → varying $T^\star$ → PR. `contour.md` walks through a single point with arithmetic; `theory.md §5` is the authoritative spec.

**Solver lessons from prior sessions** (`SESSION_SUMMARY.md`, `figures/NOTES_full_eq_status.md`) — do not relearn:
- Binning creates a noise floor (~0.012 at $G=15$–$20$) identical for CARA and CRRA — useless for distinguishing PR from artifact.
- Cubic interpolation manufactures fake curvature that grows over iterations; even CARA develops spurious PR.
- Picard ($\alpha=0.15$–$0.3$) is intuitive but slow and oscillates at low $\gamma$.
- Anderson ($m=6$–$8$) converges in ~6 steps but stalls at the interpolation floor.
- The PR branch is **not** Picard-reachable from the no-learning seed at most parameter values — it requires Newton-Krylov with carefully perturbed warm starts. The historical PR-branch tensors that produced the strong-PR numbers in the HANDOFF are not currently reproducible from any seed schedule on file.
- Always run a CARA baseline at the same $G$ to measure the noise floor; report CRRA-minus-CARA as the genuine PR signal.

## Parameter defaults (when in doubt, use these)

From `theory.md §13`: $\tau = 2.0$, $\gamma = 0.5$ (CRRA main case) or $\to\infty$ for explicit CARA, $W = 1$, $K = 3$, $u$-grid on $[-4, 4]$. Grid: $G = 5$ debug, $G = 20$ paper, $G \geq 100$ for publication-quality contour figures.

## Editing main.tex

- All proofs go in Appendix A. There is one exception currently inline (Lemma 3 / CARA-as-CRRA-limit) flagged for migration in `PAPER_ADDITIONS.md §7`.
- Open analytical/numerical TODOs render as red sansserif `\todo{...}` boxes via the macro defined at `main.tex:24`. Don't strip them silently — they're tracked in `CHAT_MEMORY.md` (items 1–13 of the round-3 list).
- `PAPER_ADDITIONS.md` is a queue of LaTeX snippets (definition of $1-R^2$, existence/uniqueness lemma, HARA remark, zero-supply footnote, multiple-equilibria paragraph, strengthened conclusion). Insert from there rather than re-deriving.
- Citations use `natbib` round/authoryear; bibliography is `references.bib`.

## Git

Working branch for this session: `claude/add-claude-documentation-xGy0S`. Historical work has happened on `main` and `REZN` — earlier instructions in `LATEX_CHAT_INSTRUCTIONS.md` reference pushing to `main`, but follow whatever branch the current task specifies. `.gitignore` covers LaTeX auxiliary files only (`.aux`, `.bbl`, `.log`, etc.) — do not commit `main.pdf` regenerations unless asked; the current `main.pdf` is the canonical compiled artifact and is tracked.
