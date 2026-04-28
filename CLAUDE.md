# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is **not a software project**. It is the working tree for an Econometrica
paper, *"Noise Traders Are Not a Primitive"* by Matthijs Breugem (Nyenrode).
The repo holds the LaTeX manuscript, its figures, the bibliography, and a set
of Markdown documents used as long-term memory across Claude sessions. There
is no application code checked in; numerical solvers are developed and run in
ephemeral workspaces, and only their *outputs* (figure `.tex`, `.pdf`, and
`.csv` data) are committed.

Top-level files:

- `main.tex` — the manuscript (~1780 lines, ~28 pages, 8 sections + 2
  appendices: `app:proofs`, `app:contour`).
- `references.bib` — natbib bibliography.
- `main.pdf` — committed build artefact; intended to stay in sync with
  `main.tex`.
- `figures/` — one `fig*.tex` + `fig*.pdf` per figure, often with sibling
  `fig*_data.csv` or `fig*_data.tex`. `figures/NOTES_full_eq_status.md`
  tracks the current state of the numerical work for the placeholder
  figures.

## The dual-chat workflow (important)

Two Claude sessions cooperate on this paper, and the Markdown files in the
root encode their division of labour. Read this before editing anything
substantive.

- **Claude Code (this agent)**: heavy computation. Runs the contour-method
  solver, generates figure data, produces `figures/fig*.tex/pdf/csv`,
  pushes them to the repo. Does *not* normally rewrite the paper.
- **A separate "paper" chat**: reads the figure outputs and edits
  `main.tex`. `LATEX_CHAT_INSTRUCTIONS.md` is the system prompt for that
  chat.
- **Shared memory**: `CHAT_MEMORY.md` is the canonical running log — read it
  first if you have lost context. It records decisions, current numerical
  results, polish-pass history, and the open TODO list visible in the PDF.

When making changes, respect the lane: if you are doing numerics, update
`figures/` and report back via `CHAT_MEMORY.md` or
`figures/NOTES_full_eq_status.md`; if you are doing prose/structure, edit
`main.tex` and update `CHAT_MEMORY.md` accordingly. `theory.md` is the
single source of truth for the model and propositions — sync it before
committing analytical changes.

## Document hierarchy (read in this order on a cold start)

1. `CHAT_MEMORY.md` — most recent state, decisions, open TODOs.
2. `theory.md` — canonical model, propositions, numerical method,
   parameter defaults.
3. `contour.md` — plain-language walkthrough of the contour method (the
   numerical core of Section 4 / Appendix B).
4. `FIGURES_SPEC.md` — exact algorithm and pgfplots template for every
   figure, in priority order.
5. `figures/NOTES_full_eq_status.md` — current solver state for the
   placeholder figures (Figs 4, 6, 7, 8, 9, 10).
6. `PAPER_ADDITIONS.md` — drop-in LaTeX snippets queued for insertion
   into `main.tex` (definition of $1-R^2$, existence lemma, HARA remark,
   zero-supply footnote, etc.). Already partially applied; check
   `git log` against `main.tex` before re-applying.
7. `SESSION_SUMMARY.md` and `project_summary.txt` — older overviews,
   superseded by `theory.md` + `CHAT_MEMORY.md` but useful for historical
   context.
8. `LATEX_CHAT_INSTRUCTIONS.md` — written for the *other* chat; treat it
   as documentation of intent, not as instructions for Claude Code.

## Build commands

LaTeX with pgfplots and natbib. There is no Makefile.

```bash
# Full build (run twice to resolve refs, plus bibtex)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Quick rebuild while editing prose (no bib changes)
pdflatex main.tex

# Build a single figure (each figures/fig*.tex is a \documentclass{standalone})
cd figures && pdflatex fig3_contour.tex
```

The `.gitignore` covers LaTeX aux files (`*.aux`, `*.bbl`, `*.log`, …).
`main.pdf` and `figures/*.pdf` *are* tracked — rebuild and commit them
together with any `.tex` change so the PDF on `main` stays current.

## Manuscript conventions

- **Paper class**: `article` (12pt letterpaper), not `econometrica.cls`.
  Macros and theorem environments are defined in the preamble of
  `main.tex` (lines ~1–55). Notable macros: `\Lam` (Λ), `\logit`,
  `\Tstar` ($T^\star$), `\R`, `\E`, `\PP`, `\Var`, `\Cov`.
- **TODO markers**: `\todo{...}` and `\todopar{...}` render red sansserif
  boxes in the PDF and mark every outstanding item. Roughly 13 are open
  (see end of `CHAT_MEMORY.md` for the canonical list — split into
  numerical items 1–7 owned by Claude Code and analytical items 8–13
  that belong in paper revisions). Do not silently delete a `\todo`;
  resolve it or report what you changed.
- **Section labels**: `sec:intro`, `sec:model`, `sec:agg-space`,
  `sec:no-learning`, `sec:ree`, `sec:welfare`, `sec:mechanisms`,
  `sec:discussion`, `sec:vanishing-noise`, `sec:conclusion`. Appendices
  `app:proofs` and `app:contour`. Cross-reference with `\ref` /
  `\eqref`.
- **Proofs go to the appendix** (`app:proofs`). When adding a result,
  put a short sketch in the main text and the full proof in Appendix A.

## Figure conventions (BC20 pgfplots style)

Every plot in `figures/` follows the same template — match it exactly when
generating new figures. Defined in both `FIGURES_SPEC.md` §STYLE and
`theory.md` §12.

```latex
\definecolor{red}{rgb}{0.7, 0.11, 0.11}
\definecolor{blue}{rgb}{0.0, 0.20, 0.42}
\definecolor{green}{rgb}{0.11, 0.35, 0.02}
\begin{axis}[
  legend style={draw=none, legend pos=north west, font=\footnotesize},
  ticklabel style={/pgf/number format/fixed,/pgf/number format/precision=5},
  scaled ticks=false,
  xticklabel style={/pgf/number format/1000 sep=, font=\scriptsize},
  width=8cm, height=8cm,
]
```

Curve order, used consistently across figures: green solid (γ small) →
red dashed (γ middle) → blue dotted (γ large) → black dashdotted, ultra
thick (CARA baseline). Always `smooth`, always `ymin=-0.001` so the CARA
zero line is visible.

Each `figures/fig*.tex` is a `\documentclass[border=2mm]{standalone}` —
it builds its own `.pdf` and `main.tex` includes that `.pdf` via
`\includegraphics`. When updating a figure, **regenerate both the
`.tex` and the `.pdf`** and commit them together.

Six figures are currently placeholders with a yellow background:
`fig4_posteriors`, `fig6_mechanisms`, `fig7_volume`, `fig8_value_info`,
`fig9_GS`, `fig10_K_agents`. The blocker is documented in
`figures/NOTES_full_eq_status.md`: the production solver collapses onto
the FR (full-revelation) fixed point and cannot reliably reach the PR
branch reported in the analytical results. Either recover the historical
seed schedule that produced the PR tensors, or implement a globalised
Newton-Krylov with perturbation homotopy. Treat this as the central
open numerical task.

## The model in one paragraph (for orientation)

Binary asset $v\in\{0,1\}$ with prior 1/2; $K\geq 3$ informed groups
with Gaussian signals of precision $\tau_k$; CRRA utility
$U(W)=W^{1-\gamma}/(1-\gamma)$; **zero net supply, no noise traders**.
The price function is $P[i,j,l]$ on a $G\times G\times G$ grid of signal
realisations. The REE is the fixed point $P=\Phi(P)$ where $\Phi$ runs,
for each grid point: (1) extract each agent's 2D price slice, (2) trace
the level set $P=p$ via a 2-pass root-find (sweep one axis on grid,
solve the other off grid), (3) integrate the prior signal density along
that contour, (4) Bayes' rule with the agent's own signal, (5) market
clearing. Under CARA the contour is a straight line in log-odds space
and reveals $T^\star=\sum\tau_k u_k$ exactly (FR); under any non-CARA
CRRA the contour is curved and only reveals $T^\star$ partially (PR).
The "knife-edge" claim is that 1−$R^2$ jumps from 0 at $\gamma=\infty$
to strictly positive for any finite $\gamma$.

Default parameters when not specified: $\tau=2$, $\gamma=0.5$ (CRRA) or
$\gamma=100$ (CARA proxy), $W=1$, $K=3$, $G\in\{5\text{ debug},
20\text{ paper}, 100\text{+ contour figure}\}$, $u\in[-4,4]$, Anderson
window $m=6\text{–}8$, Picard damping $\alpha\in[0.15, 0.3]$.

## Git workflow

- Develop on the branch named in the session brief (currently
  `claude/add-claude-documentation-AC2lV`); do not push to `main`
  without explicit instruction.
- The repo allowed by the GitHub MCP tools is `mhpbreugem/rezn` —
  no other repos.
- Commit `.tex` + matching rebuilt `.pdf` together, including
  `main.pdf` when the paper itself changes, so reviewers can read the
  PDF straight off `main`.
- The committed history shows fine-grained polish-pass commits
  (`d4f2e2e`, `99c8f6b`, `747f2bd`, …) with terse subjects describing
  the substantive change ("Tighten proofs", "Add overnight paper
  review", "Apply PAPER_ADDITIONS.md"). Match that style.
