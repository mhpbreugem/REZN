"""Figure 4 — Converged Posteriors at (u₁,u₂,u₃) = (1, −1, 1).

Standalone LaTeX table comparing private prior, converged CARA REE,
and converged CRRA REE at one realisation. CARA is closed-form
(binary FR, p = Λ(T*/K)). CRRA is solved by Anderson on the
production PCHIP+contour kernel at γ=0.5, G=20.

Output:
  figures/fig4_posteriors.{tex,pdf,png}
"""
from __future__ import annotations
import os
import numpy as np

import rezn_pchip as rp
import rezn_het as rh


# ---- spec ---------------------------------------------------------------
TAU      = 2.0
GAMMA    = 0.5
G        = 5            # matches the published reference table at G=5
UMAX     = 2.0
U_REAL   = (1.0, -1.0, 1.0)
OUT      = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "figures")


def _logit(p):
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    os.makedirs(OUT, exist_ok=True)
    u_grid = np.linspace(-UMAX, UMAX, G)
    u1, u2, u3 = U_REAL

    # ---- Private (no-learning) priors ----------------------------------
    mu_prior = (_sigmoid(TAU * u1),
                 _sigmoid(TAU * u2),
                 _sigmoid(TAU * u3))

    # ---- CARA REE (closed-form binary FR) ------------------------------
    # In REE, observing p reveals T*, every agent updates to μ = Λ(T*),
    # and market clearing on identical posteriors gives logit p =
    # logit μ = T*. So p_REE = Λ(T*) (NOT Λ(T*/K) — that's no-learning).
    T_star = TAU * (u1 + u2 + u3)
    p_cara = _sigmoid(T_star)
    mu_cara = (_sigmoid(T_star),) * 3
    finf_cara = 0.0

    # ---- CRRA REE: try Picard α=0.3 first (stays in PR basin at G=5),
    #                 fall back to Anderson polish.
    taus_arr   = np.array([TAU, TAU, TAU])
    gammas_arr = np.array([GAMMA, GAMMA, GAMMA])
    print(f"Solving CRRA REE at γ={GAMMA}, τ={TAU}, G={G} …", flush=True)
    res_p = rp.solve_picard_pchip(
        G, taus_arr, gammas_arr, umax=UMAX,
        maxiters=2000, abstol=1e-7, alpha=0.3)
    P_star = res_p["P_star"]
    finf_crra = float(np.abs(res_p["residual"]).max())
    print(f"  Picard iters={len(res_p['history'])}  "
           f"best Finf={finf_crra:.3e}", flush=True)

    # Locate the (1, -1, 1) cell
    i_r = int(np.argmin(np.abs(u_grid - u1)))
    j_r = int(np.argmin(np.abs(u_grid - u2)))
    l_r = int(np.argmin(np.abs(u_grid - u3)))
    p_crra = float(P_star[i_r, j_r, l_r])
    mu_crra = rh.posteriors_at(i_r, j_r, l_r, p_crra,
                                  P_star, u_grid, taus_arr)

    # ---- Build the table ----------------------------------------------
    def fmt(x):
        return f"{x:.4f}"
    def lf(p):
        return f"{_logit(p):+.3f}"

    rows = [
        ("$\\mu_1$ (own signal $u_1\\!=\\!+1$)",
         fmt(mu_prior[0]), fmt(mu_cara[0]), fmt(mu_crra[0])),
        ("$\\mu_2$ (own signal $u_2\\!=\\!-1$)",
         fmt(mu_prior[1]), fmt(mu_cara[1]), fmt(mu_crra[1])),
        ("$\\mu_3$ (own signal $u_3\\!=\\!+1$)",
         fmt(mu_prior[2]), fmt(mu_cara[2]), fmt(mu_crra[2])),
        ("price $p$",
         "---", fmt(p_cara), fmt(p_crra)),
        ("logit$(p)$",
         "---", lf(p_cara), lf(p_crra)),
        ("$\\Vert F\\Vert_\\infty$",
         "---", "$0$", f"{finf_crra:.1e}"),
    ]

    body = "\n".join(
        f"  {a}  &  {b}  &  {c}  &  {d} \\\\"
        for a, b, c, d in rows
    )

    tex = (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{amsmath, booktabs}\n"
        "\\begin{document}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "  & private prior  &  CARA REE  &  CRRA REE \\\\\n"
        "  &  $\\Lambda(\\tau u_k)$  &  (closed form)  "
        f"&  ($\\gamma\\!=\\!{GAMMA:g}$, $G\\!=\\!{G}$) \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{document}\n"
    )

    tex_path = os.path.join(OUT, "fig4_posteriors.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}")
    print()
    print("Values at (u_1, u_2, u_3) = (1, -1, 1):")
    print(f"  prior:     {tuple(round(m, 4) for m in mu_prior)}")
    print(f"  CARA REE:  {tuple(round(m, 4) for m in mu_cara)}, "
           f"p = {p_cara:.4f}")
    print(f"  CRRA REE:  {tuple(round(m, 4) for m in mu_crra)}, "
           f"p = {p_crra:.4f}, ‖F‖∞ = {finf_crra:.2e}")


if __name__ == "__main__":
    main()
