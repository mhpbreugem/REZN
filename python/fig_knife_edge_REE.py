"""Knife-edge figure with FULL rational-expectations equilibrium.

For each (γ, τ) we converge the REE on a G×G×G price tensor using the
production PCHIP+contour kernel (rezn_pchip.solve_anderson_pchip), then
compute 1−R² of logit(P) against T*=τ·Σu_k.

For a smooth curve we evaluate the regression with Gauss-Hermite (N=60)
quadrature against the prior weight ½(f₀³+f₁³): the converged P on the
G=15 grid is interpolated to GH nodes with a tensor-product PCHIP, and
the weighted 1−R² is summed there.

τ-continuation: solve high τ first then warm-start each next-lower τ
from the previous P*. (γ=1 is well-conditioned at every τ ∈ [0.1, 20]
so single-shot also works, but continuation halves the wall-clock.)

Starts with γ=1 (log utility); other γ values are easy to add by
extending the GAMMAS list.

Output:
  figures/fig_knife_edge_REE.{tex,pdf,png}
  figures/fig_knife_edge_REE_data.csv
"""
from __future__ import annotations
import os
import csv
import time
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.interpolate import PchipInterpolator
from numba import njit, prange

import rezn_pchip as rp


# ---- Spec ------------------------------------------------------------------
G            = 15
UMAX         = 2.0
GAMMAS       = [(1.0, "1.0", "red", "dashed", "very thick")]
TAUS         = np.logspace(np.log10(0.1), np.log10(8.0), 25)
N_GH         = 60
ANDERSON_M   = 8
MAXITERS     = 400
ABSTOL       = 1e-6
OUT          = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "figures")


# ---- GH quadrature on a converged P_star -----------------------------------
def _interp3d_pchip(P, u, U_eval):
    """Tensor-product PCHIP interpolation of 3D tensor P on grid u
    evaluated at U_eval (shape (n,3)). Returns (n,) array of P values.

    Done in three sequential 1D PCHIP fits: axis-0 first (one fit per
    (j,l)), then axis-1, then axis-2. Each fit costs G× a 1D PCHIP
    construct + evaluation; total O(G² · n + G · n + n)."""
    n = U_eval.shape[0]
    # axis 0: for each (j, l), build PCHIP over u → evaluate at U_eval[:,0]
    G0, G1, G2 = P.shape
    # Step 1: collapse axis 0
    P1 = np.empty((n, G1, G2))
    for j in range(G1):
        for l in range(G2):
            spline = PchipInterpolator(u, P[:, j, l], extrapolate=True)
            P1[:, j, l] = spline(U_eval[:, 0])
    # Step 2: collapse axis 1
    P2 = np.empty((n, G2))
    for l in range(G2):
        for ii in range(n):
            spline = PchipInterpolator(u, P1[ii, :, l], extrapolate=True)
            P2[ii, l] = spline(U_eval[ii, 1])
    # Step 3: collapse axis 2
    P3 = np.empty(n)
    for ii in range(n):
        spline = PchipInterpolator(u, P2[ii, :], extrapolate=True)
        P3[ii] = spline(U_eval[ii, 2])
    # Clip strictly inside (0, 1) to avoid log issues
    return np.clip(P3, 1e-12, 1 - 1e-12)


def _f0(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u + 0.5)**2)


def _f1(u, tau):
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau * 0.5 * (u - 0.5)**2)


def _one_minus_R2(P_star, u_grid, tau):
    """Weighted 1-R² on the discrete G×G×G grid with prior ½(f₀³+f₁³).

    GH-on-PCHIP gave essentially zero weight at low τ because most GH
    nodes land outside [-UMAX, UMAX] and were dropped. The grid sum
    integrates against the *same* prior, just on the cells the solver
    actually evaluated — fully stable and matches the no-learning
    figure's regression.
    """
    G = P_star.shape[0]
    P_clip = np.clip(P_star, 1e-12, 1 - 1e-12)
    Y = np.log(P_clip / (1 - P_clip)).ravel()
    Ts = np.empty(G ** 3)
    Ws = np.empty(G ** 3)
    f0_u = _f0(u_grid, tau)
    f1_u = _f1(u_grid, tau)
    k = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                Ts[k] = tau * (u_grid[i] + u_grid[j] + u_grid[l])
                Ws[k] = 0.5 * (f0_u[i] * f0_u[j] * f0_u[l]
                                 + f1_u[i] * f1_u[j] * f1_u[l])
                k += 1
    W   = Ws.sum()
    if W <= 0.0:
        return 0.0
    Y_m = (Ws * Y).sum() / W
    T_m = (Ws * Ts).sum() / W
    Syy = (Ws * (Y - Y_m) ** 2).sum()
    STT = (Ws * (Ts - T_m) ** 2).sum()
    SyT = (Ws * (Y - Y_m) * (Ts - T_m)).sum()
    if Syy <= 1e-30 or STT <= 1e-30:
        return 0.0
    return 1.0 - (SyT * SyT) / (Syy * STT)


# ---- Driver ----------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    u_grid = np.linspace(-UMAX, UMAX, G)

    results = {}
    for gamma, label, color, style, thick in GAMMAS:
        print(f"\n=== γ = {label} (REE) ===", flush=True)
        taus = np.asarray(TAUS, float)
        gammas = np.array([gamma, gamma, gamma])
        ones3  = np.array([1.0, 1.0, 1.0])
        vals = np.empty(len(taus))
        # Continuation: solve LOWEST τ first (cold start trivial), then
        # sweep up with warm-starts. High-τ saturation is the danger
        # zone — by the time we get there, P* is already in-basin.
        order = np.argsort(taus)
        last_P = None
        for idx in order:
            tau = float(taus[idx])
            tau3 = np.array([tau, tau, tau])
            t0 = time.time()
            res = rp.solve_anderson_pchip(
                G, tau3, gammas, umax=UMAX,
                maxiters=MAXITERS, abstol=ABSTOL,
                m_window=ANDERSON_M,
                P_init=last_P)
            P_star = res["P_star"]
            finf = float(np.abs(res["residual"]).max())
            # Skip cells where the solver clearly didn't converge
            if not np.isfinite(finf) or finf > 1e-2:
                print(f"  τ={tau:7.3f}  iters={len(res['history']):4d}  "
                       f"Finf={finf:.2e}  DIVERGED — keeping last_P, "
                       f"using R²=NaN", flush=True)
                vals[idx] = np.nan
                continue
            R2 = _one_minus_R2(P_star, u_grid, tau)
            vals[idx] = R2
            last_P = P_star
            print(f"  τ={tau:7.3f}  iters={len(res['history']):4d}  "
                   f"Finf={finf:.2e}  1-R²={R2:.5f}  "
                   f"{time.time()-t0:.1f}s", flush=True)
        results[gamma] = vals

    csv_path = os.path.join(OUT, "fig_knife_edge_REE_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau"] + [f"oneR2_g{g}" for g, *_ in GAMMAS])
        for it, tau in enumerate(TAUS):
            w.writerow([f"{tau:.6g}"]
                        + [f"{results[g][it]:.6g}" for g, *_ in GAMMAS])
    print(f"\nwrote {csv_path}", flush=True)

    addplots = []
    for gamma, label, color, style, thick in GAMMAS:
        coords = " ".join(f"({tau:.6g},{v:.6g})"
                           for tau, v in zip(TAUS, results[gamma])
                           if np.isfinite(v))
        addplots.append(
            f"\\addplot[{color}, {style}, {thick}] "
            f"coordinates {{{coords}}};\n"
            f"\\addlegendentry{{$\\gamma = {label}$ (REE)}};")
    # CARA reference (closed-form: REE = no-learning = FR)
    addplots.append(
        "\\addplot[black, dash dot, ultra thick] "
        "coordinates {(0.1,0) (8,0)};\n"
        "\\addlegendentry{$\\gamma = \\infty\\;(\\text{CARA})$};")

    tex = (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\definecolor{red}{rgb}{0.7,0.11,0.11}\n"
        "\\definecolor{blue}{rgb}{0.0,0.20,0.42}\n"
        "\\definecolor{green}{rgb}{0.11,0.35,0.02}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "    width=8cm, height=8cm,\n"
        "    xmode=log,\n"
        "    xmin=0.1, xmax=8,\n"
        "    ymin=-0.001, ymax=0.15,\n"
        "    xtick={0.1,0.2,0.5,1,2,5,8},\n"
        "    xticklabels={0.1,0.2,0.5,1,2,5,8},\n"
        "    ytick={0,0.05,0.1,0.15},\n"
        "    yticklabels={0,0.05,0.10,0.15},\n"
        "    xlabel={signal precision $\\tau$},\n"
        "    ylabel={$1 - R^2$},\n"
        "    title={REE: signal precision ($\\tau$)},\n"
        "    legend pos=north west,\n"
        "    legend style={fill=none, draw=none, font=\\footnotesize},\n"
        "    smooth,\n"
        "]\n"
        + "\n".join(addplots) + "\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{document}\n"
    )
    tex_path = os.path.join(OUT, "fig_knife_edge_REE.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
