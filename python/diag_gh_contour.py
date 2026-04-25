"""Gauss-Hermite contour vs grid-edge PCHIP sum: which discretization
limits Φ accuracy?

Hypothesis: the production kernel sums f_v · f_v at PCHIP root crossings on
each grid row/column. For a fixed converged P, that's a discrete Riemann sum
with O(1/G²) bias. Gauss-Hermite quadrature against the Gaussian signal
density gives spectral convergence — at fixed G, ‖Φ_GH(P) − Φ_pchip(P)‖∞
should be much smaller than the production residual floor IF the contour
discretization is what's stuck.

This script computes Φ both ways on a cached converged P and reports the
gap. Interpretation:
  gap ≪ Finf : contour discretization is the bottleneck — implement GH in
              production to push Finf below the current floor.
  gap ≈ Finf : contour is fine; the floor comes from the price-tensor
              representation or the linear solve, not from the integral.
  gap ≫ Finf : the two methods disagree on the equilibrium itself; the
              cached P is converged for production but is NOT a fixed point
              of the GH operator. Need to re-solve under GH to compare.
"""
from __future__ import annotations
import pickle
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import rezn_pchip as rp
import rezn_het as rh


CACHE_PKL = "/home/user/REZN/python/pchip_G11u20_cache.pkl"


# ----- logit helpers ---------------------------------------------------------

_EPS = 1e-15

def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


# ----- GH-based contour sum --------------------------------------------------

def _gh_contour_sum(slice_, u, tau_A, tau_B, p_obs, n_nodes):
    """Two-pass Gauss-Hermite contour integral, mirroring the production
    2-pass averaging.

    A_v ≈ ½ [ Σ_n w_n f_v(u_b*(u_a^n))    (parametrized by u_a)
             + Σ_n w_n f_v(u_a*(u_b^n)) ] (parametrized by u_b)
    where w_n absorb f_v(·) — the GH weight matches the signal density of
    the integrating axis under state v. No Jacobian factor (matches the
    production pseudo-likelihood form).
    """
    G = len(u)
    L = _logit(slice_)
    lp = float(_logit(np.array([p_obs]))[0])

    # 2D off-grid evaluation: PCHIP along axis-0 for each column,
    # PCHIP along axis-1 for each row.
    col_pchips = [PchipInterpolator(u, L[:, b]) for b in range(G)]
    row_pchips = [PchipInterpolator(u, L[a, :]) for a in range(G)]

    xi, w_raw = hermegauss(n_nodes)              # weight e^{-x²/2}
    w = w_raw / np.sqrt(2 * np.pi)               # so Σ w = 1

    A = np.zeros(2)

    def root_uy(L_at_x_func, u_lo, u_hi):
        """Find u_y in [u_lo, u_hi] such that L_at_x_func(u_y) = lp.
        Returns None if no sign change."""
        f_lo = float(L_at_x_func(u_lo)) - lp
        f_hi = float(L_at_x_func(u_hi)) - lp
        if f_lo * f_hi > 0:
            return None
        return brentq(lambda y: float(L_at_x_func(y)) - lp,
                       u_lo, u_hi, xtol=1e-14)

    # ---- Pass A: parametrize by u_a, GH against f_v(u_a, tau_A) -----------
    for v in (0, 1):
        mu_v = v - 0.5
        scale = 1.0 / np.sqrt(tau_A)
        for n in range(n_nodes):
            u_a = mu_v + xi[n] * scale
            if u_a < u[0] or u_a > u[-1]:
                continue
            # Build off-grid row at u_a from columns
            row_at_ua = np.array([float(c(u_a)) for c in col_pchips])
            row_pchip = PchipInterpolator(u, row_at_ua)
            u_b = root_uy(row_pchip, u[0], u[-1])
            if u_b is None:
                continue
            f_v_b = (rh._f0(u_b, tau_B) if v == 0 else rh._f1(u_b, tau_B))
            A[v] += w[n] * float(f_v_b)

    # ---- Pass B: parametrize by u_b, GH against f_v(u_b, tau_B) -----------
    A_passB = np.zeros(2)
    for v in (0, 1):
        mu_v = v - 0.5
        scale = 1.0 / np.sqrt(tau_B)
        for n in range(n_nodes):
            u_b = mu_v + xi[n] * scale
            if u_b < u[0] or u_b > u[-1]:
                continue
            col_at_ub = np.array([float(r(u_b)) for r in row_pchips])
            col_pchip = PchipInterpolator(u, col_at_ub)
            u_a = root_uy(col_pchip, u[0], u[-1])
            if u_a is None:
                continue
            f_v_a = (rh._f0(u_a, tau_A) if v == 0 else rh._f1(u_a, tau_A))
            A_passB[v] += w[n] * float(f_v_a)

    return 0.5 * (A[0] + A_passB[0]), 0.5 * (A[1] + A_passB[1])


# ----- Φ_GH map --------------------------------------------------------------

def _phi_at_cell_gh(P, u, taus, gammas, Ws, i, j, l, n_nodes):
    """Φ(P)[i,j,l] computed via Gauss-Hermite contour."""
    p_obs = float(P[i, j, l])
    mus = np.empty(3)
    for ag in range(3):
        if ag == 0:
            slice_ = P[i, :, :]
            u_own, tau_own = u[i], taus[0]
            tau_A, tau_B = taus[1], taus[2]
        elif ag == 1:
            slice_ = P[:, j, :]
            u_own, tau_own = u[j], taus[1]
            tau_A, tau_B = taus[0], taus[2]
        else:
            slice_ = P[:, :, l]
            u_own, tau_own = u[l], taus[2]
            tau_A, tau_B = taus[0], taus[1]
        A0, A1 = _gh_contour_sum(slice_, u, tau_A, tau_B, p_obs, n_nodes)
        g0 = rh._f0(u_own, tau_own)
        g1 = rh._f1(u_own, tau_own)
        den = g0 * A0 + g1 * A1
        if den <= 0:
            mus[ag] = 1.0 / (1.0 + np.exp(-tau_own * u_own))
        else:
            mus[ag] = g1 * A1 / den
    return rh._clear_price(mus, gammas, Ws)


# ----- diagnostic main -------------------------------------------------------

def _residual(P, u, taus, gammas, Ws):
    Phi = rp._phi_map_pchip(P, u, taus, gammas, Ws)
    return float(np.abs(P - Phi).max())


def main():
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {len(cache)} cache entries", flush=True)

    # For each cache entry, evaluate Φ_GH at the K cells with the largest
    # production residual (where contour-discretization noise should bite
    # hardest if it's the bottleneck) and at K random cells (for an
    # unbiased gauge across the grid).
    K_TOP, K_RAND = 5, 5
    rng = np.random.default_rng(0)
    print(f"\nFor each config: GH at {K_TOP} worst-residual cells + "
          f"{K_RAND} random cells", flush=True)
    print(f"{'τ':>5} {'γ':>6}  {'Finf_pchip':>11}  "
          f"{'gap_GH30':>11}  {'gap_GH50':>11}  {'gap_GH70':>11}",
          flush=True)
    print("-" * 76, flush=True)

    for e in cache:
        taus = np.asarray(e["taus"], float)
        gammas = np.asarray(e["gammas"], float)
        P = np.clip(e["P_star"], 1e-9, 1 - 1e-9)
        umax = float(e.get("umax", 2.0))
        G = P.shape[0]
        u = np.linspace(-umax, umax, G)
        Ws = np.array([1.0, 1.0, 1.0])

        Phi_pchip = rp._phi_map_pchip(P, u, taus, gammas, Ws)
        residual = np.abs(P - Phi_pchip)
        finf_pchip = float(residual.max())

        # Cells to sample
        flat_top = np.argsort(residual.ravel())[::-1][:K_TOP]
        flat_rand = rng.choice(G ** 3, size=K_RAND, replace=False)
        cells = np.unique(np.concatenate([flat_top, flat_rand]))

        gaps = {30: 0.0, 50: 0.0, 70: 0.0}
        for n_nodes in (30, 50, 70):
            for flat in cells:
                i, j, l = np.unravel_index(int(flat), P.shape)
                phi_gh = _phi_at_cell_gh(P, u, taus, gammas, Ws,
                                          i, j, l, n_nodes)
                d = abs(phi_gh - Phi_pchip[i, j, l])
                if d > gaps[n_nodes]:
                    gaps[n_nodes] = d

        print(f"{taus[0]:5.2f} {gammas[0]:6.2f}  {finf_pchip:11.3e}  "
              f"{gaps[30]:11.3e}  {gaps[50]:11.3e}  {gaps[70]:11.3e}",
              flush=True)


if __name__ == "__main__":
    main()
