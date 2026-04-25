"""Per-cell diagnostic for piece F bug.

Compares analytic phi_tangent_at_cell vs FD at every cell, with two test
directions:
  V_full   = random Gaussian (exercises every input cell)
  V_canon  = e_{i0,j0,l0} (single-coord perturbation)

Reports max error, top-K worst cells, and breakdown by whether the cell
is on a slice that contains the perturbed coord.
"""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import pchip_jacobian as pj


def _phi(P, u, taus, gammas, Ws):
    return rp._phi_map_pchip(P, u,
                              np.asarray(taus, float),
                              np.asarray(gammas, float),
                              np.asarray(Ws, float))


def fd_phi_dot(P, V, u, taus, gammas, Ws, eps=1e-6):
    # Clip the perturbed P (matches what Newton applies)
    Pp = np.clip(P + eps * V, 1e-9, 1 - 1e-9)
    Pm = np.clip(P - eps * V, 1e-9, 1 - 1e-9)
    return (_phi(Pp, u, taus, gammas, Ws) - _phi(Pm, u, taus, gammas, Ws)) / (2 * eps)


def analytic_phi_dot(P, V, u, taus, gammas, Ws):
    Phi_P = _phi(P, u, taus, gammas, Ws)
    G = P.shape[0]
    out = np.zeros_like(P)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                out[i, j, l] = pj.phi_tangent_at_cell(
                    P, V, i, j, l, Phi_P[i, j, l],
                    u, taus, gammas, Ws)
    return out


def main():
    G = 5
    UMAX = 2.0
    u = np.linspace(-UMAX, UMAX, G)
    taus = np.array([3.0, 3.0, 3.0])
    gammas = np.array([3.0, 3.0, 3.0])
    Ws = np.array([1.0, 1.0, 1.0])

    res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                                 maxiters=300, abstol=1e-13, alpha=1.0)
    P0 = res["P_star"]
    rng = np.random.default_rng(7)
    P = np.clip(P0 + 0.01 * rng.standard_normal(P0.shape), 1e-9, 1 - 1e-9)

    # ---- Test 1: random V ---------------------------------------------
    V = rng.standard_normal(P.shape) * 1e-3
    ana = analytic_phi_dot(P, V, u, taus, gammas, Ws)
    fd = fd_phi_dot(P, V, u, taus, gammas, Ws)
    err = np.abs(ana - fd)
    rel = err / np.maximum(1e-12, np.abs(fd))
    print(f"Random V: max_abs_err={err.max():.3e} max_rel_err={rel.max():.3e}")
    flat = np.argsort(err.ravel())[::-1][:10]
    for idx in flat:
        i, j, l = np.unravel_index(idx, P.shape)
        print(f"  cell ({i},{j},{l}): ana={ana[i,j,l]:+.4e} fd={fd[i,j,l]:+.4e} "
              f"err={err[i,j,l]:.3e}")

    # ---- Test 2: canonical V on a single slice cell -------------------
    print("\nSingle-coord V perturbations:")
    for i0, j0, l0 in [(2, 2, 2), (0, 2, 2), (2, 0, 2), (2, 2, 0),
                         (4, 2, 2), (2, 4, 2), (2, 2, 4)]:
        V = np.zeros_like(P)
        V[i0, j0, l0] = 1.0
        ana = analytic_phi_dot(P, V, u, taus, gammas, Ws)
        fd  = fd_phi_dot(P, V, u, taus, gammas, Ws, eps=1e-7)
        err = np.abs(ana - fd)
        rel = err / np.maximum(1e-12, np.abs(fd))
        # Only nonzero cells should be those touching slice
        worst_idx = err.argmax()
        ii, jj, ll = np.unravel_index(worst_idx, P.shape)
        on_slice = (ii == i0) or (jj == j0) or (ll == l0)
        print(f"  V@({i0},{j0},{l0}): max_err={err.max():.3e} "
              f"worst@({ii},{jj},{ll}) on_slice={on_slice} "
              f"ana={ana[ii,jj,ll]:+.4e} fd={fd[ii,jj,ll]:+.4e}")
        # For the worst cell, decompose: is it the slice tangent or the
        # p_obs tangent that's wrong?
        if err.max() > 1e-4:
            # Build V_slice_only: zero V[ii,jj,ll]
            V_slice = V.copy()
            V_slice[ii, jj, ll] = 0.0
            # Build V_pobs_only: only V[ii,jj,ll]
            V_pobs = np.zeros_like(P)
            V_pobs[ii, jj, ll] = V[ii, jj, ll]
            ana_s = analytic_phi_dot(P, V_slice, u, taus, gammas, Ws)[ii, jj, ll]
            ana_p = analytic_phi_dot(P, V_pobs, u, taus, gammas, Ws)[ii, jj, ll]
            fd_s = fd_phi_dot(P, V_slice, u, taus, gammas, Ws, eps=1e-7)[ii, jj, ll]
            fd_p = fd_phi_dot(P, V_pobs, u, taus, gammas, Ws, eps=1e-7)[ii, jj, ll]
            print(f"     decomp: slice ana={ana_s:+.4e} fd={fd_s:+.4e} "
                  f"err={abs(ana_s-fd_s):.2e}")
            print(f"     decomp: p_obs ana={ana_p:+.4e} fd={fd_p:+.4e} "
                  f"err={abs(ana_p-fd_p):.2e}")
            print(f"     sum:   ana={ana_s+ana_p:+.4e} fd={fd_s+fd_p:+.4e}")
            print(f"     joint: ana={ana[ii,jj,ll]:+.4e} fd={fd[ii,jj,ll]:+.4e}")


if __name__ == "__main__":
    main()
