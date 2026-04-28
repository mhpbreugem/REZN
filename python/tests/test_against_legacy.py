"""Regression: rezn_n128.phi_map (f128) ≡ legacy_k3.rezn_het._phi_map (f64).

Cast both to f64 and assert max|diff| < 1e-12 across several G's and a
moderate-heterogeneity (γ, τ) configuration.
"""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
sys.path.insert(0, PY)
sys.path.insert(0, os.path.join(PY, "legacy_k3"))

import rezn_het as rh                                    # noqa: E402
import rezn_n128                                          # noqa: E402


def _check(G, taus, gammas, Ws, umax, tol):
    u64 = np.linspace(-umax, umax, G)
    P64 = rh._nolearning_price(u64, taus, gammas, Ws)
    Phi64 = rh._phi_map(P64, u64, taus, gammas, Ws)

    u128 = u64.astype(np.float128)
    P128 = P64.astype(np.float128)
    taus128 = np.asarray(taus, dtype=np.float128)
    gammas128 = np.asarray(gammas, dtype=np.float128)
    Ws128 = np.asarray(Ws, dtype=np.float128)
    Phi128 = rezn_n128.phi_map(P128, u128, taus128, gammas128, Ws128)

    diff = float(np.abs(Phi128.astype(np.float64) - Phi64).max())
    print(f"G={G:3d}  max|diff|={diff:.3e}  (tol {tol:.0e})", flush=True)
    assert diff < tol, f"G={G}: max|diff|={diff:.3e} ≥ tol {tol:.0e}"


def test_phi_matches_legacy():
    TAUS = np.array([2.0, 3.0, 5.0])
    GAMMAS = np.array([0.5, 2.0, 8.0])
    WS = np.array([1.0, 1.0, 1.0])
    for G in (5, 7, 11):
        _check(G, TAUS, GAMMAS, WS, umax=2.0, tol=1e-12)


def test_nolearning_seed_matches_legacy():
    TAUS = np.array([1.0, 2.0, 4.0])
    GAMMAS = np.array([1.0, 1.0, 1.0])
    WS = np.array([1.0, 1.0, 1.0])
    G = 7
    umax = 2.0
    u64 = np.linspace(-umax, umax, G)
    P_ref = rh._nolearning_price(u64, TAUS, GAMMAS, WS)
    P_new = rezn_n128.nolearning_seed(
        u64.astype(np.float128),
        TAUS.astype(np.float128),
        GAMMAS.astype(np.float128),
        WS.astype(np.float128),
    )
    diff = float(np.abs(P_new.astype(np.float64) - P_ref).max())
    print(f"nolearning  max|diff|={diff:.3e}")
    assert diff < 1e-12


def test_picard_homogeneous():
    """Picard reduces residual on homogeneous γ=2 τ=2 G=5 with full step.

    Convergence-to-machine-precision is a separate (slow) test; here we
    just verify the loop runs end-to-end and the residual drops
    substantially from the initial seed.
    """
    TAUS = np.array([2.0, 2.0, 2.0])
    GAMMAS = np.array([2.0, 2.0, 2.0])
    WS = np.array([1.0, 1.0, 1.0])
    G = 5
    umax = 2.0
    u128 = np.linspace(-umax, umax, G).astype(np.float128)
    taus128 = TAUS.astype(np.float128)
    gammas128 = GAMMAS.astype(np.float128)
    Ws128 = WS.astype(np.float128)
    P0 = rezn_n128.nolearning_seed(u128, taus128, gammas128, Ws128)

    def Phi(P):
        return rezn_n128.phi_map(P, u128, taus128, gammas128, Ws128)

    F0 = float(np.abs(P0 - Phi(P0)).max())
    out = rezn_n128.picard_adaptive(
        P0, Phi, alpha0=1.0, alpha_max=1.0, alpha_min=0.5,
        maxiter=80, abstol=np.float128("1e-30"),    # don't early-exit
        slope_window=200, mono_relax=400, stall_window=200,
        log=lambda m: None,
    )
    print(f"homogeneous picard: F0={F0:.3e}  best Finf={out['Finf_best']:.3e}")
    assert out["Finf_best"] < 0.01 * F0, "picard did not reduce residual ≥ 100×"


if __name__ == "__main__":
    test_phi_matches_legacy()
    test_nolearning_seed_matches_legacy()
    test_picard_homogeneous()
    print("ALL TESTS PASSED")
