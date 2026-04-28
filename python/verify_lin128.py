"""Sanity check: rezn_lin128._phi_map should match rezn_het._phi_map
on the same inputs to within ~1e-13 (float64 precision)."""
import time
import numpy as np
import rezn_het as rh
import rezn_lin128 as rl


for G in (5, 7, 11):
    UMAX = 2.0
    TAUS = np.array([2.0, 3.0, 5.0])
    GAMMAS = np.array([0.5, 2.0, 8.0])
    WS = np.array([1.0, 1.0, 1.0])
    u = np.linspace(-UMAX, UMAX, G)
    P = rh._nolearning_price(u, TAUS, GAMMAS, WS)

    t0 = time.time()
    Phi64 = rh._phi_map(P, u, TAUS, GAMMAS, WS)
    t64 = time.time() - t0

    t0 = time.time()
    P_f128 = P.astype(np.float128)
    u_f128 = u.astype(np.float128)
    TAUS128 = TAUS.astype(np.float128)
    GAMMAS128 = GAMMAS.astype(np.float128)
    WS128 = WS.astype(np.float128)
    Phi128 = rl._phi_map(P_f128, u_f128, TAUS128, GAMMAS128, WS128)
    t128 = time.time() - t0

    diff = np.abs(Phi128.astype(np.float64) - Phi64).max()
    print(f"G={G:3d}  numba f64: {t64*1000:7.1f}ms   pure-numpy f128: "
          f"{t128*1000:8.1f}ms   max|diff|={diff:.3e}",
          flush=True)
