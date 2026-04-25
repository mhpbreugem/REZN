"""Pin down whether the discrepancy comes from clipped slice values."""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import rezn_het as rh
import pchip_jacobian as pj


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
    print(f"P range: [{P.min():.6e}, {P.max():.6e}]")
    print(f"# cells at min boundary (1e-9): {(P == 1e-9).sum()}")
    print(f"# cells at max boundary (1-1e-9): {(P == 1 - 1e-9).sum()}")
    # Cells extremely close to boundaries
    print(f"# cells P < 0.01:  {(P < 0.01).sum()}")
    print(f"# cells P > 0.99:  {(P > 0.99).sum()}")
    # Spot-check the slice at the bad cell
    j = 2
    slice_ = P[:, j, :]  # agent 1 slice
    print(f"\nAgent 1 slice (P[:,2,:]):")
    for a in range(G):
        for b in range(G):
            v = slice_[a, b]
            tag = ""
            if v < 1e-8: tag = " <<< near 0"
            elif v > 1 - 1e-8: tag = " <<< near 1"
            print(f"  ({a},{b}): {v:.10e}{tag}")

    print("\n‖P0‖ stats:")
    print(f"P0 range: [{P0.min():.6e}, {P0.max():.6e}]")


if __name__ == "__main__":
    main()
