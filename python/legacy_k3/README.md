# legacy_k3/

K=3 reference implementations preserved for verification and regression testing.
Active development is in `python/rezn_n128/`.

| file                | role                                                              |
|---------------------|-------------------------------------------------------------------|
| `rezn_het.py`       | numba float64 K=3 kernel — fast reference for the Φ map           |
| `rezn_lin128.py`    | partial pure-numpy float128 port (superseded by `rezn_n128`)      |
| `rezn_pchip.py`     | PCHIP-kernel variant of `rezn_het`                                |
| `pchip_jacobian.py` | analytic-style Jacobian-vector product against `rezn_pchip.Φ`     |

`rezn_n128` matches `rezn_het`'s linear-interp Φ to machine precision; the
regression test in `python/tests/test_against_legacy.py` enforces that.

The `rezn_lin128.py` port had a bug at the `den ≤ 0` branch (returned 0.5
instead of `logistic(τ_own·u_own)`); `rezn_n128.posterior.agent_posterior`
fixes this — see the comment there.
