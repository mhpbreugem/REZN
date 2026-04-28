"""
No-learning equilibrium at G=5. No iteration, no contour, no learning from price.

For each grid triple (u_i, u_j, u_l):
  - Each agent k uses prior posterior mu_k = Lambda(tau * u_k) (own signal only).
  - Market clearing: sum_k x_k(mu_k, p) = 0  ->  p[i,j,l].
Then: weighted regression of logit(p) on T* = tau*(u_i+u_j+u_l), where weights are
the ex ante signal density 0.5 * (prod f_1(u_k) + prod f_0(u_k)).

Reports parameters, the full 5x5x5 price array for the canonical case, the (gamma, tau)
sweep table of 1-R^2, and the corresponding T* / logit(p) regression for the canonical case.
"""

import numpy as np
from scipy.optimize import brentq

# ---------- Parameters ----------
G = 5
K = 3
UMAX = 2.0
W = 1.0
U_GRID = np.linspace(-UMAX, UMAX, G)  # [-2, -1, 0, 1, 2]


# ---------- Model primitives ----------
def Lam(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

def logit(p):
    return np.log(p / (1.0 - p))

def f_v(u, tau, v):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2.0 * (u - mean) ** 2)

def crra_demand(mu, p, gamma, W=1.0):
    z = (logit(mu) - logit(p)) / gamma
    R = np.exp(np.clip(z, -50, 50))
    return W * (R - 1.0) / ((1.0 - p) + R * p)

def cara_clear(mus, gamma=1.0):
    """Closed form: logit(p) = mean(logit(mu_k)) under CARA."""
    return Lam(np.mean(logit(np.clip(mus, 1e-9, 1 - 1e-9))))

def crra_clear(mus, gamma):
    def Z(p):
        return sum(crra_demand(mu, p, gamma, W) for mu in mus)
    lo, hi = 1e-7, 1 - 1e-7
    return brentq(Z, lo, hi, xtol=1e-14)


# ---------- Build the no-learning price array ----------
def build_no_learning_P(gamma, tau, kind):
    P = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                mus = [Lam(tau * U_GRID[i]), Lam(tau * U_GRID[j]), Lam(tau * U_GRID[l])]
                if kind == "cara":
                    P[i, j, l] = cara_clear(mus)
                else:
                    P[i, j, l] = crra_clear(mus, gamma)
    return P


# ---------- 1-R^2 on the grid (weighted) ----------
def deficit(P, tau):
    Y, X, Wts = [], [], []
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = P[i, j, l]
                if not (1e-9 < p < 1 - 1e-9):
                    continue
                T = tau * (U_GRID[i] + U_GRID[j] + U_GRID[l])
                w = 0.5 * (
                    f_v(U_GRID[i], tau, 1) * f_v(U_GRID[j], tau, 1) * f_v(U_GRID[l], tau, 1)
                    + f_v(U_GRID[i], tau, 0) * f_v(U_GRID[j], tau, 0) * f_v(U_GRID[l], tau, 0)
                )
                Y.append(logit(p)); X.append(T); Wts.append(w)
    Y, X, Wts = np.array(Y), np.array(X), np.array(Wts)
    Wts = Wts / Wts.sum()
    Ybar = (Wts * Y).sum(); Xbar = (Wts * X).sum()
    cov = (Wts * (Y - Ybar) * (X - Xbar)).sum()
    vy = (Wts * (Y - Ybar) ** 2).sum()
    vx = (Wts * (X - Xbar) ** 2).sum()
    R2 = cov ** 2 / (vy * vx) if vy * vx > 0 else 0.0
    slope = cov / vx if vx > 0 else 0.0
    intercept = Ybar - slope * Xbar
    return 1.0 - R2, slope, intercept, len(Y)


# ---------- Reporting ----------
def report_parameters():
    print("=" * 72)
    print("NO-LEARNING EQUILIBRIUM AT G = 5")
    print("=" * 72)
    print(f"  K (agents)            : {K}")
    print(f"  G (grid points)       : {G}")
    print(f"  Signal grid u_k       : {U_GRID.tolist()} (spacing {U_GRID[1] - U_GRID[0]:.2f})")
    print(f"  Wealth W              : {W}")
    print(f"  Asset                 : binary v in {{0,1}}, prior 1/2")
    print(f"  Signal model          : s_k = v + eps_k,  eps_k ~ N(0, 1/tau)")
    print(f"                          centered u_k = s_k - 1/2")
    print(f"  CRRA demand           : x_k = W(R-1)/((1-p)+R*p),  R = exp((logit mu - logit p)/gamma)")
    print(f"  CARA demand (gamma->inf): x_k = (logit mu - logit p)/gamma  (linear)")
    print(f"  Posterior (no learning): mu_k = Lambda(tau * u_k)   [uses own signal only]")
    print(f"  Market clearing       : sum_k x_k(mu_k, p) = 0   (zero supply, no noise)")
    print(f"  Sufficient statistic  : T* = tau * sum_k u_k")
    print(f"  Weights for 1-R^2     : w(u) = 0.5 * (prod f_1(u_k) + prod f_0(u_k))")
    print(f"  Total triples         : G^3 = {G**3}  (no symmetrization in this script)")
    print()


def print_price_slice(P, i, label):
    print(f"  Slice P[i={i}, :, :]  (own signal u_1 = {U_GRID[i]:+.0f}, label={label})")
    print(f"            u_3 =  {'   '.join(f'{u:+.0f}' for u in U_GRID)}")
    for j in range(G):
        row = "  ".join(f"{P[i, j, l]:.4f}" for l in range(G))
        print(f"   u_2 = {U_GRID[j]:+.0f}  {row}")
    print()


def report_canonical_case(gamma, tau, kind):
    print("-" * 72)
    label = "CARA" if kind == "cara" else f"CRRA (gamma={gamma})"
    print(f"CANONICAL CASE: {label}, tau = {tau}")
    print("-" * 72)
    P = build_no_learning_P(gamma, tau, kind)

    # Slices for own-signal u_1 = -2, 0, +2
    for i in [0, 2, 4]:
        print_price_slice(P, i, label)

    R2def, slope, intercept, n = deficit(P, tau)
    print(f"  Weighted regression: logit(p) = {intercept:+.4f} + {slope:+.4f} * T*")
    print(f"  1 - R^2 = {R2def:.6f}   (over {n} interior triples)")

    # Reference realization (1, -1, 1)
    i_p1 = int(np.argmin(np.abs(U_GRID - 1.0)))
    i_m1 = int(np.argmin(np.abs(U_GRID + 1.0)))
    p_ref = P[i_p1, i_m1, i_p1]
    Tref = tau * (U_GRID[i_p1] + U_GRID[i_m1] + U_GRID[i_p1])
    print(f"  Reference realization (u1,u2,u3) = (+1,-1,+1): p = {p_ref:.6f}, T* = {Tref:.4f}")

    # CARA closed-form check at this realization
    if kind == "cara":
        p_analytic = Lam(Tref / 3)
        print(f"    CARA closed form Lambda(T*/K) = Lambda({Tref/3:.4f}) = {p_analytic:.6f}  "
              f"(diff {p_ref - p_analytic:+.2e})")
    print()
    return P, R2def


def report_sweep():
    print("-" * 72)
    print("SWEEP: 1 - R^2 across (gamma, tau) at G = 5")
    print("-" * 72)
    gammas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0]
    taus = [0.5, 1.0, 2.0, 3.0]
    print("  gamma | " + " ".join(f"tau={t:>4}" for t in taus))
    print("  " + "-" * (8 + 9 * len(taus)))
    for g in gammas:
        kind = "cara" if g >= 100 else "crra"
        row = []
        for t in taus:
            P = build_no_learning_P(g, t, kind)
            R2d, _, _, _ = deficit(P, t)
            row.append(f"{R2d:7.5f}")
        tag = "CARA" if g >= 100 else f"{g:>4.2f}"
        print(f"  {tag:>5} | " + " ".join(f"{v:>8}" for v in row))
    print()


if __name__ == "__main__":
    report_parameters()

    # Canonical: CARA and CRRA(0.5) at tau=2
    P_cara, R2_cara = report_canonical_case(gamma=1.0, tau=2.0, kind="cara")
    P_crra, R2_crra = report_canonical_case(gamma=0.5, tau=2.0, kind="crra")

    print("-" * 72)
    print(f"NET 1-R^2 (CRRA - CARA) at canonical (gamma=0.5, tau=2.0):  {R2_crra - R2_cara:+.6f}")
    print()

    report_sweep()
