"""Debug Fig 8: trace V at one (i,j,l) for log utility."""
import numpy as np

TAU = 2.0
GAMMA = 1.0
G = 15
UMAX = 4.0
W0 = 1.0


def logit(p): return np.log(p/(1-p))


def crra_demand(mu, p, gamma):
    eps = 1e-12
    mu = max(eps, min(1-eps, mu))
    p  = max(eps, min(1-eps, p))
    R = np.exp((logit(mu) - logit(p)) / gamma)
    return (R - 1.0) / ((1.0 - p) + R * p)


def f0(u, t):
    return np.sqrt(t/(2*np.pi)) * np.exp(-t/2 * (u + 0.5)**2)


def f1(u, t):
    return np.sqrt(t/(2*np.pi)) * np.exp(-t/2 * (u - 0.5)**2)


def U(W, gamma):
    if W <= 1e-15: W = 1e-15
    if abs(gamma - 1) < 1e-12: return np.log(W)
    return W**(1-gamma) / (1-gamma)


def clear_price(mus, gamma):
    if abs(gamma-1) < 1e-12: return float(np.mean(mus))
    lo, hi = 0.002, 0.998
    f = lambda p: sum(crra_demand(m, p, gamma) for m in mus)
    fl = f(lo); fh = f(hi)
    for _ in range(120):
        m = 0.5*(lo+hi)
        fm = f(m)
        if fl*fm < 0: hi=m; fh=fm
        else: lo=m; fl=fm
    return 0.5*(lo+hi)


# At u=(0,0,0): all μ=0.5, p=0.5, x=0, V_cell = 0
u = np.linspace(-UMAX, UMAX, G)
print(f"τ={TAU} γ={GAMMA}")
print()

# Test cell (G//2, G//2, G//2) = (0, 0, 0)
i = G // 2
print(f"u[i]={u[i]:.3f}")
mu = 1/(1+np.exp(-TAU*u[i]))
print(f"μ={mu:.4f}")

# Sample some cells and print V_cell
total_w = 0; total_U = 0
for i in range(G):
    for j in range(G):
        for l in range(G):
            mu_i = 1/(1+np.exp(-TAU*u[i]))
            mu_j = 1/(1+np.exp(-TAU*u[j]))
            mu_l = 1/(1+np.exp(-TAU*u[l]))
            p = clear_price([mu_i, mu_j, mu_l], GAMMA)
            x1 = crra_demand(mu_i, p, GAMMA)
            W_v0 = 1 + x1*(0 - p)
            W_v1 = 1 + x1*(1 - p)
            w0 = 0.5 * f0(u[i], TAU) * f0(u[j], TAU) * f0(u[l], TAU)
            w1 = 0.5 * f1(u[i], TAU) * f1(u[j], TAU) * f1(u[l], TAU)
            U0 = U(W_v0, GAMMA); U1 = U(W_v1, GAMMA)
            total_U += w0*U0 + w1*U1
            total_w += w0 + w1

E_U = total_U / total_w
V = E_U - U(W0, GAMMA)
print(f"E[U_informed] = {E_U:.6f}")
print(f"U(W0)         = {U(W0, GAMMA):.6f}")
print(f"V             = {V:.6f}")

# Now let's also print the conditional KL at the no-learning cell:
# At the level where agent has μ_1 and faces equilibrium p:
# E[log W | trade optimally] - log W0 = KL(μ || p) = μ log(μ/p) + (1-μ) log((1-μ)/(1-p))
# But this is conditional on agent 1's signal AND on the equilibrium p (which depends on u_2, u_3).
# Let's compute E[KL(μ_1 || p)] over the prior:
total_w2 = 0; total_KL = 0
for i in range(G):
    for j in range(G):
        for l in range(G):
            mu_i = 1/(1+np.exp(-TAU*u[i]))
            mu_j = 1/(1+np.exp(-TAU*u[j]))
            mu_l = 1/(1+np.exp(-TAU*u[l]))
            p = clear_price([mu_i, mu_j, mu_l], GAMMA)
            mu_c = max(1e-12, min(1-1e-12, mu_i))
            p_c  = max(1e-12, min(1-1e-12, p))
            KL = mu_c*np.log(mu_c/p_c) + (1-mu_c)*np.log((1-mu_c)/(1-p_c))
            w0 = 0.5 * f0(u[i], TAU) * f0(u[j], TAU) * f0(u[l], TAU)
            w1 = 0.5 * f1(u[i], TAU) * f1(u[j], TAU) * f1(u[l], TAU)
            total_KL += (w0 + w1) * KL
            total_w2 += w0 + w1

print(f"E[KL(μ_1 || p)] over prior = {total_KL/total_w2:.6f}  (should equal V)")
