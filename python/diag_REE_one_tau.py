"""Single-τ REE diagnostic: figure out why 1-R² collapses to zero."""
from __future__ import annotations
import numpy as np
import rezn_pchip as rp
import rezn_het as rh

G    = 15
UMAX = 2.0
TAU  = 2.0
GAMMA = 1.0

u = np.linspace(-UMAX, UMAX, G)
taus = np.array([TAU, TAU, TAU])
gammas = np.array([GAMMA, GAMMA, GAMMA])

# No-learning baseline
P0 = rh._nolearning_price(u, taus, gammas, np.array([1.0, 1.0, 1.0]))
print(f"No-learning P0 range: [{P0.min():.4f}, {P0.max():.4f}]")
print(f"No-learning logit(P0) range: "
       f"[{np.log(P0.min()/(1-P0.min())):.4f}, "
       f"{np.log(P0.max()/(1-P0.max())):.4f}]")
oneR2_nl = rh.one_minus_R2(P0, u, taus)
print(f"No-learning 1-R²:   {oneR2_nl:.6e}")

# REE via solve_picard_pchip (slower but more deterministic than Anderson)
print("\nSolving REE via Picard ...")
res = rp.solve_picard_pchip(G, taus, gammas, umax=UMAX,
                             maxiters=2000, abstol=1e-9, alpha=0.5)
P_star = res["P_star"]
print(f"  history len: {len(res['history'])}")
print(f"  best diff:   {min(res['history']):.3e}")
print(f"  last diff:   {res['history'][-1]:.3e}")
F = res["residual"]
print(f"  Finf:        {np.abs(F).max():.3e}")
print(f"  P_star range: [{P_star.min():.4f}, {P_star.max():.4f}]")
oneR2_ree = rh.one_minus_R2(P_star, u, taus)
print(f"REE 1-R²:           {oneR2_ree:.6e}")

# Diagnostic: spread of (Y, T) under prior weights
P_clip = np.clip(P_star, 1e-12, 1 - 1e-12)
Y = np.log(P_clip / (1 - P_clip))
T = np.empty_like(Y)
W = np.empty_like(Y)
def f0(u, t): return np.sqrt(t/(2*np.pi))*np.exp(-t/2*(u+0.5)**2)
def f1(u, t): return np.sqrt(t/(2*np.pi))*np.exp(-t/2*(u-0.5)**2)
f0u = f0(u, TAU); f1u = f1(u, TAU)
for i in range(G):
    for j in range(G):
        for l in range(G):
            T[i,j,l] = TAU*(u[i]+u[j]+u[l])
            W[i,j,l] = 0.5*(f0u[i]*f0u[j]*f0u[l] + f1u[i]*f1u[j]*f1u[l])
W_sum = W.sum()
Y_m   = (W * Y).sum() / W_sum
T_m   = (W * T).sum() / W_sum
Syy   = (W * (Y - Y_m)**2).sum()
STT   = (W * (T - T_m)**2).sum()
SyT   = (W * (Y - Y_m) * (T - T_m)).sum()
print(f"\n  Σw      = {W_sum:.4e}")
print(f"  Y mean  = {Y_m:.4e},  range = [{Y.min():.4e}, {Y.max():.4e}]")
print(f"  T mean  = {T_m:.4e},  range = [{T.min():.4e}, {T.max():.4e}]")
print(f"  Syy     = {Syy:.4e}")
print(f"  STT     = {STT:.4e}")
print(f"  SyT     = {SyT:.4e}")
print(f"  R²      = {(SyT*SyT)/(Syy*STT):.10f}")
print(f"  1-R²    = {1.0-(SyT*SyT)/(Syy*STT):.4e}")

# Also compare REE vs no-learning logit(P)
print(f"\n  ‖logit(P_REE) - logit(P_NL)‖∞ = "
       f"{np.abs(np.log(P_star/(1-P_star)) - np.log(P0/(1-P0))).max():.4e}")
print(f"  ‖P_REE - P_NL‖∞              = {np.abs(P_star - P0).max():.4e}")
