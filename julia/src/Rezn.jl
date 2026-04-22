module Rezn
#
# =====================================================================
#   REE CONTOUR SOLVER — SYSTEM OF EQUATIONS AND UNKNOWNS
# =====================================================================
#
#   Model (see SESSION_SUMMARY.md):
#     v ∈ {0,1},  P(v=1) = 1/2
#     K = 3 agents, private signal  s_k = v + ε_k,  ε_k ~ N(0, 1/τ)
#     centered signal               u_k = s_k − 1/2
#     CRRA utility, wealth W, zero net supply, NO noise traders.
#
#   The price function is a 3-tensor P[i,j,l] defined on the grid
#     u_i, u_j, u_l ∈ u_grid  (i,j,l = 1,…,G).
#
#   --- Per cell (i,j,l), the REE system has 12G + 10 eqs/unknowns: ---
#
#   #   name            equation                                                   unknown        count
#   ----------------------------------------------------------------------------------------------------
#   D1  root-find A     P(·, u_jc, u_lc(·)) = p    [axis-A sweep, each agent]      u_lc           6G
#   D2  root-find B     P(·, u_jc(·), u_lc) = p    [axis-B sweep, each agent]      u_jc           6G
#   D3-D5 contour int.  A_v^(k) = ½[Σ_a f_v(u_a) f_v(u_lc) + Σ_b f_v(u_jc) f_v(u_b)]  A_v^(k)    6
#                       (v ∈ {0,1}, k ∈ {1,2,3})
#   D6  Bayes' rule     μ_k = f1_own · A1^(k) / (f0_own · A0^(k) + f1_own · A1^(k)) μ_k           3
#   E1  market clearing Σ_k x_k(μ_k, p) = 0                                         P[i,j,l]      1
#
#   Stacked across all G³ cells: 12G⁴ + 10G³ equations & unknowns.
#
#   --- This file solves a REDUCED form of the system. ---
#
#   We substitute D1-D6 INSIDE the evaluation of F, so the outer solver
#   only sees the G³ equations E1 in the G³ unknowns P[i,j,l]:
#
#        F(P)[i,j,l] := Σ_k x_k( μ_k(P; i,j,l),  P[i,j,l] ) = 0.           (E1 stacked)
#
#   Each evaluation of F at a candidate price array P performs, internally,
#   the 12G + 9 hidden calculations (D1-D6) per cell: 6G piecewise-linear
#   crossings (D1), 6G more (D2), sums those into six contour integrals
#   (D3-D5), applies Bayes' rule (D6), and finally reports the market
#   clearing residual (E1).
#
#   Functions in this file are tagged with their D/E role in the system.
#
# =====================================================================

using LinearAlgebra
using Printf

export Params, build_grid, nolearning_price, phi_map,
       residual_array, posteriors_at, solve_newton

# -------------------------------------------------------------------
# Parameters — model primitives (all scalars, no eqs/unknowns attached)
# -------------------------------------------------------------------
Base.@kwdef struct Params
    K::Int     = 3         # agents
    G::Int     = 5         # grid points per axis
    tau::Float64 = 2.0     # signal precision
    gamma::Float64 = 0.5   # risk-aversion coefficient (CRRA γ or CARA α depending on `utility`)
    W::Float64   = 1.0     # wealth per agent (used by CRRA only)
    umax::Float64 = 2.0    # grid range [-umax, +umax]
    utility::Symbol = :crra  # :crra  -> CRRA demand (power utility)
                             # :cara  -> CARA demand (exponential utility, linear in log-odds)
end

logistic(x) = 1.0 / (1.0 + exp(-x))
logit(p) = log(p / (1.0 - p))

# signal density f_v(u) where u = s - 1/2, so under v=1: u ~ N(+1/2, 1/tau)
# and under v=0: u ~ N(-1/2, 1/tau).
f1(u, tau) = sqrt(tau / (2pi)) * exp(-tau/2 * (u - 0.5)^2)
f0(u, tau) = sqrt(tau / (2pi)) * exp(-tau/2 * (u + 0.5)^2)

# Prior posterior (own signal only): mu = Lambda(tau*u)
prior_posterior(u, tau) = logistic(tau * u)

# [component of E1] Per-agent demand.
# Two functional forms are supported; both depend on (μ, p) via log-odds only.
#
#   :crra — CRRA / power utility
#      x_k = W·(R − 1) / ((1 − p) + R·p),    R = exp((logit μ − logit p)/γ)
#
#   :cara — CARA / exponential utility on the binary asset
#      x_k = (logit μ − logit p) / γ            [linear in log-odds]
#
# The market-clearing equation E1 is Σ_k x_k(μ_k, p) = 0 in both cases.
function demand(mu, p, P::Params)
    mu_c = clamp(mu, 1e-12, 1.0 - 1e-12)
    p_c  = clamp(p,  1e-12, 1.0 - 1e-12)
    lo_mu = logit(mu_c)
    lo_p  = logit(p_c)
    if P.utility === :cara
        return (lo_mu - lo_p) / P.gamma
    elseif P.utility === :crra
        R = exp((lo_mu - lo_p) / P.gamma)
        return P.W * (R - 1.0) / ((1.0 - p_c) + R * p_c)
    else
        error("unknown utility: $(P.utility); expected :crra or :cara")
    end
end

# [E1] Market-clearing residual  Σ_k x_k(mu_k, p).
# Setting this to zero (for each cell (i,j,l)) is the only equation that
# survives in the reduced system: E1 stacked over G³ cells.
function clearing_residual(mus::AbstractVector, p, P::Params)
    s = 0.0
    for mu in mus
        s += demand(mu, p, P)
    end
    return s
end

# 1D bisection — self-contained.
function bisect(f, a, b; tol=1e-12, maxiter=200)
    fa = f(a); fb = f(b)
    if fa == 0.0; return a; end
    if fb == 0.0; return b; end
    @assert fa * fb < 0 "bisect: f(a) and f(b) must have opposite signs (fa=$fa, fb=$fb)"
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(b - a) < tol || fm == 0.0
            return m
        end
        if fa * fm < 0
            b = m; fb = fm
        else
            a = m; fa = fm
        end
    end
    return 0.5 * (a + b)
end

function clear_price(mus::AbstractVector, P::Params)
    return bisect(p -> clearing_residual(mus, p, P), 1e-9, 1.0 - 1e-9)
end

# -------------------------------------------------------------------
# Grid helper
# -------------------------------------------------------------------
build_grid(P::Params) = collect(range(-P.umax, P.umax; length=P.G))

# -------------------------------------------------------------------
# No-learning initialization: each agent uses only own signal.
# -------------------------------------------------------------------
function nolearning_price(P::Params, u)
    G = P.G
    P0 = Array{Float64}(undef, G, G, G)
    for i in 1:G, j in 1:G, l in 1:G
        mus = [prior_posterior(u[i], P.tau),
               prior_posterior(u[j], P.tau),
               prior_posterior(u[l], P.tau)]
        P0[i,j,l] = clear_price(mus, P)
    end
    return P0
end

# -------------------------------------------------------------------
# [D1 / D2] Piecewise-linear 1D crossings.
#
# Both D1 (pass A) and D2 (pass B) are instances of the same 1D problem:
# given a 1D slice of the price tensor at the conjectured price p,
# find all grid off-axis points where the piecewise-linear interpolant
# equals p. Each crossing is one unknown (u_lc in D1 or u_jc in D2);
# we enumerate all of them because the price function need not be
# monotone along the axis.
# -------------------------------------------------------------------
function find_crossings!(out::Vector{Float64}, xvals::AbstractVector, yvals::AbstractVector, target)
    empty!(out)
    n = length(xvals)
    for k in 1:n-1
        y1, y2 = yvals[k], yvals[k+1]
        d1 = y1 - target
        d2 = y2 - target
        if d1 * d2 < 0
            t = d1 / (d1 - d2)  # = (target - y1)/(y2 - y1)
            push!(out, xvals[k] + t * (xvals[k+1] - xvals[k]))
        elseif d1 == 0 && d2 != 0
            push!(out, xvals[k])      # only report left endpoint of segment
        elseif k == n - 1 && d2 == 0
            push!(out, xvals[k+1])    # catch final endpoint on the last segment only
        end
    end
    return out
end

# -------------------------------------------------------------------
# [D1, D2, D3-D5, D6] Posterior of agent `ag` at cell (i,j,l),
# given conjectured price array Pg and observed price p_obs.
#
# Role of each block inside this function:
#   - the `for a in 1:G` loop performs D1 (sweep axis-A of the agent's
#     slice, root-find for u_{axis-B}) and accumulates the axis-A half
#     of D3-D5.
#   - the `for b in 1:G` loop performs D2 (sweep axis-B, root-find for
#     u_{axis-A}) and accumulates the axis-B half of D3-D5.
#   - the final Bayes combination  μ_k = f1_own·A1 / (f0_own·A0 + f1_own·A1)
#     implements D6 and returns the one scalar μ_k.
#
# The contour integrals D3-D5 are approximated by the two ½-weighted
# sums of f_v(u_a)·f_v(u_b) evaluated at every (a, b) crossing pair.
# -------------------------------------------------------------------
function agent_posterior(ag::Int, i::Int, j::Int, l::Int,
                         p_obs, Pg::AbstractArray{<:Real,3},
                         u::AbstractVector, P::Params;
                         buf::Vector{Float64}=Float64[])
    G = P.G
    tau = P.tau

    if ag == 1
        u_own = u[i]
        slice = @view Pg[i, :, :]
    elseif ag == 2
        u_own = u[j]
        slice = @view Pg[:, j, :]
    else
        u_own = u[l]
        slice = @view Pg[:, :, l]
    end

    A0 = 0.0    # accumulator for A_0^(k) — contour integral under v=0
    A1 = 0.0    # accumulator for A_1^(k) — contour integral under v=1
    cross = buf

    # [D1] Pass A: sweep u_a over the grid, root-find u_b on the contour
    # {(u_a, u_b): slice(u_a, u_b) = p_obs}.  Each crossing contributes one
    # term to D3-D5 via f_v(u_a)*f_v(u_b).
    for a in 1:G
        yvals = @view slice[a, :]
        find_crossings!(cross, u, yvals, p_obs)
        for ub in cross
            ua = u[a]
            A0 += f0(ua, tau) * f0(ub, tau)
            A1 += f1(ua, tau) * f1(ub, tau)
        end
    end

    # [D2] Pass B: sweep u_b over the grid, root-find u_a on the contour.
    for b in 1:G
        yvals = @view slice[:, b]
        find_crossings!(cross, u, yvals, p_obs)
        for ua in cross
            ub = u[b]
            A0 += f0(ua, tau) * f0(ub, tau)
            A1 += f1(ua, tau) * f1(ub, tau)
        end
    end

    # [D3-D5] Finalize contour integrals: A_v^(k) = ½·(pass-A + pass-B)
    A0 *= 0.5
    A1 *= 0.5

    # [D6] Bayes' rule using the agent's own signal likelihood and the
    # contour-integrated pair-likelihoods of the other two signals:
    #   μ_k = f1_own * A1^(k) / (f0_own * A0^(k) + f1_own * A1^(k))
    g0 = f0(u_own, tau)
    g1 = f1(u_own, tau)

    num = g1 * A1
    den = g0 * A0 + g1 * A1
    if den <= 0
        return prior_posterior(u_own, tau)
    end
    return num / den
end

function posteriors_at(i, j, l, p_obs, Pg, u, P::Params)
    buf = Float64[]
    return (agent_posterior(1, i, j, l, p_obs, Pg, u, P; buf=buf),
            agent_posterior(2, i, j, l, p_obs, Pg, u, P; buf=buf),
            agent_posterior(3, i, j, l, p_obs, Pg, u, P; buf=buf))
end

# -------------------------------------------------------------------
# [E1 stacked over G³ cells]
# Residual F(Pg)[i,j,l] = Σ_k x_k(μ_k(Pg; i,j,l), Pg[i,j,l]).
# The G³ unknowns are the prices P[i,j,l]; the G³ equations are the
# market-clearing residuals. Setting F = 0 gives REE.
# -------------------------------------------------------------------
function residual_array(Pg::AbstractArray{<:Real,3}, u::AbstractVector, P::Params)
    G = P.G
    F = similar(Pg)
    mus = Vector{Float64}(undef, 3)
    buf = Float64[]
    for i in 1:G, j in 1:G, l in 1:G
        p = Pg[i,j,l]
        mus[1] = agent_posterior(1, i, j, l, p, Pg, u, P; buf=buf)
        mus[2] = agent_posterior(2, i, j, l, p, Pg, u, P; buf=buf)
        mus[3] = agent_posterior(3, i, j, l, p, Pg, u, P; buf=buf)
        F[i,j,l] = clearing_residual(mus, p, P)
    end
    return F
end

# -------------------------------------------------------------------
# Phi map (Picard): new price array by clearing market at each cell.
# -------------------------------------------------------------------
function phi_map(Pg::AbstractArray{<:Real,3}, u::AbstractVector, P::Params)
    G = P.G
    Pnew = similar(Pg)
    buf = Float64[]
    for i in 1:G, j in 1:G, l in 1:G
        p_cur = Pg[i,j,l]
        mus = [agent_posterior(1, i, j, l, p_cur, Pg, u, P; buf=buf),
               agent_posterior(2, i, j, l, p_cur, Pg, u, P; buf=buf),
               agent_posterior(3, i, j, l, p_cur, Pg, u, P; buf=buf)]
        Pnew[i,j,l] = clear_price(mus, P)
    end
    return Pnew
end

# -------------------------------------------------------------------
# Flattened residual F(x) where x = vec(Pg)
# -------------------------------------------------------------------
function F!(F_flat::AbstractVector, x_flat::AbstractVector, u::AbstractVector, P::Params)
    G = P.G
    Pg = reshape(x_flat, G, G, G)
    Fa = residual_array(Pg, u, P)
    copyto!(F_flat, vec(Fa))
    return F_flat
end

function F(x_flat::AbstractVector, u::AbstractVector, P::Params)
    G = P.G
    Pg = reshape(x_flat, G, G, G)
    vec(residual_array(Pg, u, P))
end

# -------------------------------------------------------------------
# Finite-difference Jacobian (dense, central differences).
# For G=5: 125 variables, 250 F evaluations per Jacobian.
# -------------------------------------------------------------------
function fd_jacobian(x0::AbstractVector, u::AbstractVector, P::Params; h=1e-6)
    n = length(x0)
    J = Matrix{Float64}(undef, n, n)
    xp = copy(x0); xm = copy(x0)
    for k in 1:n
        xp[k] = x0[k] + h
        xm[k] = x0[k] - h
        fp = F(xp, u, P)
        fm = F(xm, u, P)
        @. J[:, k] = (fp - fm) / (2h)
        xp[k] = x0[k]; xm[k] = x0[k]
    end
    return J
end

# -------------------------------------------------------------------
# Damped Newton (Levenberg-Marquardt style) with backtracking line search.
# -------------------------------------------------------------------
function solve_newton(P::Params;
                      maxiters::Int=50,
                      abstol::Float64=1e-10,
                      h::Float64=1e-6,
                      lambda0::Float64=0.0,
                      verbose::Bool=true)
    u = build_grid(P)
    P0 = nolearning_price(P, u)
    x = vec(copy(P0))
    n = length(x)
    Fx = F(x, u, P)
    fnorm = norm(Fx, Inf)
    lambda = lambda0
    history = Float64[fnorm]

    if verbose
        @printf "iter %3d  ‖F‖∞ = %.6e  (initial, no-learning)\n" 0 fnorm
    end

    for it in 1:maxiters
        if fnorm < abstol
            break
        end
        J = fd_jacobian(x, u, P; h=h)
        # Direct Newton step: J dx = -F  (square system, n eqs = n unknowns)
        # With LM damping we solve (J + lambda I) dx = -F instead.
        if lambda > 0
            Jd = copy(J)
            @inbounds for k in 1:n
                Jd[k, k] += lambda
            end
            dx = Jd \ (-Fx)
        else
            dx = J \ (-Fx)
        end

        # Backtracking line search on ‖F‖∞
        alpha = 1.0
        xtrial = x + alpha * dx
        Ftrial = F(xtrial, u, P)
        nt = norm(Ftrial, Inf)
        tries = 0
        while nt > fnorm && tries < 25
            alpha *= 0.5
            xtrial = x + alpha * dx
            Ftrial = F(xtrial, u, P)
            nt = norm(Ftrial, Inf)
            tries += 1
        end

        # If line search failed to reduce F, bump lambda and retry next iter
        if nt >= fnorm
            lambda = lambda > 0 ? 10 * lambda : 1e-3
        else
            lambda = max(lambda / 2, 0.0)
        end

        x = xtrial
        Fx = Ftrial
        fnorm = nt
        push!(history, fnorm)
        if verbose
            @printf "iter %3d  ‖F‖∞ = %.6e   step=%.3g   lambda=%.2e\n" it fnorm alpha lambda
        end
    end

    Pg_star = reshape(x, P.G, P.G, P.G)
    return (; P_star=Pg_star, P0=P0, u=u, residual=reshape(Fx, P.G, P.G, P.G),
            history=history, converged=(fnorm < abstol))
end

# -------------------------------------------------------------------
# Anderson-accelerated Picard on Φ (for large G where a dense Jacobian
# is infeasible).
#
#   Picard step:       P^{n+1} = Φ(P^n)
#   Anderson mixes the last m iterates to form a quasi-Newton step
#   on the residual g(P) := Φ(P) - P.
#
# At REE both F(P)=0 (stacked E1) and g(P)=0 hold simultaneously — the
# two formulations share the same fixed point.
# -------------------------------------------------------------------
function solve_anderson(P::Params;
                        maxiters::Int=60,
                        m::Int=6,
                        abstol::Float64=1e-9,
                        damping::Float64=1.0,
                        verbose::Bool=true)
    u = build_grid(P)
    P0 = nolearning_price(P, u)
    x = vec(copy(P0))
    n = length(x)

    gx = vec(phi_map(reshape(x, P.G, P.G, P.G), u, P)) .- x
    gnorm = norm(gx, Inf)
    history = Float64[gnorm]

    # storage for last m iterates / residuals
    X = Vector{Vector{Float64}}()
    G = Vector{Vector{Float64}}()
    push!(X, copy(x))
    push!(G, copy(gx))

    if verbose
        @printf "iter %3d  ‖Φ(P)-P‖∞ = %.6e  (initial)\n" 0 gnorm
    end

    for it in 1:maxiters
        if gnorm < abstol; break; end

        # next iterate via linear least-squares over stored residuals
        if length(G) == 1
            x_new = X[end] .+ damping .* G[end]
        else
            # differences of residuals
            ΔG = hcat([G[k+1] .- G[k] for k in 1:length(G)-1]...)
            ΔX = hcat([X[k+1] .- X[k] for k in 1:length(X)-1]...)
            # solve min_γ ‖G[end] - ΔG*γ‖
            γ = ΔG \ G[end]
            x_new = X[end] .+ damping .* G[end] .- (ΔX .+ damping .* ΔG) * γ
        end

        g_new = vec(phi_map(reshape(x_new, P.G, P.G, P.G), u, P)) .- x_new
        gnorm = norm(g_new, Inf)

        # slide history window
        push!(X, copy(x_new))
        push!(G, copy(g_new))
        if length(X) > m + 1
            popfirst!(X); popfirst!(G)
        end
        x = x_new
        push!(history, gnorm)
        if verbose
            @printf "iter %3d  ‖Φ(P)-P‖∞ = %.6e\n" it gnorm
        end
    end

    Pg_star = reshape(x, P.G, P.G, P.G)
    Fa = residual_array(Pg_star, u, P)
    return (; P_star=Pg_star, P0=P0, u=u, residual=Fa,
            history=history, converged=(gnorm < abstol))
end

# -------------------------------------------------------------------
# Plain Picard iteration:
#
#   P^{n+1}[i,j,l] = Φ(P^n)[i,j,l]  for every cell,
#
# where Φ at each cell is ONE 1-D root-find of the pointwise market
# clearing equation E1 given posteriors computed from the conjectured
# tensor P^n. Each cell's solve is independent in Φ — the only
# coupling between cells is the (shared) conjectured price function.
#
# Damping:  P^{n+1} = α·Φ(P^n) + (1-α)·P^n  stabilises when Φ is not a
# contraction near the fixed point.
# -------------------------------------------------------------------
function solve_picard(P::Params;
                      maxiters::Int=500,
                      abstol::Float64=1e-12,
                      alpha::Float64=1.0,
                      verbose::Bool=true)
    u = build_grid(P)
    P0 = nolearning_price(P, u)
    Pcur = copy(P0)
    history = Float64[]

    if verbose
        Fa = residual_array(Pcur, u, P)
        gnorm0 = maximum(abs, Fa)
        @printf "iter %3d  ‖Φ-I‖∞=%.2e   ‖F‖∞=%.2e   (initial, no-learning)\n" 0 NaN gnorm0
    end

    gnorm_g = Inf
    for it in 1:maxiters
        Pnew = phi_map(Pcur, u, P)
        diff = maximum(abs, Pnew .- Pcur)
        Pcur = alpha .* Pnew .+ (1 - alpha) .* Pcur
        push!(history, diff)

        # also report the stacked market-clearing residual at the current P
        Fa = residual_array(Pcur, u, P)
        fnorm = maximum(abs, Fa)
        gnorm_g = diff

        if verbose
            @printf "iter %3d  ‖Φ-I‖∞=%.2e   ‖F‖∞=%.2e\n" it diff fnorm
        end
        if diff < abstol
            break
        end
    end

    Fa = residual_array(Pcur, u, P)
    return (; P_star=Pcur, P0=P0, u=u, residual=Fa,
            history=history, converged=(gnorm_g < abstol))
end

end # module
