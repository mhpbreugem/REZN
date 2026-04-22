module Rezn

using LinearAlgebra
using Printf

export Params, build_grid, nolearning_price, phi_map,
       residual_array, posteriors_at, solve_newton

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
Base.@kwdef struct Params
    K::Int     = 3         # agents
    G::Int     = 5         # grid points per axis
    tau::Float64 = 2.0     # signal precision
    gamma::Float64 = 0.5   # CRRA coefficient (gamma -> inf approaches CARA)
    W::Float64   = 1.0     # wealth per agent
    umax::Float64 = 2.0    # grid range [-umax, +umax]
end

logistic(x) = 1.0 / (1.0 + exp(-x))
logit(p) = log(p / (1.0 - p))

# signal density f_v(u) where u = s - 1/2, so under v=1: u ~ N(+1/2, 1/tau)
# and under v=0: u ~ N(-1/2, 1/tau).
f1(u, tau) = sqrt(tau / (2pi)) * exp(-tau/2 * (u - 0.5)^2)
f0(u, tau) = sqrt(tau / (2pi)) * exp(-tau/2 * (u + 0.5)^2)

# Prior posterior (own signal only): mu = Lambda(tau*u)
prior_posterior(u, tau) = logistic(tau * u)

# CRRA demand per agent:
# x_k = W*(R-1)/((1-p) + R*p), R = exp((logit(mu) - logit(p))/gamma).
function demand(mu, p, P::Params)
    mu_c = clamp(mu, 1e-12, 1.0 - 1e-12)
    p_c  = clamp(p,  1e-12, 1.0 - 1e-12)
    R = exp((logit(mu_c) - logit(p_c)) / P.gamma)
    return P.W * (R - 1.0) / ((1.0 - p_c) + R * p_c)
end

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
# Piecewise-linear 1D crossings.
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
# Posterior of agent `ag` at realization (i,j,l) given conjectured price array Pg.
# 2-pass contour method.
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

    A0 = 0.0
    A1 = 0.0
    cross = buf

    # Pass A: sweep a over grid, root-find b
    for a in 1:G
        yvals = @view slice[a, :]
        find_crossings!(cross, u, yvals, p_obs)
        for ub in cross
            ua = u[a]
            A0 += f0(ua, tau) * f0(ub, tau)
            A1 += f1(ua, tau) * f1(ub, tau)
        end
    end

    # Pass B: sweep b over grid, root-find a
    for b in 1:G
        yvals = @view slice[:, b]
        find_crossings!(cross, u, yvals, p_obs)
        for ua in cross
            ub = u[b]
            A0 += f0(ua, tau) * f0(ub, tau)
            A1 += f1(ua, tau) * f1(ub, tau)
        end
    end

    A0 *= 0.5
    A1 *= 0.5

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
# Residual F(Pg)[i,j,l] = Σ_k x_k(mu_k(Pg; i,j,l), Pg[i,j,l]).
# At REE this is zero everywhere.
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
        # Levenberg-Marquardt step: (J'J + lambda I) dx = -J' F
        A = J' * J
        if lambda > 0
            @inbounds for k in 1:n
                A[k, k] += lambda
            end
        end
        rhs = -(J' * Fx)
        dx = A \ rhs

        # simple backtracking line search
        alpha = 1.0
        xtrial = x + alpha * dx
        Ftrial = F(xtrial, u, P)
        nt = norm(Ftrial, Inf)
        tries = 0
        while nt > fnorm && tries < 20
            alpha *= 0.5
            xtrial = x + alpha * dx
            Ftrial = F(xtrial, u, P)
            nt = norm(Ftrial, Inf)
            tries += 1
        end

        x = xtrial
        Fx = Ftrial
        fnorm = nt
        push!(history, fnorm)
        if verbose
            @printf "iter %3d  ‖F‖∞ = %.6e   step=%.3g   lambda=%.1g\n" it fnorm alpha lambda
        end
    end

    Pg_star = reshape(x, P.G, P.G, P.G)
    return (; P_star=Pg_star, P0=P0, u=u, residual=reshape(Fx, P.G, P.G, P.G),
            history=history, converged=(fnorm < abstol))
end

end # module
