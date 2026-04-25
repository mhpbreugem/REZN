using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

const G        = 10
const TAU      = 2.0
const UMAX     = 2.0
const U_REPORT = (1.0, -1.0, 1.0)

function params(utility, gamma)
    Rezn.Params(G=G, tau=TAU, gamma=gamma, umax=UMAX, utility=utility)
end

function run_picard(P::Rezn.Params; maxiters::Int=800, abstol=1e-13)
    t0 = time()
    res = Rezn.solve_picard(P; maxiters=maxiters, abstol=abstol, alpha=1.0, verbose=false)
    dt = time() - t0
    return (; res, dt)
end

function run_newton(P::Rezn.Params; maxiters::Int=30, abstol=1e-10, x0=nothing)
    t0 = time()
    # if a warm-start was provided, patch the module-level initial condition
    # by constructing a modified driver: we reuse solve_newton which reads
    # no-learning. For warm-start we short-circuit with a small helper below.
    res = if x0 === nothing
        Rezn.solve_newton(P; maxiters=maxiters, abstol=abstol, verbose=false)
    else
        _solve_newton_from(P, x0; maxiters=maxiters, abstol=abstol)
    end
    dt = time() - t0
    return (; res, dt)
end

# mimic Rezn.solve_newton but accept a warm-start x0 (flattened price tensor)
function _solve_newton_from(P::Rezn.Params, x0::AbstractVector;
                            maxiters::Int=30, abstol=1e-10,
                            h::Float64=1e-6)
    u = Rezn.build_grid(P)
    x = copy(x0)
    n = length(x)
    Fx = Rezn.F(x, u, P)
    fnorm = norm(Fx, Inf)
    lambda = 0.0
    history = Float64[fnorm]
    P0 = Rezn.nolearning_price(P, u)
    for it in 1:maxiters
        fnorm < abstol && break
        J = Rezn.fd_jacobian(x, u, P; h=h)
        if lambda > 0
            Jd = copy(J); @inbounds for k in 1:n; Jd[k,k] += lambda; end
            dx = Jd \ (-Fx)
        else
            dx = J \ (-Fx)
        end
        alpha = 1.0
        xtrial = x + alpha*dx
        Ftrial = Rezn.F(xtrial, u, P)
        nt = norm(Ftrial, Inf)
        tries = 0
        while nt > fnorm && tries < 25
            alpha *= 0.5
            xtrial = x + alpha*dx
            Ftrial = Rezn.F(xtrial, u, P)
            nt = norm(Ftrial, Inf)
            tries += 1
        end
        if nt >= fnorm
            lambda = lambda > 0 ? 10*lambda : 1e-3
        else
            lambda = max(lambda/2, 0.0)
        end
        x = xtrial; Fx = Ftrial; fnorm = nt
        push!(history, fnorm)
    end
    return (; P_star=reshape(x, P.G, P.G, P.G), P0=P0, u=u,
            residual=reshape(Fx, P.G, P.G, P.G), history, converged=(fnorm<abstol))
end

# report at (1,-1,1)
function report_cell(res, P)
    u = res.u
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    p_star = res.P_star[i,j,l]
    mus = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    return (i=i, j=j, l=l, p_star=p_star, mus=mus,
            Finf=maximum(abs, res.residual), iters=length(res.history))
end

function do_case(tag, utility, gamma)
    P = params(utility, gamma)
    @printf "\n=== %s   G=%d  τ=%.2f  γ=%.3g ===\n" tag G TAU gamma

    @printf "  [Picard]  running..."
    pic = run_picard(P)
    rP = report_cell(pic.res, P)
    @printf " %d iters, %.1fs\n" rP.iters pic.dt

    @printf "  [Newton]  from no-learning..."
    new = run_newton(P; maxiters=30, abstol=1e-10)
    rN = report_cell(new.res, P)
    @printf " %d iters, %.1fs\n" rN.iters new.dt

    @printf "  [Newton from Picard warm-start]..."
    newP = run_newton(P; maxiters=20, abstol=1e-10, x0=vec(pic.res.P_star))
    rNP = report_cell(newP.res, P)
    @printf " %d iters, %.1fs\n" rNP.iters newP.dt

    return (tag=tag, P=P, pic=pic, new=new, newP=newP,
            rP=rP, rN=rN, rNP=rNP)
end

crra = do_case("CRRA γ=0.5",  :crra, 0.5)
cara = do_case("CARA γ=1",    :cara, 1.0)

function print_block(tag, case)
    println("\n------- $tag ------")
    @printf "%-26s | %-15s | %-15s | %-15s\n" "" "Picard" "Newton(cold)" "Newton(warm)"
    @printf "%-26s | %15d | %15d | %15d\n"     "iterations" case.rP.iters case.rN.iters case.rNP.iters
    @printf "%-26s | %15.1f | %15.1f | %15.1f\n" "time (s)"  case.pic.dt   case.new.dt   case.newP.dt
    @printf "%-26s | %15.12f | %15.12f | %15.12f\n" "p* (1,-1,1)"  case.rP.p_star   case.rN.p_star   case.rNP.p_star
    @printf "%-26s | %15.12f | %15.12f | %15.12f\n" "μ₁ (u=+1)"     case.rP.mus[1]   case.rN.mus[1]   case.rNP.mus[1]
    @printf "%-26s | %15.12f | %15.12f | %15.12f\n" "μ₂ (u=-1)"     case.rP.mus[2]   case.rN.mus[2]   case.rNP.mus[2]
    @printf "%-26s | %15.12f | %15.12f | %15.12f\n" "μ₃ (u=+1)"     case.rP.mus[3]   case.rN.mus[3]   case.rNP.mus[3]
    @printf "%-26s | %15.2e | %15.2e | %15.2e\n"   "‖F(P*)‖∞"      case.rP.Finf     case.rN.Finf     case.rNP.Finf
end
print_block("CRRA γ=0.5  @ (1,-1,1), G=10, τ=2", crra)
print_block("CARA γ=1    @ (1,-1,1), G=10, τ=2", cara)

println("\n(‖F‖∞ is the market-clearing residual over the full G³=$(G^3)-cell tensor)")
