using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

const G        = 3
const TAU      = 2.0
const GAMMA    = 2.0
const UMAX     = 2.0
const U_REPORT = (1.0, -1.0, 1.0)

P = Rezn.Params(G=G, tau=TAU, gamma=GAMMA, umax=UMAX, utility=:crra)
u = Rezn.build_grid(P)
@printf "Grid (G=%d, umax=%.2f):  u = %s\n" G UMAX u

idx(x) = argmin(abs.(u .- x))
i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
@printf "Report cell:  (i,j,l) = (%d,%d,%d)  =>  (u1,u2,u3) = (%.3f, %.3f, %.3f)\n\n" i j l u[i] u[j] u[l]

function report_cell(res, P)
    u = res.u
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    p_star = res.P_star[i,j,l]
    mus = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    return (i=i, j=j, l=l, p_star=p_star, mus=mus,
            Finf=maximum(abs, res.residual), iters=length(res.history))
end

function _solve_newton_from(P::Rezn.Params, x0::AbstractVector;
                            maxiters::Int=80, abstol=1e-12, h::Float64=1e-6)
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

# Picard
t0 = time()
pic = Rezn.solve_picard(P; maxiters=2000, abstol=1e-14, alpha=1.0, verbose=false)
dt_pic = time() - t0
rP = report_cell(pic, P)
@printf "[Picard]      iters=%4d  t=%.2fs   p*=%.12f   ‖F‖∞=%.3e\n" rP.iters dt_pic rP.p_star rP.Finf

# Newton warm
t0 = time()
newW = _solve_newton_from(P, vec(pic.P_star); maxiters=80, abstol=1e-12)
dt_w = time() - t0
rW = report_cell(newW, P)
@printf "[Newton warm] iters=%4d  t=%.2fs   p*=%.12f   ‖F‖∞=%.3e\n" rW.iters dt_w rW.p_star rW.Finf

# Newton cold
t0 = time()
newC = Rezn.solve_newton(P; maxiters=80, abstol=1e-12, verbose=false)
dt_c = time() - t0
rC = report_cell(newC, P)
@printf "[Newton cold] iters=%4d  t=%.2fs   p*=%.12f   ‖F‖∞=%.3e\n" rC.iters dt_c rC.p_star rC.Finf

println("\n=== CRRA γ=2.0 @ (u1,u2,u3)=(1.0,-1.0,1.0), G=3, τ=2.0 ===")
@printf "%-15s | %15s | %15s | %15s | %12s | %8s\n" "method" "p*" "μ₁ (u=+1)" "μ₂ (u=-1)" "‖F‖∞" "PR gap"
println("-"^95)
for (name, r) in [("Picard", rP), ("Newton warm", rW), ("Newton cold", rC)]
    gap = r.mus[1] - r.mus[2]
    @printf "%-15s | %15.12f | %15.12f | %15.12f | %12.3e | %8.5f\n" name r.p_star r.mus[1] r.mus[2] r.Finf gap
end
println()
@printf "No-learning price at this cell:  p0 = %.12f\n" pic.P0[i,j,l]
