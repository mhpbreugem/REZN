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

params_crra(gamma) = Rezn.Params(G=G, tau=TAU, gamma=gamma, umax=UMAX, utility=:crra)

function _solve_newton_from(P::Rezn.Params, x0::AbstractVector;
                            maxiters::Int=15, abstol=1e-11, h::Float64=1e-6)
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

function report_cell(res, P)
    u = res.u
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    p_star = res.P_star[i,j,l]
    mus = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    return (i=i, j=j, l=l, p_star=p_star, mus=mus,
            Finf=maximum(abs, res.residual), iters=length(res.history))
end

function run_gamma(gamma; picard_iters=400, newton_iters=15)
    P = params_crra(gamma)
    @printf "\n=== CRRA γ=%.3g ===\n" gamma

    t0 = time()
    pic_res = Rezn.solve_picard(P; maxiters=picard_iters, abstol=1e-14, alpha=1.0, verbose=false)
    dt_pic = time() - t0
    rP = report_cell(pic_res, P)
    @printf "  Picard      : %4d iters, %6.1fs, p*=%.9f, ‖F‖∞=%.2e\n" rP.iters dt_pic rP.p_star rP.Finf

    # warm-start Newton from Picard
    t0 = time()
    newW = _solve_newton_from(P, vec(pic_res.P_star); maxiters=newton_iters, abstol=1e-11)
    dt_w = time() - t0
    rW = report_cell(newW, P)
    @printf "  Newton warm : %4d iters, %6.1fs, p*=%.9f, ‖F‖∞=%.2e\n" rW.iters dt_w rW.p_star rW.Finf

    # cold-start Newton from no-learning
    t0 = time()
    newC = Rezn.solve_newton(P; maxiters=newton_iters, abstol=1e-11, verbose=false)
    dt_c = time() - t0
    rC = report_cell(newC, P)
    @printf "  Newton cold : %4d iters, %6.1fs, p*=%.9f, ‖F‖∞=%.2e\n" rC.iters dt_c rC.p_star rC.Finf

    return (gamma=gamma, rP=rP, rW=rW, rC=rC, dt_pic=dt_pic, dt_w=dt_w, dt_c=dt_c)
end

results = []
for g in (0.3, 0.5, 1.0, 3.0, 10.0)
    push!(results, run_gamma(g))
end

println("\n" * "="^95)
@printf "CRRA gamma sweep @ (1,-1,1) ≈ (1.111,-1.111,1.111), G=%d, τ=%.2f\n" G TAU
println("="^95)
@printf "%6s | %-13s | %13s %11s %7s | %11s %7s | %11s %7s\n" "γ" "method" "p*" "‖F‖∞" "iters" "mu1" "mu2" "PR_gap" ""
println("-"^95)
for r in results
    for (tag, rc, dt) in [("Picard", r.rP, r.dt_pic), ("Newton warm", r.rW, r.dt_w), ("Newton cold", r.rC, r.dt_c)]
        gap = rc.mus[1] - rc.mus[2]
        @printf "%6.2f | %-13s | %13.9f %11.2e %7d | %11.6f %7.4f | %11.6f %s\n" r.gamma tag rc.p_star rc.Finf rc.iters rc.mus[1] rc.mus[2] gap (rc.Finf < 1e-10 ? "✓" : "")
    end
    println("-"^95)
end

println("\nNote: PR_gap = μ_1 − μ_2 (larger ⇒ more partial revelation).")
