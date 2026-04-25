using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

const U_REPORT = (1.0, -1.0, 1.0)

function run_picard(; gamma::Float64, utility::Symbol, tag::String,
                     G::Int=5, tau::Float64=2.0, umax::Float64=2.0,
                     alpha::Float64=1.0, maxiters::Int=400, abstol::Float64=1e-13)
    P = Rezn.Params(G=G, tau=tau, gamma=gamma, umax=umax, utility=utility)
    @printf "\n======================================================\n"
    @printf "  %s   (G=%d, τ=%.2f, γ=%.3g, α=%.2f)\n" tag G tau gamma alpha
    @printf "======================================================\n"
    t0 = time()
    res = Rezn.solve_picard(P; maxiters=maxiters, abstol=abstol, alpha=alpha, verbose=false)
    dt = time() - t0

    u = res.u
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    p_star = res.P_star[i,j,l]
    mus = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)

    Finf = maximum(abs, res.residual)
    diff_final = isempty(res.history) ? NaN : res.history[end]

    @printf "iterations:                %d\n"      length(res.history)
    @printf "‖Φ(P)-P‖∞ (final step):    %.3e\n"    diff_final
    @printf "‖F(P)‖∞ at last iterate:   %.3e\n"    Finf
    @printf "time:                      %.2f s\n"  dt
    @printf "(1,-1,1) cell:\n"
    @printf "  p*  = %.15f\n"   p_star
    @printf "  μ_1 = %.15f   (u=+1)\n" mus[1]
    @printf "  μ_2 = %.15f   (u=-1)\n" mus[2]
    @printf "  μ_3 = %.15f   (u=+1)\n" mus[3]
    return (; tag, p_star, mus, Finf, diff_final, history=res.history, dt)
end

crra = run_picard(gamma=0.5, utility=:crra, tag="CRRA γ=0.5", alpha=1.0)
cara = run_picard(gamma=1.0, utility=:cara, tag="CARA γ=1",   alpha=1.0)

println("\n=== Picard convergence history (first 10 + last 10 iters) ===")
for (r, name) in [(crra, "CRRA"), (cara, "CARA")]
    h = r.history
    println("-- $name: $(length(h)) iters --")
    n = min(10, length(h))
    for k in 1:n
        @printf "  iter %3d: ‖Φ-I‖∞ = %.3e\n" k h[k]
    end
    if length(h) > 20
        println("  ...")
        for k in length(h)-9:length(h)
            @printf "  iter %3d: ‖Φ-I‖∞ = %.3e\n" k h[k]
        end
    end
end

println("\n=== Summary at (1,-1,1), G=5, τ=2 ===")
@printf "                 |   CRRA γ=0.5    |   CARA γ=1      |\n"
@printf "p*               | %.12f  | %.12f  |\n" crra.p_star cara.p_star
@printf "μ₁ (u=+1)        | %.12f  | %.12f  |\n" crra.mus[1] cara.mus[1]
@printf "μ₂ (u=-1)        | %.12f  | %.12f  |\n" crra.mus[2] cara.mus[2]
@printf "μ₃ (u=+1)        | %.12f  | %.12f  |\n" crra.mus[3] cara.mus[3]
@printf "‖Φ(P)-P‖∞ final  | %.3e       | %.3e       |\n" crra.diff_final cara.diff_final
@printf "‖F(P)‖∞ final    | %.3e       | %.3e       |\n" crra.Finf cara.Finf
