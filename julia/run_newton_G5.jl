using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

function run_case(; gamma::Float64, tag::String, u_report=(1.0, -1.0, 1.0),
                   maxiters=50)
    P = Rezn.Params(G=5, tau=2.0, gamma=gamma, umax=2.0)
    @printf "\n=== %s (gamma=%.3g) ===\n" tag gamma
    t0 = time()
    res = Rezn.solve_newton(P; verbose=true, maxiters=maxiters, abstol=1e-10)
    dt = time() - t0
    @printf "solve time = %.2f s\n" dt
    @printf "‖F‖∞ = %.3e    converged=%s\n" norm(vec(res.residual), Inf) res.converged

    u = res.u
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(u_report[1]), idx(u_report[2]), idx(u_report[3])
    p0     = res.P0[i,j,l]
    p_star = res.P_star[i,j,l]
    mus    = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    @printf "grid indices (i,j,l) = (%d,%d,%d) -> u = (%.2f, %.2f, %.2f)\n" i j l u[i] u[j] u[l]
    @printf "nolearning price      = %.6f\n" p0
    @printf "converged price p*    = %.6f\n" p_star
    @printf "posteriors at p*      : mu1=%.6f, mu2=%.6f, mu3=%.6f\n" mus[1] mus[2] mus[3]
    return (gamma=gamma, p0=p0, p_star=p_star, mus=mus, res=res)
end

cara = run_case(gamma=100.0, tag="CARA (gamma=100, FR limit)")
crra = run_case(gamma=0.5,   tag="CRRA (gamma=0.5, PR expected)")

println("\n=== Comparison at (u1,u2,u3) = (1,-1,1), G=5, tau=2.0 ===")
@printf "                     |   CARA    |   CRRA    |\n"
@printf "price (no-learning)  | %.6f | %.6f |\n" cara.p0 crra.p0
@printf "price (REE, Newton)  | %.6f | %.6f |\n" cara.p_star crra.p_star
@printf "mu_1 (u=+1)          | %.6f | %.6f |\n" cara.mus[1] crra.mus[1]
@printf "mu_2 (u=-1)          | %.6f | %.6f |\n" cara.mus[2] crra.mus[2]
@printf "mu_3 (u=+1)          | %.6f | %.6f |\n" cara.mus[3] crra.mus[3]

println("\nSESSION_SUMMARY benchmark (G=5, tau=2):")
println("  CARA FR prediction:  p* = Lambda(2) = 0.880797, mu_k = 0.880797 all")
println("  CRRA PR prediction:  p* ~= 0.9077, mu ~= {0.9185, 0.8889, 0.9185}")
