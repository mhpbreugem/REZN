using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

const TAU      = 2.0
const GAMMA    = 2.0
const UMAX     = 2.0
const U_REPORT = (1.0, -1.0, 1.0)

function run_G(G::Int; picard_iters::Int=2000, abstol=1e-13)
    P = Rezn.Params(G=G, tau=TAU, gamma=GAMMA, umax=UMAX, utility=:crra)
    u = Rezn.build_grid(P)
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    u_hit = (u[i], u[j], u[l])

    @printf "\n=== G=%2d  (grid step = %.4f) — cell (%d,%d,%d) -> u = (%.4f, %.4f, %.4f) ===\n" G (u[2]-u[1]) i j l u_hit[1] u_hit[2] u_hit[3]

    t0 = time()
    res = Rezn.solve_picard(P; maxiters=picard_iters, abstol=abstol,
                            alpha=1.0, verbose=false)
    dt = time() - t0

    p_star = res.P_star[i,j,l]
    mus = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    Finf = maximum(abs, res.residual)
    PhiI = isempty(res.history) ? NaN : res.history[end]

    @printf "  Picard iters  = %d\n"       length(res.history)
    @printf "  time          = %.1f s\n"   dt
    @printf "  ‖Φ(P)-P‖∞     = %.3e\n"     PhiI
    @printf "  ‖F(P*)‖∞      = %.3e\n"     Finf
    @printf "  p*            = %.10f\n"    p_star
    @printf "  μ₁ (u=+1)     = %.10f\n"    mus[1]
    @printf "  μ₂ (u=-1)     = %.10f\n"    mus[2]
    @printf "  μ₃ (u=+1)     = %.10f\n"    mus[3]
    @printf "  PR gap μ₁-μ₂  = %.6f\n"     mus[1]-mus[2]

    return (; G, iters=length(res.history), dt, PhiI, Finf,
            p_star, mus, u_hit)
end

results = []
for G in (5, 10, 15, 20)
    push!(results, run_G(G))
end

println("\n" * "="^115)
@printf "CRRA γ=%.2f, τ=%.2f,  target (1,-1,1) -> nearest grid cell\n" GAMMA TAU
println("="^115)
@printf "%3s | %-18s | %7s | %9s | %9s | %14s | %14s | %14s | %8s\n" "G" "cell (u1,u2,u3)" "iters" "time(s)" "‖F‖∞" "p*" "μ₁" "μ₂" "PR gap"
println("-"^115)
for r in results
    @printf "%3d | (%5.2f,%5.2f,%5.2f) | %7d | %9.1f | %9.2e | %14.10f | %14.10f | %14.10f | %8.5f\n" r.G r.u_hit[1] r.u_hit[2] r.u_hit[3] r.iters r.dt r.Finf r.p_star r.mus[1] r.mus[2] r.mus[1]-r.mus[2]
end
println("-"^115)
@printf "Note: target signal is (1,-1,1); grid is symmetric range [-%.1f, %.1f]/G-1 so odd/even G picks the nearest grid cell.\n" UMAX UMAX
