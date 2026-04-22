using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf
using LinearAlgebra

const TAU   = 2.0
const GAMMA = 0.5
const U_REPORT = (1.0, -1.0, 1.0)

"""
Run REE solve at the given grid size G, return a named tuple with p*, μ*, ‖F‖∞.
- G ≤ 10: damped Newton on the 125..1000-var system (FD Jacobian).
- G ≥ 15: Anderson-accelerated Picard on Φ.
"""
function run_at_G(G::Int; gamma::Float64=GAMMA, umax::Float64=2.0)
    P = Rezn.Params(G=G, tau=TAU, gamma=gamma, umax=umax)
    t0 = time()
    if G <= 10
        res = Rezn.solve_newton(P; verbose=false, maxiters=80, abstol=1e-12)
        method = "Newton"
    else
        res = Rezn.solve_anderson(P; verbose=false, maxiters=120, m=6, abstol=1e-10)
        method = "Anderson"
    end
    dt = time() - t0

    u = res.u
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    p0     = res.P0[i,j,l]
    p_star = res.P_star[i,j,l]
    mus    = Rezn.posteriors_at(i, j, l, p_star, res.P_star, u, P)
    Finf   = norm(vec(res.residual), Inf)

    return (; G, method, dt, p0, p_star,
              mu1=mus[1], mu2=mus[2], mu3=mus[3],
              Finf, converged=res.converged,
              u_report=(u[i], u[j], u[l]))
end

println("REE contour sweep:  τ=$(TAU), γ=$(GAMMA), (u1,u2,u3) ≈ $(U_REPORT)")
println("="^78)

rows = Any[]
for G in (5, 10, 20, 50)
    @printf "running G=%2d ..." G
    r = run_at_G(G)
    push!(rows, r)
    @printf " done in %.1fs  (%s, ‖F‖∞=%.2e)\n" r.dt r.method r.Finf
end

println()
println("G-sweep at (u1,u2,u3) = $(U_REPORT),  τ=$(TAU),  γ=$(GAMMA):")
println("-"^82)
@printf "%3s | %8s | %10s | %10s | %8s | %8s | %8s | %10s | %7s\n" "G" "method" "p0 (NL)" "p* (REE)" "mu1" "mu2" "mu3" "‖F‖∞" "time(s)"
println("-"^82)
for r in rows
    @printf "%3d | %8s | %10.6f | %10.6f | %8.6f | %8.6f | %8.6f | %10.2e | %7.2f\n" r.G r.method r.p0 r.p_star r.mu1 r.mu2 r.mu3 r.Finf r.dt
end
println("-"^82)
println("SESSION_SUMMARY target at (1,-1,1), γ=0.5, τ=2:")
println("  CRRA PR:  p* ≈ 0.9077,  μ ≈ (0.9185, 0.8889, 0.9185)")
