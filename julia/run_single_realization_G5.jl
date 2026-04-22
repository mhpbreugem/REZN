using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Rezn.jl"))
using .Rezn
using Printf

"""
Partial-REE solve at a single realization (i,j,l):
  - every other cell of the G³ price tensor is held at its no-learning
    value P0[·,·,·];
  - only the one cell P[i,j,l] is free;
  - agents use the 2-pass contour method on the (partially-updated)
    tensor to form posteriors, then market clears at the cell.
This is NOT the full REE — it isolates one realization. It is a strictly
1D root problem in p = P[i,j,l], and bisection drives it below any tol.
"""
function single_realization(; gamma::Float64, utility::Symbol,
                             tag::String, u_report=(1.0, -1.0, 1.0),
                             G::Int=5, tau::Float64=2.0, umax::Float64=2.0,
                             tol::Float64=1e-14)
    P = Rezn.Params(G=G, tau=tau, gamma=gamma, umax=umax, utility=utility)
    u = Rezn.build_grid(P)
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(u_report[1]), idx(u_report[2]), idx(u_report[3])

    # Initialise full tensor at no-learning
    Pg = Rezn.nolearning_price(P, u)
    p0 = Pg[i, j, l]

    # h(p) := market-clearing residual at (i,j,l) when P[i,j,l]=p, all other
    # cells fixed at no-learning.
    function h(p)
        Pg[i, j, l] = p
        mus = Rezn.posteriors_at(i, j, l, p, Pg, u, P)
        return Rezn.clearing_residual(collect(mus), p, P)
    end

    # Bracket and bisect
    a, b = 1e-9, 1.0 - 1e-9
    ha, hb = h(a), h(b)
    if ha * hb >= 0
        error("h(p) does not change sign on (0,1); ha=$ha hb=$hb")
    end

    # High-precision bisection
    iters = 0
    while (b - a) > tol && iters < 200
        m = 0.5 * (a + b)
        hm = h(m)
        if ha * hm <= 0
            b, hb = m, hm
        else
            a, ha = m, hm
        end
        iters += 1
    end
    p_star = 0.5 * (a + b)
    Pg[i, j, l] = p_star
    mus = Rezn.posteriors_at(i, j, l, p_star, Pg, u, P)
    err = h(p_star)

    @printf "\n=== %s (γ=%.3g) — single realization (u1,u2,u3)=(%.2f,%.2f,%.2f), G=%d, τ=%.1f ===\n" tag gamma u_report[1] u_report[2] u_report[3] G tau
    @printf "  iterations (bisection):  %d\n"  iters
    @printf "  bracket width at end:    %.2e\n" (b - a)
    @printf "  market-clearing residual |h(p*)|:  %.3e      (target < 1e-10: %s)\n" abs(err) (abs(err) < 1e-10 ? "YES ✓" : "NO")
    @printf "  no-learning price  p0  = %.15f\n" p0
    @printf "  solved price       p*  = %.15f\n" p_star
    @printf "  posterior μ1 (u=+1)    = %.15f\n" mus[1]
    @printf "  posterior μ2 (u=-1)    = %.15f\n" mus[2]
    @printf "  posterior μ3 (u=+1)    = %.15f\n" mus[3]

    return (tag=tag, p0=p0, p_star=p_star, mus=mus, err=err, iters=iters)
end

cara = single_realization(gamma=1.0, utility=:cara,
                          tag="CARA explicit",          u_report=(1.0, -1.0, 1.0))
crra = single_realization(gamma=0.5, utility=:crra,
                          tag="CRRA γ=0.5",             u_report=(1.0, -1.0, 1.0))

println()
println("=== Side-by-side, single realization (1,-1,1), G=5, τ=2.0 ===")
@printf "                     |     CARA      |     CRRA      |\n"
@printf "price (no-learning)  | %.10f  | %.10f  |\n" cara.p0 crra.p0
@printf "price p* (solved)    | %.10f  | %.10f  |\n" cara.p_star crra.p_star
@printf "mu_1 (u=+1)          | %.10f  | %.10f  |\n" cara.mus[1] crra.mus[1]
@printf "mu_2 (u=-1)          | %.10f  | %.10f  |\n" cara.mus[2] crra.mus[2]
@printf "mu_3 (u=+1)          | %.10f  | %.10f  |\n" cara.mus[3] crra.mus[3]
@printf "|residual h(p*)|     | %.3e   | %.3e   |\n" abs(cara.err) abs(crra.err)
