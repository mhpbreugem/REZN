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

fmt_mb(b) = @sprintf("%.1f", b / (1024^2))

function maxrss_mb()
    # Linux: /proc/self/status VmRSS in kB; fall back to Sys.maxrss
    try
        for ln in eachline("/proc/self/status")
            if startswith(ln, "VmHWM:")
                parts = split(ln)
                return parse(Float64, parts[2]) / 1024.0
            end
        end
    catch; end
    return NaN
end

function run_G(G::Int;
               picard_iters::Int=1200, picard_tol=1e-13,
               newton_iters::Int=5, newton_tol=1e-11,
               do_newton::Bool=true)
    P = Rezn.Params(G=G, tau=TAU, gamma=GAMMA, umax=UMAX, utility=:crra)
    u = Rezn.build_grid(P)
    idx(x) = argmin(abs.(u .- x))
    i, j, l = idx(U_REPORT[1]), idx(U_REPORT[2]), idx(U_REPORT[3])
    u_hit = (u[i], u[j], u[l])
    unknowns = G^3
    jac_MB = unknowns^2 * 8 / (1024^2)

    @printf "\n=== G=%2d  (N=%d, J ~ %.1f MB dense)  cell (%d,%d,%d) -> u=(%.3f,%.3f,%.3f) ===\n" G unknowns jac_MB i j l u_hit[1] u_hit[2] u_hit[3]
    flush(stdout)

    # Picard
    GC.gc(); b0 = Base.gc_bytes(); t0 = time()
    pic = Rezn.solve_picard(P; maxiters=picard_iters, abstol=picard_tol, alpha=1.0, verbose=false)
    dt_pic = time() - t0
    bytes_pic = Base.gc_bytes() - b0
    rss_pic = maxrss_mb()
    p_pic = pic.P_star[i,j,l]
    mus_pic = Rezn.posteriors_at(i, j, l, p_pic, pic.P_star, u, P)
    Finf_pic = maximum(abs, pic.residual)
    PhiI_pic = isempty(pic.history) ? NaN : pic.history[end]
    oneR2_pic = Rezn.one_minus_R2(pic.P_star, u, TAU)
    @printf "  Picard     : iters=%5d   t=%6.2fs   alloc=%7s MB   peakRSS=%6.0f MB   ‖Φ-I‖∞=%.2e   ‖F‖∞=%.2e   1-R²=%.3e\n" length(pic.history) dt_pic fmt_mb(bytes_pic) rss_pic PhiI_pic Finf_pic oneR2_pic
    @printf "              p*=%.10f   μ=(%.6f, %.6f, %.6f)   PR_gap=%.5f\n" p_pic mus_pic[1] mus_pic[2] mus_pic[3] mus_pic[1]-mus_pic[2]
    flush(stdout)

    pic_summary = (method="Picard", G=G, iters=length(pic.history), dt=dt_pic, bytes=bytes_pic,
                   rss_mb=rss_pic, p=p_pic, mus=mus_pic, Finf=Finf_pic, oneR2=oneR2_pic)

    newton_summary = nothing
    if do_newton
        # Newton warm-start from Picard with LU factorization
        GC.gc(); b0 = Base.gc_bytes(); t0 = time()
        new = Rezn.solve_newton_lu(P; x0=vec(pic.P_star),
                                    maxiters=newton_iters, abstol=newton_tol,
                                    threaded=true, verbose=false)
        dt_new = time() - t0
        bytes_new = Base.gc_bytes() - b0
        rss_new = maxrss_mb()
        p_new = new.P_star[i,j,l]
        mus_new = Rezn.posteriors_at(i, j, l, p_new, new.P_star, u, P)
        Finf_new = maximum(abs, new.residual)
        tm = new.timings
        oneR2_new = Rezn.one_minus_R2(new.P_star, u, TAU)
        @printf "  Newton-LU  : iters=%5d   t=%6.2fs   alloc=%7s MB   peakRSS=%6.0f MB   ‖F‖∞=%.2e   1-R²=%.3e\n" length(new.history)-1 dt_new fmt_mb(bytes_new) rss_new Finf_new oneR2_new
        @printf "              breakdown: jac=%.2fs  lu=%.2fs  solve=%.2fs  line-search=%.2fs  (threads=%d)\n" tm.jac tm.lu tm.solve tm.ls Threads.nthreads()
        @printf "              p*=%.10f   μ=(%.6f, %.6f, %.6f)   PR_gap=%.5f\n" p_new mus_new[1] mus_new[2] mus_new[3] mus_new[1]-mus_new[2]
        flush(stdout)
        newton_summary = (method="Newton-LU", G=G, iters=length(new.history)-1, dt=dt_new,
                          bytes=bytes_new, rss_mb=rss_new, p=p_new, mus=mus_new, Finf=Finf_new,
                          oneR2=oneR2_new, timings=tm)
    end

    return pic_summary, newton_summary, jac_MB
end

println("Julia threads = ", Threads.nthreads())
println("CRRA γ=$(GAMMA), τ=$(TAU), target signal (u₁,u₂,u₃)=(1,-1,1)")

results = []
# Newton feasibility by dense Jacobian memory:
# G=5 → 0.12 MB, G=9 → 4 MB, G=13 → 37 MB, G=17 → 185 MB — all OK
# but Jacobian BUILD TIME scales as G^7 (n F-evals * O(G^4) work each)
for G in (5, 9, 13, 17)
    # Newton-LU feasibility budget. FD Jacobian is the bottleneck —
    # O(G^7) flops per iter, so per Newton iter: G=5 <1s, G=9 ~10s,
    # G=13 ~4min, G=17 ~30min. Run Newton at G=5,9,13; skip at G=17.
    do_newton = G <= 13
    sp, sn, jmb = run_G(G; do_newton=do_newton)
    push!(results, (sp, sn, jmb))
end

println("\n" * "="^127)
@printf "CRRA γ=%.2f, τ=%.2f, target (1,-1,1) — Julia threads = %d\n" GAMMA TAU Threads.nthreads()
println("="^127)
@printf "%3s | %-10s | %5s | %8s | %10s | %10s | %10s | %10s | %15s | %15s | %8s\n" "G" "method" "iters" "time(s)" "alloc(MB)" "peakRSS(MB)" "‖F‖∞" "1-R²" "p*" "μ₁" "PR_gap"
println("-"^140)
for (sp, sn, jmb) in results
    @printf "%3d | %-10s | %5d | %8.2f | %10s | %10.0f | %10.2e | %10.3e | %15.10f | %15.10f | %8.5f\n" sp.G sp.method sp.iters sp.dt fmt_mb(sp.bytes) sp.rss_mb sp.Finf sp.oneR2 sp.p sp.mus[1] sp.mus[1]-sp.mus[2]
    if sn !== nothing
        @printf "%3d | %-10s | %5d | %8.2f | %10s | %10.0f | %10.2e | %10.3e | %15.10f | %15.10f | %8.5f\n" sn.G sn.method sn.iters sn.dt fmt_mb(sn.bytes) sn.rss_mb sn.Finf sn.oneR2 sn.p sn.mus[1] sn.mus[1]-sn.mus[2]
    end
    println("-"^140)
end
