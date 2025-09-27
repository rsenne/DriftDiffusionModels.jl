# Helper: approximate integral by midpoint rule
function riemann_sum(f, tmin, tmax, n)
    h = (tmax - tmin) / n
    s = 0.0
    t = tmin + 0.5h
    @inbounds for _ in 1:n
        s += f(t) * h
        t += h
    end
    return s
end

@testset "wfpt sanity & normalization (module API with τ)" begin
    v   = 0.5
    B   = 2.0
    w   = 0.3
    τ   = 0.2
    err = 1e-12

    for t in range(1e-4, 5.0; length=50)
        p = wfpt(t, v, B, w, τ, err)
        @test isfinite(p) && p ≥ 0
    end

    f_total = t->wfpt(t, v, B, w, τ, err) + wfpt(t, -v, B, 1-w, τ, err)
    mass = riemann_sum(f_total, τ + 1e-4, 6.0, 30000)
    @test 0.99 ≤ mass ≤ 1.01
end

@testset "logdensity basic invariants (module API with τ)" begin
    B, v, a0, τ, σ = 2.5, 0.7, 0.25, 0.15, 1.0
    ll = logdensityof(B, v, a0, τ, σ, 0.350, +1) + logdensityof(B, v, a0, τ, σ, 0.420, -1)
    @test isfinite(ll)
    @test logdensityof(B, v, a0, τ, σ, -0.1, +1) == -Inf
    # rt <= τ should also not contribute meaningful density; the implementation returns small floor
    @test wfpt(τ - 1e-3, v, B, a0, τ) ≥ 0
end
