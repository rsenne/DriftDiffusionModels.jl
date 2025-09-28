@testset "MLE recovery (module API) — conditional (with τ)" begin
    rng = MersenneTwister(2024)
    true_model = DriftDiffusionModel(B=1.8, v=0.9, a₀=0.35, τ=0.12)
    init_model = DriftDiffusionModel(B=0.8, v=0.2, a₀=0.6, τ=0.05)

    # simulate data using package API
    N = 1500
    data = [rand(rng, true_model) for _ in 1:N]

    # fit in-place via package API
    DriftDiffusionModels.fit!(init_model, data)

    @test isfinite(init_model.B) && isfinite(init_model.v) && isfinite(init_model.a₀) && isfinite(init_model.τ)
    @test abs(init_model.B - true_model.B) ≤ 0.35
    @test abs(init_model.v - true_model.v) ≤ 0.22
    @test abs(init_model.a₀ - true_model.a₀) ≤ 0.12
    @test abs(init_model.τ - true_model.τ) ≤ 0.08
end
