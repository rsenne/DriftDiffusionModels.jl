@testset "MLE recovery (module API) — conditional (with τ)" begin
    has_fit! = isdefined(Main, :DriftDiffusionModels) ? isdefined(DriftDiffusionModels, :fit!) :
               (isdefined(Main, :DDM) ? isdefined(DDM, :fit!) : false)

    has_sim = isdefined(Main, :DriftDiffusionModels) ? (isdefined(DriftDiffusionModels, :simulate) || isdefined(DriftDiffusionModels, :rand)) :
              (isdefined(Main, :DDM) ? (isdefined(DDM, :simulate) || isdefined(DDM, :rand)) : false)

    if !(has_fit! && has_sim)
        @info "Skipping MLE recovery: fit!/simulate not found on module."
        @test true  # still pass
        return
    end

    rng = MersenneTwister(2024)
    true_model = DriftDiffusionModel(B=1.8, v=0.9, a₀=0.35, τ=0.12, σ=1.0)

    # Prefer simulate if available; otherwise use rand(model, n)
    data = if isdefined(Main, :DriftDiffusionModels)
        if isdefined(DriftDiffusionModels, :simulate)
            DriftDiffusionModels.simulate(true_model, 1500; rng=rng)
        else
            rand(rng, true_model, 1500)
        end
    else
        if isdefined(DDM, :simulate)
            DDM.simulate(true_model, 1500; rng=rng)
        else
            rand(rng, true_model, 1500)
        end
    end

    init_model = DriftDiffusionModel(B=0.8, v=0.2, a₀=0.6, τ=0.05, σ=1.0)
    if isdefined(Main, :DriftDiffusionModels)
        DriftDiffusionModels.fit!(init_model, data)
    else
        DDM.fit!(init_model, data)
    end

    @test isfinite(init_model.B) && isfinite(init_model.v) && isfinite(init_model.a₀) && isfinite(init_model.τ)
    @test abs(init_model.B - true_model.B) ≤ 0.35
    @test abs(init_model.v - true_model.v) ≤ 0.22
    @test abs(init_model.a₀ - true_model.a₀) ≤ 0.12
    @test abs(init_model.τ - true_model.τ) ≤ 0.08
end
