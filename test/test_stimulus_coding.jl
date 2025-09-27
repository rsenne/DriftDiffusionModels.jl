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

# Probability mass to "Right" response (upper boundary) for a given effective drift.
prob_right_response(v, B, w, τ; err=1e-12) = riemann_sum(t->wfpt(t, v, B, w, τ, err), τ + 1e-4, 6.0, 50000)

@testset "4-case stimulus × correctness probabilities (with τ)" begin
    B, v, w, τ = 1.7, 0.6, 0.5, 0.12

    p_R_resp_given_Rstim = prob_right_response( v, B, w, τ)
    p_L_resp_given_Rstim = 1 - p_R_resp_given_Rstim

    p_R_resp_given_Lstim = prob_right_response(-v, B, w, τ)
    p_L_resp_given_Lstim = 1 - p_R_resp_given_Lstim

    p_right_correct   = p_R_resp_given_Rstim          # r=+1, s=+1
    p_right_incorrect = p_L_resp_given_Rstim          # r=-1, s=+1
    p_left_correct    = p_L_resp_given_Lstim          # r=-1, s=-1
    p_left_incorrect  = p_R_resp_given_Lstim          # r=+1, s=-1

    @test all(x -> isfinite(x) && 0 ≤ x ≤ 1, (p_right_correct, p_right_incorrect, p_left_correct, p_left_incorrect))

    # test symmetry
    @test isapprox(p_right_correct, p_left_correct; atol=1e-6)
    @test isapprox(p_right_incorrect, p_left_incorrect, atol=1e-6)

    # With balanced stimuli and v>0, overall accuracy > 0.5
    acc = p_right_correct + p_left_correct
    @test acc > 0.5

    @test isapprox(p_right_correct + p_left_correct + p_right_incorrect + p_left_incorrect, 1.; atol=1e-3)
end

# Stimulus-coded negative log-likelihood using primitive logdensityof with τ
# θ = (logB, v, logit(a0), τ_unconstrained); σ fixed to 1.0, τ = softplus(τ_unconstrained)
nll_stimulus(θ, rts, resps, stims) = begin
    logB, v, logit_a, τu = θ
    B = exp(logB)
    a0 = 1/(1 + exp(-logit_a))
    τ = log1p(exp(τu)) # softplus to keep τ ≥ 0
    σ = 1.0
    s = 0.0
    @inbounds for i in eachindex(rts)
        veff = v * stims[i]
        s -= logdensityof(B, veff, a0, τ, σ, rts[i], resps[i])
    end
    s
end

@testset "Stimulus-coded likelihood gradient (AD vs FiniteDiff) with τ" begin
    using ForwardDiff, FiniteDiff
    # Small synthetic set without simulator dependency
    rts   = [0.41, 0.52, 0.38, 0.75, 0.61, 0.47, 0.83, 0.49]
    resps = [ +1,  -1,  +1,  -1,  +1,  +1,  -1,  +1]
    stims = [ +1,  +1,  -1,  -1,  +1,  -1,  +1,  -1]

    θ0 = [log(1.6), 0.6, 0.0, log(0.15)]
    g_ad = ForwardDiff.gradient(θ->nll_stimulus(θ, rts, resps, stims), θ0)

    g_fd = FiniteDiff.finite_difference_gradient(θ->nll_stimulus(θ, rts, resps, stims), θ0; relstep=1e-6)
    @test all(isfinite, g_ad) && all(isfinite, g_fd)
    @test isapprox(g_ad, g_fd; rtol=3e-3, atol=3e-3)
end

@testset "Stimulus/response flip invariance (with τ)" begin
    rts   = [0.53, 0.67, 0.59, 0.78]
    resps = [ +1,  -1,  +1,  -1]
    stims = [ +1,  +1,  -1,  -1]

    θ = [log(1.8), 0.5, 0.2, log(0.12)]

    nll1 = nll_stimulus(θ, rts, resps, stims)
    nll2 = nll_stimulus(θ, rts, -resps, -stims)
    @test isapprox(nll1, nll2; rtol=1e-10, atol=1e-12)
end
