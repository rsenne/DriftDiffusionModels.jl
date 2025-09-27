unction riemann_sum(f, tmin, tmax, n)
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
prob_right_response(v, B, w; err=1e-12) = riemann_sum(t->wfpt(t, v, B, w, err), 1e-4, 6.0, 50000)

@testset "4-case stimulus × correctness probabilities" begin
    B, v, w = 1.7, 0.6, 0.4

    p_R_resp_given_Rstim = prob_right_response( v, B, w)
    p_L_resp_given_Rstim = 1 - p_R_resp_given_Rstim

    p_R_resp_given_Lstim = prob_right_response(-v, B, w)
    p_L_resp_given_Lstim = 1 - p_R_resp_given_Lstim

    p_right_correct   = p_R_resp_given_Rstim          # r=+1, s=+1
    p_right_incorrect = p_L_resp_given_Rstim          # r=-1, s=+1
    p_left_correct    = p_L_resp_given_Lstim          # r=-1, s=-1
    p_left_incorrect  = p_R_resp_given_Lstim          # r=+1, s=-1

    @test all(x -> isfinite(x) && 0 ≤ x ≤ 1, (p_right_correct, p_right_incorrect, p_left_correct, p_left_incorrect))

    # With balanced stimuli and v>0, overall accuracy > 0.5
    acc = 0.5*(p_right_correct + p_left_correct)
    @test acc > 0.5
end

# Stimulus-coded negative log-likelihood using primitive logdensityof (no wrappers, just composition)
# θ = (logB, v, logit(a0)); σ fixed to 1.0
nll_stimulus(θ, rts, resps, stims) = begin
    logB, v, logit_a = θ
    B = exp(logB)
    a0 = 1/(1 + exp(-logit_a))
    σ = 1.0
    s = 0.0
    @inbounds for i in eachindex(rts)
        veff = v * stims[i]
        s -= logdensityof(B, veff, a0, σ, rts[i], resps[i])
    end
    s
end

@testset "Stimulus-coded likelihood gradient (AD vs FiniteDiff)" begin
    # Small synthetic set without any simulator dependency: pick plausible RTs/choices/stims
    rts   = [0.31, 0.42, 0.28, 0.65, 0.51, 0.37, 0.73, 0.29]
    resps = [ +1,  -1,  +1,  -1,  +1,  +1,  -1,  +1]
    stims = [ +1,  +1,  -1,  -1,  +1,  -1,  +1,  -1]

    θ0 = [log(1.6), 0.6, 0.0]
    g_ad = ForwardDiff.gradient(θ->nll_stimulus(θ, rts, resps, stims), θ0)

    g_fd = FiniteDiff.finite_difference_gradient(θ->nll_stimulus(θ, rts, resps, stims), θ0; relstep=1e-6)
    @test all(isfinite, g_ad) && all(isfinite, g_fd)
    @test isapprox(g_ad, g_fd; rtol=2e-3, atol=2e-3)
end

@testset "Stimulus/response flip invariance" begin
    rts   = [0.33, 0.47, 0.39, 0.58]
    resps = [ +1,  -1,  +1,  -1]
    stims = [ +1,  +1,  -1,  -1]

    θ = [log(1.8), 0.5, 0.2]

    nll1 = nll_stimulus(θ, rts, resps, stims)
    nll2 = nll_stimulus(θ, rts, -resps, -stims)
    @test isapprox(nll1, nll2; rtol=1e-10, atol=1e-12)
end
