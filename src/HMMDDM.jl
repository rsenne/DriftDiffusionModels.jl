"""
    calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int) -> Float64

Calculate the log-likelihood ratio between a multi-state DDM and a single-state DDM. Importantly, we express this ratio in bits.
"""
function calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int)
    ll_ratio = ℓℓ - ℓℓ₀
    return ll_ratio / (n * log(2))
end


"""
    PriorHMM(init, trans, dists; αₜ = 1.0, αᵢ = 1.0)

A Hidden Markov Model that places **symmetric Dirichlet priors** on the
transition matrix rows (hyper‑parameter `αₜ`) and on the initial state
probabilities (hyper‑parameter `αᵢ`).  Setting `αₜ = αᵢ = 1` recovers the
standard maximum‑likelihood estimate.
"""
struct PriorHMM{T<:Real,D} <: HiddenMarkovModels.AbstractHMM
    init  :: Vector{T}      # π
    trans :: Matrix{T}      # A (rows sum to 1)
    dists :: Vector{D}      # emission distributions
    αₜ    :: T              # symmetric Dirichlet mass for each row of A
    αᵢ    :: T              # symmetric Dirichlet mass for π

    function PriorHMM(init::Vector{T}, trans::Matrix{T}, dists::Vector{D};
                      αₜ::T = one(T), αᵢ::T = one(T)) where {T<:Real,D}
        K = length(init)
        @assert size(trans, 1) == K && size(trans, 2) == K  "Transition matrix must be K×K"
        @assert abs(sum(init) - one(T)) < 1e-8              "Initial probabilities must sum to 1"
        @assert all(abs.(sum(trans; dims = 2) .- 1) .< 1e-8) "Each row of transition matrix must sum to 1"
        new{T,D}(copy(init), copy(trans), deepcopy(dists), αₜ, αᵢ)
    end
end

##########################  Required interface  ##########################

HiddenMarkovModels.initialization(hmm::PriorHMM)      = hmm.init
HiddenMarkovModels.transition_matrix(hmm::PriorHMM)   = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::PriorHMM)   = hmm.dists

##########################  Prior log‑density  ###########################

"""
    DensityInterface.logdensityof(hmm::PriorHMM)

Log‑density of the **prior** `p(π, A | αᵢ, αₜ)`.
For each K‑dimensional symmetric Dirichlet with concentration `α` the
log normalisation constant is
```
log B(α) = log Γ(K⋅α) − K·log Γ(α).
```
The density is then
```
log p(π) = log B(αᵢ) + Σ_k (αᵢ−1) log π_k,
log p(A) = Σ_i [ log B(αₜ) + Σ_j (αₜ−1) log A_{ij} ].
```
"""
function DensityInterface.logdensityof(hmm::PriorHMM)
    K    = length(hmm.init)
    αₜ   = hmm.αₜ
    αᵢ   = hmm.αᵢ

    logB_T = loggamma(K*αₜ) - K*loggamma(αₜ)
    logB_π = loggamma(K*αᵢ) - K*loggamma(αᵢ)

    # transition part
    lp = K * logB_T + sum((αₜ - 1) .* log.(hmm.trans))
    # initial part
    lp += logB_π + sum((αᵢ - 1) .* log.(hmm.init))
    return lp
end

##########################  MAP Baum‑Welch  ##############################

"""
    StatsAPI.fit!(hmm::PriorHMM, seqs; max_iter = 100, atol = 1e-6, w = ones(length(seqs)))

Expectation–Maximisation (Baum–Welch) **with Dirichlet pseudocounts**.
Adding `αᵢ` and `αₜ` to the expected counts yields the Maximum A Posteriori
(MAP) estimate under the symmetric Dirichlet priors.
"""
function StatsAPI.fit!(hmm::PriorHMM, seqs::Vector{<:AbstractVector};
                       w::AbstractVector{<:Real} = ones(length(seqs)),
                       max_iter::Int = 100,
                       atol::Float64 = 1e-6)
    K      = length(hmm.init)
    π      = copy(hmm.init)
    A      = copy(hmm.trans)
    dists  = deepcopy(hmm.dists)
    αᵢ, αₜ = hmm.αᵢ, hmm.αₜ

    for _ in 1:max_iter
        γ₁      = zeros(eltype(π), K)          # expected initial counts
        ξ_tot   = zeros(eltype(π), K, K)       # expected transition counts
        suff    = [HiddenMarkovModels.zero_stats(d) for d in dists]
        ll_prev = 0.0

        ################  E‑STEP  ################
        for (seq, wt) in zip(seqs, w)
            α, c = HiddenMarkovModels.forward(seq, π, A, dists)
            β     = HiddenMarkovModels.backward(seq, A, dists, c)
            Tseq  = length(seq)

            # γ_t(i) ∝ α_t(i) β_t(i)
            γ = α .* β
            normγ = sum(γ; dims = 1)
            γ ./= normγ

            γ₁      .+= wt .* vec(γ[:, 1])
            ll_prev += wt * sum(log.(c))

            # ξ_t(i,j) ∝ α_t(i) A_{ij} p(y_{t+1}|j) β_{t+1}(j)
            for t in 1:(Tseq - 1)
                ξ = A .* (α[:, t] .* HiddenMarkovModels.pdf.(dists, seq[t+1])') .* β[:, t+1]'
                ξ ./= sum(ξ)
                ξ_tot .+= wt .* ξ
            end

            # accumulate emission sufficient statistics
            for k in 1:K
                HiddenMarkovModels.accum_stats!(suff[k], seq, vec(γ[k, :]), wt)
            end
        end

        ################  M‑STEP  ################
        γ₁    .+= αᵢ
        ξ_tot .+= αₜ

        new_π = γ₁ / sum(γ₁)
        new_A = ξ_tot ./ sum(ξ_tot; dims = 2)

        for k in 1:K
            HiddenMarkovModels.update!(dists[k], suff[k])
        end

        if maximum(abs.(π .- new_π)) < atol && maximum(abs.(A .- new_A)) < atol
            π, A = new_π, new_A
            break
        end
        π, A = new_π, new_A
    end

    hmm.init  .= π
    hmm.trans .= A
    hmm.dists .= dists
    return hmm
end