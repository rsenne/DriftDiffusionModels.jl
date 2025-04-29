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
    PriorHMM(init, trans, dists, αₜ, αᵢ)

Hidden‑Markov model whose **rows of the transition matrix** and **initial
state probabilities** have symmetric Dirichlet priors.  The
hyper‑parameters are:
* `αₜ` – added to every element of the *transition* rows (Dirichlet mass)
* `αᵢ` – added to every element of the *initial* distribution

Both the keyword and positional‑argument constructors are provided for
convenience; giving no hyper‑parameters defaults to 1 (= uniform prior).
"""
struct PriorHMM{T<:Real,D} <: HiddenMarkovModels.AbstractHMM
    init  :: Vector{T}      # π – initial probabilities (∑ = 1)
    trans :: Matrix{T}      # A – transition matrix (rows sum to 1)
    dists :: Vector{D}      # length‑K emission distributions
    αₜ    :: T              # transition‑row Dirichlet mass
    αᵢ    :: T              # initial‑state Dirichlet mass

    function PriorHMM(init::Vector{T}, trans::Matrix{T}, dists::Vector{D};
                      αₜ::T = one(T), αᵢ::T = one(T)) where {T<:Real,D}
        K = length(init)
        @assert size(trans, 1) == K && size(trans, 2) == K  "trans must be K×K"
        @assert abs(sum(init) - one(T)) < 1e-8              "init must sum to 1"
        @assert all(abs.(sum(trans; dims = 2) .- 1) .< 1e-8) "each row of trans must sum to 1"
        new{T,D}(copy(init), copy(trans), deepcopy(dists), αₜ, αᵢ)
    end
end

"""  Positional‑argument convenience wrapper.  """
function PriorHMM(init::Vector{T}, trans::Matrix{T}, dists::Vector{D}, αₜ::T, αᵢ::T) where {T<:Real,D}
    PriorHMM(init, trans, dists; αₜ = αₜ, αᵢ = αᵢ)
end

Base.length(hmm::PriorHMM) = length(hmm.init)

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
    StatsAPI.fit!(hmm::PriorHMM,
                  fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
                  obs_seq::AbstractVector;
                  seq_ends)

Baum–Welch M‑step **compatible with HiddenMarkovModels.jl**.  The arrays
`fb_storage.γ` and `fb_storage.ξ` already contain the expected state and
transition counts from the forward–backward pass.  We fold **Dirichlet
pseudocounts** `αᵢ` and `αₜ` into those counts before renormalising.
"""
function StatsAPI.fit!(hmm::PriorHMM,
                       fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
                       obs_seq::AbstractVector; seq_ends)
    K = length(hmm)

    ####################  accumulate counts  ####################
    init_counts  = fill(hmm.αᵢ, K)            # prior pseudocounts for π
    trans_counts = fill(hmm.αₜ, K, K)         # prior pseudocounts for A

    for k in eachindex(seq_ends)
        t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
        init_counts  .+= fb_storage.γ[:, t1]
        trans_counts .+= sum(fb_storage.ξ[t1:t2])
    end

    ####################  renormalise  ##########################
    hmm.init  .= init_counts ./ sum(init_counts)
    hmm.trans .= trans_counts ./ sum(trans_counts; dims = 2)

    ####################  emissions  ############################
    for i in 1:K
        weight_seq = fb_storage.γ[i, :]
        StatsAPI.fit!(hmm.dists[i], obs_seq, weight_seq)
    end

    @assert HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end