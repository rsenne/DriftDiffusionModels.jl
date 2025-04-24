"""
    calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int) -> Float64

Calculate the log-likelihood ratio between a multi-state DDM and a single-state DDM. Importantly, we express this ratio in bits.
"""
function calculate_ll_ratio(ℓℓ::Float64, ℓℓ₀::Float64, n::Int)
    ll_ratio = ℓℓ - ℓℓ₀
    return ll_ratio / (n * log(2))
end

"""
    PriorHMM

Custom HMM for HiddenMarkovModels.jl that extends the standard HMM to support a prior distribtuion over the transition matrix and the initial state distribution.
"""
struct PriorHMM{T,D} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector:{D}
    αₜ::Int # hyperparameter for the Dirichlet distribution over each row of the transition matrix (assuming symmetry)
    αᵢ::Int # hyperparameter for the Dirichlet distribution over the initial state distribution (assuming symmetry)
end

# Necessary for the HMM interface
HiddenMarkovModels.initialization(hmm::PriorHMM) = hmm.init
HiddenMarkovModels.transition_matrix(hmm::PriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::PriorHMM) = hmm.dists

"""
    DensityInterface.logdensityof(hmm::PriorHMM)

Compute the prior loglikelihood of the PriorHMM type. This is the loglikelihood of the transition matrix and initial state distribution given the Dirichlet hyperparameters.
"""
function DensityInterface.logdensityof(hmm::PriorHMM)
    return
end

"""
    StatsAPI.fit!(hmm::PriorHMM, x::vector{DDMResults}, w::Vector{Float64}=ones(length(x)))

Fits the Prior HMM modelto the data using the Baum-Welch algorithm.
"""
function StatsAPI.fit!(hmm::PriorHMM, x::Vector{DDMResult}, w::AbstractVector{<:Real}=ones(length(x)))
    return
end