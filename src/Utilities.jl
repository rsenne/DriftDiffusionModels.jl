export init_hmm_ddm

# Utilities for fitting a DDM-HMM model.

"""
    fit_global_ddm(data::Vector{DDMResult})

Fit a single DDM to all data using MLE.

# Arguments
- `data::Vector{DDMResult}`: Vector of DDMResult observations.

# Returns
- `model::DriftDiffusionModel`: Fitted DDM model.
"""
 function fit_global_ddm(data::Vector{DDMResult})
     model = DriftDiffusionModel(; τ=0.1)  # initial guess, all other values default
     StatsAPI.fit!(model, data)
     return model
 end


"""
    init_ddm_emissions(global_ddm::DriftDiffusionModel, K::Int)


Initialize a vector of DDM emissions for an HMM by perturbing
parameters of a global DDM.

# Arguments
- `global_ddm::DriftDiffusionModel`: Fitted global DDM model.
- `K::Int`: Number of hidden states.

# Returns
- `emissions::Vector{DriftDiffusionModel}`: Vector of DDM emissions.
"""
 function init_ddm_emissions(global_ddm::DriftDiffusionModel,
                             K::Int)
     emissions = Vector{DriftDiffusionModel}(undef, K)
     for i in 1:K
         B_perturbed  = global_ddm.B  * (1.0 + 0.1 * randn())
         v_perturbed  = global_ddm.v  * (1.0 + 0.1 * randn())
         a₀_perturbed = clamp(global_ddm.a₀ * (1.0 + 0.1 * randn()), 0.1, 0.9)
         τ_perturbed  = max(global_ddm.τ * (1.0 + 0.1 * randn()), 0.01)

         emissions[i] = DriftDiffusionModel(B_perturbed, v_perturbed,
                                           a₀_perturbed, τ_perturbed)
     end
     return emissions
 end

"""
    init_pi(K::Int)

Sample initial state distribution from uniform Dirichlet prior.
"""
function init_pi(K::Int)
    α_init = ones(K)
    return rand(Dirichlet(α_init))
end

"""
    init_trans(K::Int; stay_prob = 0.95)


Initialize a sticky transition matrix.
"""
function init_trans(K::Int; stay_prob = 0.95)
    A = fill((1 - stay_prob) / (K - 1), K, K)
    for k in 1:K
        A[k, k] = stay_prob
    end
    return A
end

"""
    init_dirichlet_priors(K::Int;
                          α_sticky = 2.0,
                          α_offdiag = 1.0,
                          α_init_val = 1.0)

Initialize Dirichlet hyper-parameters for HMM priors.
"""
function init_dirichlet_priors(K::Int;
                               α_sticky = 2.0,
                               α_offdiag = 1.0,
                               α_init_val = 1.0)
    α_trans = fill(α_offdiag, K, K)
    for k in 1:K
        α_trans[k, k] = α_sticky
    end
    α_init  = fill(α_init_val, K)
    return α_trans, α_init
end

"""
    init_hmm_ddm(data::Vector{DDMResult}, K::Int;
                 stay_prob = 0.95,
                 α_sticky = 2.0,
                 α_offdiag = 1.0,
                 α_init_val = 1.0)

Initialize a PriorHMM with DDM emissions and Dirichlet priors.
"""
function init_hmm_ddm(data::Vector{DDMResult}, K::Int;
                      stay_prob = 0.95,
                      α_sticky = 2.0,
                      α_offdiag = 1.0,
                      α_init_val = 1.0)

    global_ddm = fit_global_ddm(data)
    emissions  = init_ddm_emissions(global_ddm, K)
    init_dist  = init_pi(K)
    trans_mat  = init_trans(K; stay_prob=stay_prob)
    α_trans, α_init = init_dirichlet_priors(K;
                                            α_sticky=α_sticky,
                                            α_offdiag=α_offdiag,
                                            α_init_val=α_init_val)

    return PriorHMM(init_dist, trans_mat, emissions; α_trans = α_trans, α_init = α_init)
end