"""
    randomDDM()
    
Generate a random Drift Diffusion Model.
"""
function randomDDM()
    # Generate random parameters for the DDM
    B = rand(Uniform(0.1, 20.0)) # Boundary Separation
    v = rand(Uniform(1e-3, 9.0))
    
    # Generate a₀ as a fraction of B to ensure proper bounds
    a₀ = rand(Uniform(-0.9, 0.9))  # Keep a₀ within ±90% of boundary
    
    σ = 1.0 # fixed for identifiability
    return DriftDiffusionModel(B, v, a₀, σ)
end

"""
    crossvalidate(x;
                 n_folds::Int = 5,
                 n_states::Int = 5,
                 n_iter::Int  = 5,
                 rng           = Random.GLOBAL_RNG)

k-fold cross-validation for an HMM-DDM with multiple random restarts.

Returns a `Dict{Int,Matrix{Float64}}` such that `cv[n][iter,fold]`
stores the mean log-likelihood of *fold* in restart *iter* for an n-state model.
"""
function crossvalidate(x::Vector{Vector{DDMResult}};
                       n_folds::Int = 5,
                       n_states::Int = 5,
                       n_iter::Int  = 5,
                       rng           = Random.GLOBAL_RNG)

    ######## 0. build folds once ########
    shuffled  = shuffle(rng, x)
    fold_size = ceil(Int, length(x) / n_folds)
    folds = [shuffled[((i-1)*fold_size+1):min(i*fold_size, length(shuffled))] for i in 1:n_folds]

    ######## 1. output containers ########
    ll   = Dict{Int,Matrix{Float64}}()              # log-likelihoods
    for n in 1:n_states
        ll[n] = Matrix{Float64}(undef, n_iter, n_folds)
    end
    nobs = Matrix{Int}(undef, n_iter, n_folds)      # # observations

    ######## 2. iterate ########
    for n in 1:n_states
        @info "⇢ evaluating $n hidden state(s)"
        for iter in 1:n_iter
            for fold in 1:n_folds
                ###### build random initial HMM ######
                init_guess  = rand(rng, Dirichlet(10 .* ones(n)))
                trans_guess = reduce(vcat, transpose.([rand(rng, Dirichlet(10 .* ones(n))) for _ in 1:n]))
                ddms_guess  = [randomDDM() for _ in 1:n]

                # set priors
                αᵢ = ones(n)
                αₜ = ones(n, n)
                αₜ[diagind(αₜ)] .= 10.0 # sticky prior

                hmm_guess   = PriorHMM(init_guess, trans_guess, ddms_guess, αₜ, αᵢ)

                ###### split data ######
                train_idx = setdiff(1:n_folds, fold)
                train_set = vcat(folds[train_idx]...)
                test_set  = folds[fold]

                concat_train = reduce(vcat, train_set)
                train_ends   = cumsum(length.(train_set))

                concat_test  = reduce(vcat, test_set)
                test_ends    = cumsum(length.(test_set))
                ##### train and test ######
                try
                    hmm_hat, _ = baum_welch(hmm_guess, concat_train; seq_ends=train_ends)
                
                    # 2. score test set
                    _, ml      = forward(hmm_hat, concat_test; seq_ends=test_ends)
                    ll[n][iter,fold] = sum(ml)
                
                catch err
                    @warn "Baum–Welch failed for n=$n, iter=$iter, fold=$fold ⇒ $(err)"
                    ll[n][iter,fold] = -Inf           # mark the run as unusable
                end
                
                # store nobs once
                if n == 1
                    nobs[iter,fold] = length(concat_test)
                end
            end
        end
    end
    return (ll = ll, nobs = nobs)
end

