export crossvalidate

"""
    randomDDM()
    
Generate a random Drift Diffusion Model.
"""
function randomDDM()
    # Generate random parameters for the DDM
    B = rand(Uniform(0.1, 20.0)) # Bound Height
    v = rand(Uniform(1e-3, 9.0))
    a₀ = rand(Normal(0.0, 1.0))
    σ = 1.0 # fixed for identifiability
    return DriftDiffusionModel(B, v, a₀, σ)
end

"""
    crossvalidate(ddm::AbstractHMM, x::Vector{Vector{DDMResult}}, folds::Int=5, n_states::Int=5)

Perform k-fold cross validation on an HMM-DDM.
"""
function crossvalidate(x::Vector{Vector{DDMResult}}, n_folds::Int=5, n_states::Int=5)
    # Shuffle the sessions of data
    shuffled_data = shuffle(x)

    # Calculate the size of each fold
    fold_size = ceil(Int, length(x) / n_folds)

    # Create the folds
    folds = Vector{Vector{Vector{DDMResult}}}(undef, n_folds)
    for i in 1:n_folds
        start = (i - 1) * fold_size + 1
        stop = min(i * fold_size, length(x))
        folds[i] = shuffled_data[start:stop]
    end
    
    # Create dict to hold results
    cvresults = Dict()
    
    # Test different numbers of states
    for n in 1:n_states
        cvresults[n] = Vector{Float64}(undef, n_folds)
        
        # For each fold
        for i in 1:n_folds
            # Create a fresh model for each fold
            init_guess = rand(Dirichlet(10 .* ones(n)))
            trans_guess = Matrix{Float64}(undef, n, n)
            for row in eachrow(trans_guess)
                row .= rand(Dirichlet(10 .* ones(n)))
            end
            
            # Create a set number of DDMs
            ddms_guess = [randomDDM() for _ in 1:n]
            hmm_guess = HMM(init_guess, trans_guess, ddms_guess)
            
            # Create the train and test set
            train_indices = setdiff(1:n_folds, i)  # All folds except the current test fold
            train_set = vcat(folds[train_indices]...)
            test_set = folds[i]

            # Concatenate the train and test sets, also create seq_ends for HiddenMarkovModels.jl
            concatenated_training = reduce(vcat, train_set)
            train_seq_ends = cumsum(length.(train_set))

            concatenated_test = reduce(vcat, test_set)
            test_seq_ends = cumsum(length.(test_set))

            # Fit model
            hmm_est, lls = baum_welch(hmm_guess, concatenated_training; seq_ends=train_seq_ends)

            # Calculate the loglikelihood of the test set
            γ, ml = forward(hmm_est, concatenated_test; seq_ends=test_seq_ends)

            # Save the log likelihood results
            cvresults[n][i] = ml

            println("Fold $i for $n states done.")
        end
    end
    return cvresults
end