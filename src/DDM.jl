export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!

"""
    DriftDiffusionModel

Implement a simple drift diffusion model: da/dt = v + σdW.
"""
mutable struct DriftDiffusionModel
    B::Float64 #Bound Height
    v::Float64 # Drift Rate
    a₀::Float64 # Initial Accumulation
    σ::Float64 # Noise--set to 1.0 be default for identifiability
end

function DriftDiffusionModel(;
    B::Float64=10.0, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.0, # Initial Accumulation
    σ::Float64=1.0 # Noise--set to 1.0 be default for identifiability
) 
    return DriftDiffusionModel(B, v, a₀, σ)
end

"""
    DDMResult

A tuple of RT and choice. The first element is the RT, the second is the choice.
"""
struct DDMResult
    rt::Float64 # RT
    choice::Int # Choice (-1 or 1)
end

function DDMResult(;rt::Float64, choice::Int)
    return DDMResult(rt, choice)
end

"""
    Random.rand(model::DriftDiffusionModel)

Generates RT and choices from a drift diffusion model. 
"""
function Random.rand(model::DriftDiffusionModel)
    @unpack B, v, a₀, σ = model
    
    # For symmetric boundaries at +B and -B, the probability of hitting 
    # the upper boundary can be calculated as:
    if abs(v) < 1e-10
        # With zero drift, probability depends only on starting position
        p_upper = (B + a₀) / (2 * B)
    else
        # Standard formula for probability of hitting upper boundary with symmetric bounds
        p_upper = 1 / (1 + exp(-2 * v * a₀ / σ^2))
    end
    
    # Ensure p_upper is a valid probability
    p_upper = clamp(p_upper, 0.0, 1.0)
    
    # Determine choice based on probability
    u = rand()
    choice = u < p_upper ? 1 : -1
    
    # Calculate parameters for the first passage time distribution
    if choice == 1
        # For upper boundary
        μ = (B - a₀) / abs(v)
        λ = (B - a₀)^2 / σ^2
    else
        # For lower boundary
        μ = (B + a₀) / abs(v)
        λ = (B + a₀)^2 / σ^2
    end
    
    # Sample from the inverse Gaussian distribution
    ig = InverseGaussian(μ, λ)
    rt = rand(ig)
    
    return DDMResult(rt, choice)
end

"""
    Random.rand(model::DriftDiffusionModel, n::Int)

Generates n trials (RT and choices) from a drift diffusion model.
Returns a vector of DDMResult objects.
"""
function Random.rand(model::DriftDiffusionModel, n::Int)
    results = Vector{DDMResult}(undef, n)
    for i in 1:n
        results[i] = rand(model)
    end
    return results
end

# Tell Density Interface that DDMResult is a distribution
DensityInterface.DensityKind(::DriftDiffusionModel) = HasDensity()

"""
    DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a drift diffusion model given a DDMResult.
"""
function DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)
    @unpack B, v, a₀, σ = model
    @unpack rt, choice = x
    
    # Early return for invalid data
    if rt <= 0
        return -Inf
    end
    
    # Adjust sign of drift based on choice
    v_adj = choice * v
    
    # Calculate distance to the crossed boundary
    boundary = choice * B
    distance = boundary - a₀
    
    # Compute log density using first passage time distribution
    # for Wiener process with drift
    exponent = -(distance - v_adj * rt)^2 / (2 * σ^2 * rt)
    log_normalizer = -0.5 * log(2 * π * σ^2 * rt^3)
    
    return log_normalizer + exponent
end

"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))
    @unpack B, v, a₀, σ = model
    
    # Define negative log-likelihood function for optimization
    function neg_log_likelihood(params)
        # Now we optimize boundary, drift rate, and starting point
        B_temp, v_temp, a₀_temp = params
        
        # Early return for invalid boundary (must be positive)
        if B_temp <= 0
            return Inf
        end
        
        # Early return if starting point is outside boundaries
        if abs(a₀_temp) >= B_temp
            return Inf
        end
        
        # Create temporary model with current parameter estimates
        temp_model = DriftDiffusionModel(B=B_temp, v=v_temp, a₀=a₀_temp, σ=σ)
        
        # Calculate weighted log-likelihood across all observations
        ll = 0.0
        for i in 1:length(x)
            ll += w[i] * logdensityof(temp_model, x[i])
        end
        
        # Return negative since optimizers typically minimize
        return -ll
    end
    
    # Set up optimization
    initial_params = [B, v, a₀]  # Start with current values
    
    # Add lower bounds to prevent negative boundary values
    lower_bounds = [0.001, -Inf, -B + 0.001]  # Prevent B = 0, no constraint on v, prevent a₀ from being at or beyond boundary
    upper_bounds = [Inf, Inf, B - 0.001]      # No upper limit on B or v, prevent a₀ from being at or beyond boundary
    
    # Optimize using L-BFGS-B to respect the bounds
    result = optimize(neg_log_likelihood, lower_bounds, upper_bounds, initial_params, Fminbox(LBFGS()), autodiff=:forward)
    
    # Extract the optimized parameters
    optimal_params = Optim.minimizer(result)
    
    # Update the model with new parameter estimates
    model.B = optimal_params[1]
    model.v = optimal_params[2]
    model.a₀ = optimal_params[3]
    
    return model
end

    