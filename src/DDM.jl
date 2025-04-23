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
    
    return logdensityof(B, v, a₀, σ, rt, choice)
end

function logdensityof(B::TB, v::TV, a₀::TA, σ::TS, rt::Float64, choice::Int) where {TB<:Real, TV<:Real, TA<:Real, TS<:Real}
    # Early return for invalid data
    if rt <= 0
        return -Inf
    end
    
    # Promote types to a common type to avoid type issues in calculations
    T = promote_type(TB, TV, TA, TS)
    B_p, v_p, a₀_p, σ_p = T(B), T(v), T(a₀), T(σ)
    
    # Adjust sign of drift based on choice
    v_adj = choice * v_p
    
    # Calculate distance to the crossed boundary
    boundary = choice * B_p
    distance = boundary - a₀_p
    
    # Compute log density using first passage time distribution
    # for Wiener process with drift
    exponent = -(distance - v_adj * rt)^2 / (2 * σ_p^2 * rt)
    log_normalizer = -0.5 * log(2 * π * σ_p^2 * rt^3)
    
    return log_normalizer + exponent
end

"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))
    @unpack B, v, a₀, σ = model
    
    # Calculate the initial a₀ as a fraction of B
    a₀_frac = a₀ / B
    
    # Define negative log-likelihood function for optimization
    function neg_log_likelihood(params)
        # We optimize B, drift rate, and a₀ as a fraction of B
        B_temp, v_temp, a₀_frac = params
        
        # Calculate actual a₀ based on fraction (always stays within bounds)
        a₀_temp = a₀_frac * B_temp
        
        # Early return for invalid boundary (must be positive)
        if B_temp <= 0
            return convert(typeof(B_temp), Inf)
        end
        
        # Calculate log-likelihood using the raw parameters version
        ll = 0.0
        for i in 1:length(x)
            ll += w[i] * logdensityof(B_temp, v_temp, a₀_temp, σ, x[i].rt, x[i].choice)
        end
        
        # Return negative since optimizers typically minimize
        return -ll
    end
    
    # Set up optimization with reparameterized a₀
    initial_params = [B, v, a₀_frac]
    
    # Add bounds - a₀_frac must be between -1 and 1
    lower_bounds = [0.001, -Inf, -0.999]
    upper_bounds = [Inf, Inf, 0.999]
    
    # Optimize using L-BFGS-B to respect the bounds
    result = optimize(neg_log_likelihood, lower_bounds, upper_bounds, initial_params, Fminbox(LBFGS()), autodiff=:forward)
    
    # Extract the optimized parameters
    optimal_params = Optim.minimizer(result)
    
    # Update the model with new parameter estimates
    model.B = optimal_params[1]
    model.v = optimal_params[2]
    model.a₀ = optimal_params[3] * optimal_params[1]  # Convert fraction back to absolute value
    
    return model
end

    