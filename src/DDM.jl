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
    wfpt(t, v, a, z, err=1e-8)

Calculate the Wiener First Passage Time (WFPT) density at time t for a drift diffusion model
with drift rate v, boundary separation a, starting point z, and error tolerance err.

This implementation follows the algorithm described in Navarro & Fuss (2009).
"""
function wfpt(t::Real, v::Real, a::Real, z::Real, err::Real=1e-8)
    # Check for valid inputs
    if t <= 0
        return 0.0
    end
    
    # Use normalized time and relative start point
    tt = t / (a^2)
    w = z / a
    
    # Calculate number of terms needed for large t version
    if π * tt * err < 1  # if error threshold is set low enough
        kl = sqrt(-2 * log(π * tt * err) / (π^2 * tt))  # bound
        kl = max(kl, 1 / (π * sqrt(tt)))  # ensure boundary conditions met
    else  # if error threshold set too high
        kl = 1 / (π * sqrt(tt))  # set to boundary condition
    end
    
    # Calculate number of terms needed for small t version
    if 2 * sqrt(2 * π * tt) * err < 1  # if error threshold is set low enough
        ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * π * tt) * err))  # bound
        ks = max(ks, sqrt(tt) + 1)  # ensure boundary conditions are met
    else  # if error threshold was set too high
        ks = 2  # minimal kappa for that case
    end
    
    # Compute f(tt|0,1,w)
    p = 0.0  # initialize density
    if ks < kl  # if small t is better...
        K = ceil(Int, ks)  # round to smallest integer meeting error
        for k in -floor(Int, (K-1)/2):ceil(Int, (K-1)/2)  # loop over k
            p += (w + 2 * k) * exp(-((w + 2 * k)^2) / 2 / tt)  # increment sum
        end
        p /= sqrt(2 * π * tt^3)  # add constant term
    else  # if large t is better...
        K = ceil(Int, kl)  # round to smallest integer meeting error
        for k in 1:K
            p += k * exp(-(k^2) * (π^2) * tt / 2) * sin(k * π * w)  # increment sum
        end
        p *= π  # add constant term
    end
    
    # Convert to f(t|v,a,w)
    return p * exp(-v * a * w - (v^2) * t / 2) / (a^2)
end

"""
    simulateDDM(model::DriftDiffusionModel, dt::FLoat64=1e-5)

Generate a single trial of a drift diffusion model using the Euler-Maruyama method.
"""
function simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5)
    @unpack B, v, a₀, σ = model

    # initialize variables
    t = 0.0
    a = a₀

    while a < B && a > -B
        # run the model forward in time
        a += v * dt + (σ * sqrt(dt) * randn())
        t += dt
    end

    if a >= B
        choice = 1  # upper boundary hit
    else
        choice = -1  # lower boundary hit
    end

    return DDMResult(t, choice)
end

"""
    simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
"""
function simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
    results = Vector{DDMResult}(undef, n)
    @threads for i in 1:n
        results[i] = simulateDDM(model, dt)
    end
    return results
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

function Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)
    @unpack B, v, a₀, σ = model

    p_upper = abs(v) < 1e-10 ? (B + a₀) / (2 * B) : 1 / (1 + exp(-2 * v * a₀ / σ^2))
    p_upper = clamp(p_upper, 0.0, 1.0)

    u = rand(rng)
    choice = u < p_upper ? 1 : -1

    if choice == 1
        μ = (B - a₀) / abs(v)
        λ = (B - a₀)^2 / σ^2
    else
        μ = (B + a₀) / abs(v)
        λ = (B + a₀)^2 / σ^2
    end

    rt = rand(rng, InverseGaussian(μ, λ))
    return DDMResult(rt, choice)
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
    if rt <= 0
        return -Inf
    end

    T = promote_type(TB, TV, TA, TS)
    B, v, a₀, σ = T(B), T(v), T(a₀), T(σ)

    distance = choice == 1 ? B - a₀ : B + a₀

    # Check if distance is non-positive
    if distance <= 0
        return -Inf
    end

    # Reparameterize the parameters of the model to pass to the wfpt function.
    A = 2 * B # boundary separation
    z = α₀ + B

    density = choice == 1 ? wfpt(rt, -v, A, 1 - z) : wfpt(rt, v, A, z)
    return log(density)
end


"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::AbstractVector{<:Real}=ones(length(x)))
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

    