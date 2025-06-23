"""
    DriftDiffusionModel

Implement a simple drift diffusion model: da/dt = v + σdW.
"""
abstract type EmissionModel end

mutable struct DriftDiffusionModel <: EmissionModel
    B::Float64 # Boundary Separation
    v::Float64 # Drift Rate
    a₀::Float64 # Initial Accumulation: parameterized as a fraction of B
    τ::Float64 # Non-decision time
    σ::Float64 # Noise--set to 1.0 be default for identifiability
end

function DriftDiffusionModel(;
    B::Float64=5.0, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.5, # Initial Accumulation
    τ::Float64=0.0, # Non-decision time
    σ::Float64=1.0 # Noise--set to 1.0 be default for identifiability
) 
    return DriftDiffusionModel(B, v, a₀, τ, σ)
end

mutable struct UniformEmission <: EmissionModel
    a::Float64
    b::Float64 
end

abstract type AbstractResult end

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

function UniformResult(;rt::Float64, choice::Int)
    return DDMResult(rt, choice)
end

Base.eltype(::DriftDiffusionModel) = DDMResult
Base.eltype(::UniformEmission) = DDMResult

"""
    wfpt(t, v, B, z, τ, err=1e-8)

Calculate the Wiener First Passage Time (WFPT) density at time t for a drift diffusion model
with drift rate v, boundary separation B, starting point α₀, non-decision time τ, and error tolerance err.

This implementation follows the algorithm described in Navarro & Fuss (2009).
"""
function wfpt(t::TB, v::TV, B::TA, w::TT, τ::TS; err::Float64=1e-12
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real, TS<:Real}
    # Check for valid inputs (pass t = 0 for sigmoid later)
    if t < τ
        return 0
    end
    
    # Use normalized time and relative start point
    tt = (t - τ) / (B^2)
    
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
    
    # Convert to f(t|v,B,w)
    density = p * exp(-v * B * w - (v^2) * (t - τ) / 2) / (B^2)
    return max(density, 1e-12)  # ensure non-negative density (occasionaly generates neg values e.g., -1e-21) (maybe return +ϵ instead?)
end

"""
    simulateDDM(model::DriftDiffusionModel, dt::FLoat64=1e-5)

Generate a single trial of a drift diffusion model using the Euler-Maruyama method.
"""
function simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5, rng::AbstractRNG=Random.default_rng())
    @unpack B, v, a₀, τ, σ = model

    # initialize variables
    t = 0.0
    a = a₀ * B  # initial accumulation

    while a < B && a > 0
        # run the model forward in time
        if t < τ
            t += dt
        else
            a += v * dt + (σ * sqrt(dt) * randn(rng))
            t += dt
        end
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
Uniform rand function 
"""
function Random.rand(rng::AbstractRNG, model::UniformEmission)
    rt = rand(rng, Uniform(model.a, model.b))      
    choice = rand(rng, Bool) ? 1 : -1                   
    return UniformResult(rt=rt, choice=choice)
end

"""
    Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)

Generate a single trial of the drift diffusion model using the Euler-Maruyama method--needed for HiddenMarkovModels.jl
"""
function Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)
    # Generate a single trial of the drift diffusion model
    return simulateDDM(model, 1e-6, rng)
end

# Tell Density Interface that DDMResult is a distribution
DensityInterface.DensityKind(::DriftDiffusionModel) = HasDensity()
DensityInterface.DensityKind(::UniformEmission) = HasDensity()

"""
    DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a drift diffusion model given a DDMResult.
"""
function DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)
    @unpack B, v, a₀, τ, σ = model
    @unpack rt, choice = x
    
    return logdensityof(B, v, a₀, τ, σ, rt, choice)
end

function logdensityof(
    B::TB, v::TV, a₀::TA, τ::TT, σ::TS, rt::Float64, choice::Int; 
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real, TS<:Real}
    if rt <= 0
        return -Inf
    end

    T = promote_type(TB, TV, TA, TT, TS)
    B, v, a₀, τ, σ = T(B), T(v), T(a₀), T(τ), T(σ)

    # determine which version of the wpft to use: upper or lower boundary
    v, w = choice == 1 ? -v : v, choice == 1 ? 1 - a₀ : a₀


    # calculate the Wiener first passage time density
    density = wfpt(rt, v, B, w, τ)
    logdens = log(density)

    # check if density is Inf (i.e., log(0)) and return a very large value if so
    return isinf(logdens) ? -1e16 : logdens
end


function DensityInterface.logdensityof(model::UniformEmission, x::DDMResult)
    return logdensityof(model, x.rt)
end

function logdensityof(model::UniformEmission, rt::Float64)
    if model.a <= rt <= model.b
        width = model.b - model.a
        return width > 0 ? -log(width) : -1e16
    else
        return -Inf
    end
end

"""
Fallbacks for HMM 
"""
penalty = -1e6

function logdensityof(model::DriftDiffusionModel, x::AbstractResult)
    return x isa DDMResult ? DensityInterface.logdensityof(model, x) : penalty
end

"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::AbstractVector{<:Real}=ones(length(x)))
    @unpack B, v, a₀, τ, σ = model
    
    # Define negative log-likelihood function for optimization
    function neg_log_likelihood(params)
        # We optimize B, drift rate, and a₀ as a fraction of B
        B_temp, v_temp, τ_temp = params
        
        # Early return for invalid boundary (must be positive)
        if B_temp < 0
            return convert(typeof(B_temp), Inf)
        end
        
        # Calculate log-likelihood using the raw parameters version
        ll = 0.0
        for i in 1:length(x)
            ll += w[i] * logdensityof(B_temp, v_temp, a₀, τ_temp, σ, x[i].rt, x[i].choice)
        end
        
        # Return negative since optimizers typically minimize
        return -ll
    end
    
    # Set up optimization
    initial_params = [B, v, τ]
    
    # Add bounds - a₀_frac must be between -1 and 1
    lower_bounds = [0.001, -Inf, 1e-3]
    upper_bounds = [50.0, 10.0, 5.0]
    
    # Optimize using L-BFGS-B to respect the bounds
    result = optimize(neg_log_likelihood, lower_bounds, upper_bounds, initial_params, Fminbox(LBFGS()), autodiff=:forward)
    
    # Extract the optimized parameters
    optimal_params = Optim.minimizer(result)
    
    # Update the model with new parameter estimates
    model.B = optimal_params[1]
    model.v = optimal_params[2]
    # model.a₀ = optimal_params[3]
    model.τ = optimal_params[3]
    
    return model
end

function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{Any}, w::AbstractVector{<:Real})
    x_ddm = DDMResult[]
    w_ddm = Float64[]
    for (xi, wi) in zip(x, w)
        if xi isa DDMResult
            push!(x_ddm, xi)
            push!(w_ddm, wi)
        end
    end
    return fit!(model, x_ddm, w_ddm)
end

function StatsAPI.fit!(model::UniformEmission, x::Vector{UniformResult}, w::AbstractVector{<:Real})
    # Compute weighted min and max RTs
    rts = getfield.(x, :rt)
    weighted_mean = sum(w .* rts) / sum(w)
    weighted_std = sqrt(sum(w .* (rts .- weighted_mean).^2) / sum(w))

    # Update model bounds (safely)
    model.a = minimum(rts) - 0.05 * weighted_std
    model.b = maximum(rts) + 0.05 * weighted_std
    return model
end

function StatsAPI.fit!(model::UniformEmission, x::Vector{Any}, w::AbstractVector{<:Real})
    x_filtered = filter(xi -> xi isa UniformResult, x)
    x_cast = UniformResult[xi for xi in x_filtered]
    return fit!(model, x_cast, w[1:length(x_cast)])
end
    