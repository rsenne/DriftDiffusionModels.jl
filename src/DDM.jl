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

mutable struct ExponentialEmission <: EmissionModel
    λ::Float64 
end

function ExponentialEmission(;
    λ::Float64   
)
    return ExponentialEmission(λ)
end

mutable struct ScaledBetaEmission <: EmissionModel 
    α::Float64
    β::Float64
    a::Float64
    b::Float64
end 

function ScaledBetaEmission(; 
    α::Float64, 
    β::Float64, 
    a::Float64, 
    b::Float64, 
)
    return ScaledBetaEmission(α, β, a, b)
end 
abstract type AbstractResult end

"""
    DDMResult

A tuple of RT and choice. The first element is the RT, the second is the choice.
"""
struct DDMResult <: AbstractResult
    rt::Float64
    choice::Int
end

function DDMResult(;rt::Float64, choice::Int)
    return DDMResult(rt, choice)
end

Base.eltype(::DriftDiffusionModel) = DDMResult

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
        return 1e-12
    end
    
    # Use normalized time and relative start point
    tt = (t - τ) / (B^2)
    tt = max(tt, 1e-12)
    
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
    Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)

Generate a single trial of the drift diffusion model using the Euler-Maruyama method--needed for HiddenMarkovModels.jl
"""
function Random.rand(rng::AbstractRNG, model::DriftDiffusionModel)
    # Generate a single trial of the drift diffusion model
    return simulateDDM(model, 1e-6, rng)
end

"""
Generate a single random exponential 
"""
function Random.rand(rng::AbstractRNG, model::ExponentialEmission)
    rt = rand(rng, Exponential(model.λ))
    choice = rand(rng, (-1, 1)) 
    return DDMResult(rt, choice)
end

"""
Scaled beta rand function 
"""
function Random.rand(rng::AbstractRNG, model::ScaledBetaEmission)
    z = rand(rng, Beta(model.α, model.β))
    
    rt = model.a + z * (model.b - model.a)
    
    choice = rand(rng, (-1, 1))
    
    return DDMResult(rt, choice)
end

"""
Edited rand function from HiddenMarkovModels.jl because vector{Any}
"""
function Random.rand(rng::AbstractRNG, hmm::AbstractHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, HiddenMarkovModels.LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    @views for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t + 1])
        state_seq[t + 1] = rand(
            rng, HiddenMarkovModels.LightCategorical(trans[state_seq[t], :], dummy_log_probas)
        )
    end

    dists1 = HiddenMarkovModels.obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1])
    obs_seq = Vector{AbstractResult}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = HiddenMarkovModels.obs_distributions(hmm, control_seq[t])
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

function Random.rand(hmm::AbstractHMM, control_seq::AbstractVector)
    return rand(default_rng(), hmm, control_seq)
end

function Random.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    return rand(rng, hmm, Fill(nothing, T))
end

function Random.rand(hmm::AbstractHMM, T::Integer)
    return rand(hmm, Fill(nothing, T))
end


DensityInterface.DensityKind(::DriftDiffusionModel) = HasDensity()
DensityInterface.DensityKind(::ExponentialEmission) = HasDensity()
DensityInterface.DensityKind(::ScaledBetaEmission) = HasDensity()
"""
    DensityInterface.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a drift diffusion model given a DDMResult.
"""
function DensityInterface.logdensityof(model::DriftDiffusionModel, x::AbstractResult)
    @unpack B, v, a₀, τ, σ = model
    @unpack rt, choice = x

    return logdensityof(B, v, a₀, τ, σ, rt, choice)
end

function logdensityof(
    B::TB, v::TV, a₀::TA, τ::TT, σ::TS, rt::Float64, choice::Int; 
) where {TB<:Real, TV<:Real, TA<:Real, TT<:Real, TS<:Real}
    if rt <= 0
        return -1e12
    end

    T = promote_type(TB, TV, TA, TT, TS)
    B, v, a₀, τ, σ = T(B), T(v), T(a₀), T(τ), T(σ)

    # determine which version of the wpft to use: upper or lower boundary
    v, w = choice == 1 ? -v : v, choice == 1 ? 1 - a₀ : a₀


    # calculate the Wiener first passage time density
    density = wfpt(rt, v, B, w, τ)
    logdens = log(density)

    # check if density is Inf (i.e., log(0)) and return a very large value if so
    return isfinite(logdens) ? logdens : -1e16
end

"""
    DensityInterface.logdensityof(model::ExponentialEmission, x::AbstractResult)

Calculate log density of Exponential Result
"""
function DensityInterface.logdensityof(model::ExponentialEmission, x::AbstractResult)
    rt = x.rt
    λ = model.λ
    if rt < 0 || λ <= 0
        return -1e6  
    else
        return log(λ) - λ * rt
    end
end

"""
log density of Scaled Beta result 
"""
function DensityInterface.logdensityof(model::ScaledBetaEmission, x::AbstractResult)
    rt = x.rt
    if rt < model.a || rt > model.b
        return -1e6
    end
    z = (rt - model.a) / (model.b - model.a)
    return logpdf(Beta(model.α, model.β), z) - log(model.b - model.a)
end


"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{<:AbstractResult}, w::AbstractVector{<:Real}=ones(length(x)))
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

"""
Exponential Fit function 
"""
function StatsAPI.fit!(
    model::ExponentialEmission, 
    x::Vector{<:AbstractResult}, 
    w::AbstractVector{<:Real}=ones(length(x)), 
)
    rts = getfield.(x, :rt)

    # weighted MLE of Exponential from StackExchange 
    numerator = sum(w)
    denominator = sum(w .* rts)

    if denominator <= 0
        model.λ = 1e-6 
    else
        model.λ = numerator / denominator
    end

    return model
end

"""
Scaled Beta fit function 
"""
function StatsAPI.fit!(
    model::ScaledBetaEmission, 
    x::Vector{<:AbstractResult}, 
    w::AbstractVector{<:Real}=ones(length(x)), 
)
    rts = [r.rt for r in x]
    rts_scaled = (rts .- model.a) ./ (model.b - model.a)
    total_w = sum(w)
    μ̂ = sum(w .* rts_scaled) / total_w
    σ²̂ = sum(w .* (rts_scaled .- μ̂).^2) / total_w

    common = μ̂ * (1 - μ̂) / σ²̂ - 1
    α̂ = μ̂ * common
    β̂ = (1 - μ̂) * common

    model.α = α̂
    model.β = β̂
    return model
end 