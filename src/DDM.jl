"""
    DriftDiffusionModel

Implement a simple drift diffusion model: da/dt = v + σdW.
"""
@with_kw mutable struct DriftDiffusionModel
    B::Float64=10, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.0, # Initial Accumulation
    σ::Float64=1.0 # Noise--set to 1.0 be default for identifiability
end

function DriftDiffusionModel(
    B::Float64=10.0, #Bound Height
    v::Float64=1.0, # Drift Rate
    a₀::Float64=0.0, # Initial Accumulation
    σ::Float64=1.0 # Noise--set to 1.0 be default for identifiability
) 
    return DriftDiffusionModel(B=B, v=v, a₀=a₀, σ=σ)
end

"""
    DDMResult

A tuple of RT and choice. The first element is the RT, the second is the choice.
"""
struct DDMResult
    rt::Float64 # RT
    choice::Int # Choice (-1 or 1)
end

function DDMResult(rt::Float64, choice::Int)
    return DDMResult(rt=rt, choice=choice)
end

"""
    Random.rand(model::DriftDiffusionModel)

Generates RT and choices from a drift diffusion model. 

"""
function Random.rand(model::DriftDiffusionModel)

    @unpack B, v, a₀, σ = model

    # Calculate of crossing the upper bound 

end

"""
    Distributions.logdensityof(model::DriftDiffusionModel, x::DDMResult)

Calculate the loglikelihood of a DDM result i.e., the tuple of RT and choice.
"""
function Distributions.logdensityof(model::DriftDiffusionModel, x::DDMResult)

    @unpack B, v, a₀, σ = model
    @unpack rt, choice = x

    # Calculate the loglikelihood of the DDM result
end

"""
    StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))

Perform parameter estimation of a drift diffusion model using MLE given a vector of DDM observtions. Takes an optional weights vector to support for use in an HMM.
"""
function StatsAPI.fit!(model::DriftDiffusionModel, x::Vector{DDMResult}, w::Vector{Float64}=ones(length(x)))
    
        @unpack B, v, a₀, σ = model

        
    end

    