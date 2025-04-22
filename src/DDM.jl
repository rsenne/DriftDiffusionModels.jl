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