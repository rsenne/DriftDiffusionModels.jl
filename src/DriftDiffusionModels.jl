module DriftDiffusionModels

using LinearAlgebra
using Random
using StatsAPI
using Distributions
using Optim
using ForwardDiff
using UnPack
using Random: Random, AbstractRNG, default_rng
using DensityInterface
using HiddenMarkovModels
using FillArrays
using SpecialFunctions
using ArgCheck
using Base.Threads: @threads
import DensityInterface: logdensityof

# Import the fit! function specifically--its being weird about fit!
import StatsAPI: fit!

include("DDM.jl")
include("HMMDDM.jl")
include("Utilities.jl")

export EmissionModel, ScaledBetaEmission, ExponentialEmission, DriftDiffusionModel, DDMResult, AbstractResult 
export rand, logdensityof, fit!, crossvalidate, PriorHMM, simulateDDM, wfpt, randomDDM

end