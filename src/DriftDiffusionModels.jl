module DriftDiffusionModels

using LinearAlgebra
using Random
using StatsAPI
using Distributions
using Optim
using ForwardDiff
using UnPack
using DensityInterface

# Import the fit! function specifically--its being weird about fit!
import StatsAPI: fit!

include("DDM.jl")

export DriftDiffusionModel, DDMResult, rand, logdensityof, fit!

end