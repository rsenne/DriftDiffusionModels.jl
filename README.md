# DriftDiffusionModels.jl

**DriftDiffusionModels.jl** is a Julia package for simulating, fitting, and analyzing Drift Diffusion Models (DDMs), with support for multi-state Hidden Markov Models (HMMs) whose emission distributions are governed by DDMs. The package provides tools for simulation, inference, and model selection via cross-validation.

> **Note**: This package is **not intended as a production-ready DDM toolkit**. For a more complete and sophisticated implementation of sequential sampling models including advanced DDM variants, we recommend using [`SequentialSamplingModels.jl`](https://github.com/itsdfish/SequentialSamplingModels.jl).

---

## Features

* Simulation of DDM trajectories using the Euler–Maruyama method
* Wiener First Passage Time (WFPT) density computation (Navarro & Fuss, 2009)
* Log-likelihood and parameter estimation for DDMs via MLE
* Hidden Markov Models with DDM emissions and Dirichlet priors
* Baum–Welch (EM) training with MAP updates
* Cross-validation for model selection and evaluation

---

## Installation

Clone the repository and include the module in your Julia environment:

```julia
include("DriftDiffusionModels.jl")
using .DriftDiffusionModels
```

---

## Module Overview

### `DriftDiffusionModel`

The core structure representing a DDM:

```julia
DriftDiffusionModel(B, v, a₀, τ, σ)
```

* `B`: Boundary separation
* `v`: Drift rate
* `a₀`: Initial fraction of the boundary
* `τ`: Non-decision time
* `σ`: Noise (fixed to 1.0 for identifiability)

### `DDMResult`

Result of a single DDM simulation:

```julia
DDMResult(rt, choice)
```

* `rt`: Response time
* `choice`: Decision outcome (1 or -1)

---

## ⚙Key Functions

### Simulation

```julia
simulateDDM(model::DriftDiffusionModel, dt::Float64=1e-5)
simulateDDM(model::DriftDiffusionModel, n::Int, dt::Float64=1e-5)
```

Simulates one or multiple trials of the DDM using Euler–Maruyama integration.

### Likelihood

```julia
wfpt(t, v, B, w, τ)
logdensityof(model::DriftDiffusionModel, result::DDMResult)
```

Computes the WFPT density and log-likelihood for observed DDM results.

### Fitting

```julia
StatsAPI.fit!(model::DriftDiffusionModel, data::Vector{DDMResult}, weights=ones(length(data)))
```

Fits a DDM to data using Maximum Likelihood Estimation (MLE), supporting observation weights (useful for HMM training).

---

## Hidden Markov Models with DDM Emissions

### `PriorHMM`

A wrapper for Hidden Markov Models with Dirichlet priors on initial probabilities and transition matrices:

```julia
PriorHMM(init, trans, dists; α_trans, α_init)
```

Supports MAP updates via Baum–Welch.

### Training

```julia
baum_welch(hmm, data; seq_ends)
```

Trains an HMM-DDM model using EM.

---

## Model Comparison

### Log-Likelihood Ratio

```julia
calculate_ll_ratio(ll, ll₀, n)
```

Computes the per-observation log-likelihood ratio (in bits) between multi-state and single-state models.

### Cross-Validation

```julia
crossvalidate(data;
              n_folds=5,
              n_states=5,
              n_iter=5,
              rng=Random.GLOBAL_RNG)
```

Performs k-fold cross-validation across multiple HMM state counts and random restarts.

---

## Utilities

* `randomDDM()` — generate a random DDM
* Emission model fitting is integrated with `StatsAPI.fit!`
* Compatible with `DensityInterface` for probabilistic programming use cases

---

## References

* Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models.
* HiddenMarkovModels.jl — backend for HMM routines.

---

## File Structure

* `DriftDiffusionModels.jl` – Main module file
* `DDM.jl` – Drift Diffusion Model definitions and utilities
* `HMMDDM.jl` – HMM wrapper with DDM emissions and training
* `Utilities.jl` – Cross-validation and helper functions

---

## Contributions

Feel free to contribute pull requests or file issues to suggest features or report bugs!
