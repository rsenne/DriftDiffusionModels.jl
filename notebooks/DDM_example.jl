### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ f7166b82-1fc7-11f0-187a-01fb6c95d3b8
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 23318261-4fbd-4518-8bde-100cdde3a171
begin
	using DriftDiffusionModels
	using Plots
end

# ╔═╡ 4c136540-5865-4dc1-81db-c009776a252f
md"Sample a DDM"

# ╔═╡ 43fa3923-1946-4781-a86c-1aabd3ee3bb7
begin
	# create a DDM we will sample RT and choices from
	B = 10
	v = 1.25
	α₀ = 1.2
	σ = 1.0 # default value always

	ddm = DriftDiffusionModel(B, v, α₀, σ)

	# now we sample
	data = rand(ddm, 10)
end
	

# ╔═╡ 732377de-54f5-46f6-91b4-8cb06fcdd85b
methods(StatsAPI.fit!)

# ╔═╡ b9e64f50-7659-45ce-b933-22c9d8655187
md"""Fit the DDM"""

# ╔═╡ b6f6d452-d9e4-483e-a732-b2186558206b
begin
	# using the data from generated above, fit the DDM. Here we assume a single state model so no weights necessary.
	naive_ddm = DriftDiffusionModel()

	DriftDiffusionModels.StatsAPI.fit!(naive_ddm, data)
end
	

# ╔═╡ Cell order:
# ╠═f7166b82-1fc7-11f0-187a-01fb6c95d3b8
# ╠═23318261-4fbd-4518-8bde-100cdde3a171
# ╟─4c136540-5865-4dc1-81db-c009776a252f
# ╠═43fa3923-1946-4781-a86c-1aabd3ee3bb7
# ╠═732377de-54f5-46f6-91b4-8cb06fcdd85b
# ╟─b9e64f50-7659-45ce-b933-22c9d8655187
# ╠═b6f6d452-d9e4-483e-a732-b2186558206b
