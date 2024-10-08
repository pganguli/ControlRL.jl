### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 04f3e318-6489-11ef-1ccc-89d0ed2d1029
begin
  using Pkg
  Pkg.activate(".")
  Pkg.instantiate()

  using Revise
  using LinearAlgebra
  using PlutoUI
  using Plots
  plotlyjs()

  push!(LOAD_PATH, "$(@__DIR__)/../src")
  using ControlRL

  TableOfContents()
end

# ╔═╡ abb82687-12f0-4754-b3dd-49c03f3c9ad1
md"""
## Import Packages
"""

# ╔═╡ 5ff9a070-12a3-45fd-a1b4-b619998e468a
md"""
## Define policy π
"""

# ╔═╡ e8996f64-b030-4624-9e34-d452e5d2d5db
begin
  const safety_margin = 10
  const H = 1200
  const Tₛ = 0.01
  const γ = 0.99
  const α = 0.5
  @bind dₜ_optimal Slider(0:0.001:1, show_value=true, default=0.014)
end

# ╔═╡ fac6522b-c189-4b7c-be35-11d038a854bc
π(actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, dₜ::Float64) = (length(actions) == 0) || (distances[end] > dₜ * safety_margin)

# ╔═╡ 4798f6c2-ce68-4818-8a84-75323bdd8306
dₜ_optimal

# ╔═╡ e705f0df-bdcc-4673-976f-5d096c6a2abc
md"""
## Helper Functions
"""

# ╔═╡ 5402045e-83ef-4b90-9f14-1796fdadacc0
function plotH(states::Matrix{Float64})
  # Transposing `states` because plot uses the first dimension as indices.
  plot(0:size(states, 2)-1, states')
end

# ╔═╡ c34375b4-70ab-4468-9186-c43948f22a7a
plotH(v::Vector{Float64}) = plotH(reshape(v, 1, :))

# ╔═╡ 843ed7d1-692c-4579-911d-32534a167280
function reward(distances::Vector{Float64}, utilizations::Vector{Float64}, α::Float64, γ::Float64, H::Int64)
	discounted_rewards = [-1 * (γ^i) * (α * distances[i] + (1 - α) * utilizations[i]) for i = 1:H]
    return discounted_rewards, sum(discounted_rewards)
end

# ╔═╡ 2d1d69f5-aed9-4f99-a63a-2f94347c7e15
begin
  sys_discrete = c2d(benchmarks[:F1], Tₛ)
  env = Environment(sys_discrete)

 #  dₜ_optimal = 0
 #  max_total_reward = -999
 #  for dₜ = 0:0.001:1
 #    actions, states, distances, utilizations, ideal_states = sim!(env, π, dₜ, H)
	# discounted_rewards, total_reward = reward(distances, utilizations, α, γ, H)

	# if max_total_reward < total_reward
	#   max_total_reward = total_reward
	#   dₜ_optimal = dₜ
	# end
 #  end

  actions, states, distances, utilizations, ideal_states = sim!(env, π, dₜ_optimal, H)
  discounted_rewards, total_reward = reward(distances, utilizations, α, γ, H)
end

# ╔═╡ d58e85a8-513f-45e2-906c-41d3bdea7bb5
sum(actions)

# ╔═╡ 19634c38-ec2c-45a0-862a-3f92beecb5a1
plotH(ideal_states)

# ╔═╡ 66fbe369-33e7-409f-ae66-29c1984518b3
plotH(states)

# ╔═╡ 7c54f79a-6bb4-4190-ab13-bca71bb868a8
plotH(distances)

# ╔═╡ 022d0123-0051-4815-8dda-0611edfceba6
plotH(utilizations)

# ╔═╡ 2d1ae051-9ea6-4525-9a53-769c9849915f
plotH(discounted_rewards)

# ╔═╡ Cell order:
# ╟─abb82687-12f0-4754-b3dd-49c03f3c9ad1
# ╠═04f3e318-6489-11ef-1ccc-89d0ed2d1029
# ╟─5ff9a070-12a3-45fd-a1b4-b619998e468a
# ╠═fac6522b-c189-4b7c-be35-11d038a854bc
# ╠═e8996f64-b030-4624-9e34-d452e5d2d5db
# ╠═2d1d69f5-aed9-4f99-a63a-2f94347c7e15
# ╠═4798f6c2-ce68-4818-8a84-75323bdd8306
# ╠═d58e85a8-513f-45e2-906c-41d3bdea7bb5
# ╠═19634c38-ec2c-45a0-862a-3f92beecb5a1
# ╠═66fbe369-33e7-409f-ae66-29c1984518b3
# ╠═7c54f79a-6bb4-4190-ab13-bca71bb868a8
# ╠═022d0123-0051-4815-8dda-0611edfceba6
# ╠═2d1ae051-9ea6-4525-9a53-769c9849915f
# ╟─e705f0df-bdcc-4673-976f-5d096c6a2abc
# ╠═5402045e-83ef-4b90-9f14-1796fdadacc0
# ╠═c34375b4-70ab-4468-9186-c43948f22a7a
# ╠═843ed7d1-692c-4579-911d-32534a167280
