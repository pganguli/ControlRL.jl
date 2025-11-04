### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
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
  #plotlyjs()

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

# ╔═╡ d7445d32-ea62-4eac-b90d-a93256ee37a0
rand_bool(p) = rand() < p

# ╔═╡ 384689c0-9d04-4471-adf5-86d06789a6ed
sigmoid(z::Float64) = 1 / (1 + ℯ^(-z))

# ╔═╡ 68d8bc54-23c7-47a4-bccb-7064c06d8193
p_hit_1(d::Float64, p::Float64, β::Float64, κ::Float64) = sigmoid(κ * ((d/p) - β))

# ╔═╡ ea3727f1-a693-4d35-8bee-7cdf31d20c21
p_hit_3(d::Float64, p::Float64, β::Float64, λ::Float64) = 1 − ℯ^(−λ * max(d − β*p, 0))

# ╔═╡ fac6522b-c189-4b7c-be35-11d038a854bc
π(
	actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, β::Float64, p::Float64 
) = (length(actions) == 0) || (distances[end] > β * p)

# ╔═╡ e8996f64-b030-4624-9e34-d452e5d2d5db
begin
  const safety_margin = 1.5
  const H = 800
  const Tₛ = 0.02
  const γ = 0.99
  const α = 0.35
end

# ╔═╡ 0fc1b2e2-3f40-4e14-bc24-802c9ffdb022
π_1(
	actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, β::Float64, κ::Float64
) = (length(actions) == 0) || rand_bool(p_hit_1(distances[end], safety_margin, β, κ))

# ╔═╡ 0607c3a0-dc13-4094-98fa-2573c8814c6d
π_3(
	actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, β::Float64, λ::Float64
) = (length(actions) == 0) || rand_bool(p_hit_3(distances[end], safety_margin, β, λ))

# ╔═╡ 26a352c7-abfb-40d7-8964-b65fa7688d9d
@bind β_optimal Slider(0:0.001:1, show_value=true, default=0.243)

# ╔═╡ 3312da25-f2dd-49bf-9b57-701cf6219dc0
@bind κ_optimal Slider(0:0.001:1, show_value=true, default=0.498)

# ╔═╡ 81cf68ef-f215-4afd-9e1b-ede6a4b29188
@bind λ_optimal Slider(0:0.001:1, show_value=true, default=0.802)

# ╔═╡ e705f0df-bdcc-4673-976f-5d096c6a2abc
md"""
## Helper Functions
"""

# ╔═╡ 5402045e-83ef-4b90-9f14-1796fdadacc0
function plotH(states::Matrix{Float64}; xlabel::String="Iteration", ylabel::String="", save::Bool=false, show_legend::Bool=false)
    p = plot(
        0:size(states, 2)-1, states';
        xlabel=xlabel,
        ylabel=ylabel,
        xguidefontsize=22,
        yguidefontsize=22,
        legendfontsize=12,
        legend=show_legend ? :bottomright : false,
        xtickfontsize=14,
        ytickfontsize=14
    )

    if save
        filename = ylabel == "" ? "figure.pdf" : string(ylabel, ".pdf")
        savefig(p, filename)
    end

    return p
end

# ╔═╡ c34375b4-70ab-4468-9186-c43948f22a7a
plotH(
	v::Vector{Float64}; xlabel::String="Iteration", ylabel::String="", save::Bool=false, show_legend::Bool=false
) = plotH(reshape(v, 1, :); xlabel=xlabel, ylabel=ylabel, save=save, show_legend=show_legend)

# ╔═╡ 843ed7d1-692c-4579-911d-32534a167280
function reward(distances::Vector{Float64}, utilizations::Vector{Float64}, α::Float64, γ::Float64, H::Int64)
	discounted_rewards = [-1 * (γ^i) * (α * distances[i] + (1 - α) * utilizations[i]) for i = 1:H]
    return discounted_rewards, sum(discounted_rewards)
end

# ╔═╡ 2d1d69f5-aed9-4f99-a63a-2f94347c7e15
begin
  sys_discrete = c2d(benchmarks[:CC], Tₛ)
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

  #actions, states, distances, utilizations, ideal_states = sim!(env, π, β_optimal, safety_margin, H)
  actions, states, distances, utilizations, ideal_states = sim!(env, π_1, β_optimal, κ_optimal, H)
  #actions, states, distances, utilizations, ideal_states = sim!(env, π_3, β_optimal, λ_optimal, H)
  discounted_rewards, total_reward = reward(distances, utilizations, α, γ, H)
end

# ╔═╡ d58e85a8-513f-45e2-906c-41d3bdea7bb5
sum(actions)

# ╔═╡ 036b19be-0900-43cf-b8b9-e746ed1a8c0d
actions

# ╔═╡ 19634c38-ec2c-45a0-862a-3f92beecb5a1
plotH(ideal_states; ylabel="Ideal Dynamics", show_legend=true, save=true)

# ╔═╡ 66fbe369-33e7-409f-ae66-29c1984518b3
plotH(states; ylabel="System Dynamics", show_legend=true, save=true)

# ╔═╡ 7c54f79a-6bb4-4190-ab13-bca71bb868a8
plotH(distances; ylabel="Deviation", save=true)

# ╔═╡ 022d0123-0051-4815-8dda-0611edfceba6
plotH(utilizations; ylabel="Utilization", save=true)

# ╔═╡ 2d1ae051-9ea6-4525-9a53-769c9849915f
plotH(discounted_rewards; ylabel="Cumulative Rewards", save=true)

# ╔═╡ 6202326a-ec46-42af-899d-5c2c1fb33d4f
maximum(distances[400:end])

# ╔═╡ Cell order:
# ╟─abb82687-12f0-4754-b3dd-49c03f3c9ad1
# ╠═04f3e318-6489-11ef-1ccc-89d0ed2d1029
# ╟─5ff9a070-12a3-45fd-a1b4-b619998e468a
# ╠═d7445d32-ea62-4eac-b90d-a93256ee37a0
# ╠═384689c0-9d04-4471-adf5-86d06789a6ed
# ╠═68d8bc54-23c7-47a4-bccb-7064c06d8193
# ╠═ea3727f1-a693-4d35-8bee-7cdf31d20c21
# ╠═fac6522b-c189-4b7c-be35-11d038a854bc
# ╠═0fc1b2e2-3f40-4e14-bc24-802c9ffdb022
# ╠═0607c3a0-dc13-4094-98fa-2573c8814c6d
# ╠═e8996f64-b030-4624-9e34-d452e5d2d5db
# ╠═26a352c7-abfb-40d7-8964-b65fa7688d9d
# ╠═3312da25-f2dd-49bf-9b57-701cf6219dc0
# ╠═81cf68ef-f215-4afd-9e1b-ede6a4b29188
# ╠═2d1d69f5-aed9-4f99-a63a-2f94347c7e15
# ╠═d58e85a8-513f-45e2-906c-41d3bdea7bb5
# ╠═036b19be-0900-43cf-b8b9-e746ed1a8c0d
# ╠═19634c38-ec2c-45a0-862a-3f92beecb5a1
# ╠═66fbe369-33e7-409f-ae66-29c1984518b3
# ╠═7c54f79a-6bb4-4190-ab13-bca71bb868a8
# ╠═022d0123-0051-4815-8dda-0611edfceba6
# ╠═2d1ae051-9ea6-4525-9a53-769c9849915f
# ╠═6202326a-ec46-42af-899d-5c2c1fb33d4f
# ╟─e705f0df-bdcc-4673-976f-5d096c6a2abc
# ╠═5402045e-83ef-4b90-9f14-1796fdadacc0
# ╠═c34375b4-70ab-4468-9186-c43948f22a7a
# ╠═843ed7d1-692c-4579-911d-32534a167280
