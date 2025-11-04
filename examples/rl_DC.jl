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

# ╔═╡ 62e96da2-5456-47bf-a132-9c68b4d1ca61
rand_bool(p) = rand() < p

# ╔═╡ 57449c23-77a7-4fb4-b387-42da4832c735
sigmoid(z::Float64) = 1 / (1 + ℯ^(-z))

# ╔═╡ 620a8a2c-3df1-4f01-8655-c99bc87ff2c7
p_hit_1(d::Float64, p::Float64, β::Float64, κ::Float64) = sigmoid(κ * ((d/p) - β))

# ╔═╡ 148595ff-cbc5-4933-b3e5-c92baa7a3809
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

# ╔═╡ 8975349b-c18e-4be6-8020-8d2f5a68ea71
π_1(
	actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, β::Float64, κ::Float64
) = (length(actions) == 0) || rand_bool(p_hit_1(distances[end], safety_margin, β, κ))

# ╔═╡ 006e6b21-8fe9-4421-9287-4408731b22ca
π_3(
	actions::Vector{Bool}, states::Vector{Vector{Float64}}, distances::Vector{Float64}, β::Float64, λ::Float64
) = (length(actions) == 0) || rand_bool(p_hit_3(distances[end], safety_margin, β, λ))

# ╔═╡ 64850b2f-fe93-47f3-8f7e-0de17239dfea
@bind β_optimal Slider(0:0.001:1, show_value=true, default=0.052)

# ╔═╡ 7753ba07-dfcb-4f9d-aa22-131fee96e456
@bind κ_optimal Slider(0:0.001:1, show_value=true, default=1.0)

# ╔═╡ 508a4c4a-a6e3-4c5b-8064-2892decd93da
@bind λ_optimal Slider(0:0.001:1, show_value=true, default=1.0)

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
  sys_discrete = c2d(benchmarks[:DC], Tₛ)
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

# ╔═╡ 10033ad2-b054-4444-bf3e-25d0ac0810ec
maximum(distances[1:end])

# ╔═╡ Cell order:
# ╟─abb82687-12f0-4754-b3dd-49c03f3c9ad1
# ╠═04f3e318-6489-11ef-1ccc-89d0ed2d1029
# ╟─5ff9a070-12a3-45fd-a1b4-b619998e468a
# ╠═62e96da2-5456-47bf-a132-9c68b4d1ca61
# ╠═57449c23-77a7-4fb4-b387-42da4832c735
# ╠═620a8a2c-3df1-4f01-8655-c99bc87ff2c7
# ╠═148595ff-cbc5-4933-b3e5-c92baa7a3809
# ╠═fac6522b-c189-4b7c-be35-11d038a854bc
# ╠═8975349b-c18e-4be6-8020-8d2f5a68ea71
# ╠═006e6b21-8fe9-4421-9287-4408731b22ca
# ╠═e8996f64-b030-4624-9e34-d452e5d2d5db
# ╠═64850b2f-fe93-47f3-8f7e-0de17239dfea
# ╠═7753ba07-dfcb-4f9d-aa22-131fee96e456
# ╠═508a4c4a-a6e3-4c5b-8064-2892decd93da
# ╠═2d1d69f5-aed9-4f99-a63a-2f94347c7e15
# ╠═d58e85a8-513f-45e2-906c-41d3bdea7bb5
# ╠═036b19be-0900-43cf-b8b9-e746ed1a8c0d
# ╠═19634c38-ec2c-45a0-862a-3f92beecb5a1
# ╠═66fbe369-33e7-409f-ae66-29c1984518b3
# ╠═7c54f79a-6bb4-4190-ab13-bca71bb868a8
# ╠═022d0123-0051-4815-8dda-0611edfceba6
# ╠═2d1ae051-9ea6-4525-9a53-769c9849915f
# ╠═10033ad2-b054-4444-bf3e-25d0ac0810ec
# ╟─e705f0df-bdcc-4673-976f-5d096c6a2abc
# ╠═5402045e-83ef-4b90-9f14-1796fdadacc0
# ╠═c34375b4-70ab-4468-9186-c43948f22a7a
# ╠═843ed7d1-692c-4579-911d-32534a167280
