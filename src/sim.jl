import Base: step
using ControlSystemsBase: lqr, c2d, ss, StateSpace, Discrete
using LinearAlgebra: I, norm

mutable struct Environment
  sys::StateSpace{<:Discrete}
  K::Matrix{Float64}
  state::Vector{Float64}
  ideal_state::Vector{Float64}
end

"""
    Environment(sys)

Constructor for `Environment` with the given system `sys`.
"""
function Environment(sys::StateSpace{<:Discrete})
  return Environment(sys, lqr(Discrete, sys.A, sys.B, I, I), make_x0(sys), make_x0(sys))
end

"""
    step!(env, action)

Simulate the plant's dynamics for one step, according to the action `action` (true for hitting the deadline, false for missing it).
"""
function step!(env::Environment, action::Bool)
  env.state = env.sys.A * env.state - env.sys.B * env.K * env.state * action
  env.ideal_state = env.sys.A * env.ideal_state - env.sys.B * env.K * env.ideal_state
  return env.state, d_ideal(env)
end

"""
    sim!(env, π, H)

Simulate the environment for `H` steps using policy `π`.
Returns vectors containing the actions, states, distances, and ideal states, respectively.
"""
function sim!(env::Environment, π::Function, dₜ::Float64, H::Integer)
  actions = Vector{Bool}(undef, H)
  states = Vector{Vector{Float64}}(undef, H + 1)
  utilizations = Vector{Float64}(undef, H + 1)
  distances = Vector{Float64}(undef, H + 1)
  ideal_states = Vector{Vector{Float64}}(undef, H + 1)

  states[1] = env.state
  utilizations[1] = 1.0
  distances[1] = d_ideal(env)
  ideal_states[1] = env.ideal_state

  for t in 1:H
    actions[t] = π(actions[1:t-1], states[1:t], distances[1:t], dₜ)
    utilizations[t+1] = sum(actions[1:t]) / length(actions[1:t])
    states[t+1], distances[t+1] = step!(env, actions[t])
    ideal_states[t+1] = env.ideal_state
  end

  return actions, stack(states), distances, utilizations, stack(ideal_states)
end

"""
    d_ideal(env)

Calculate the distance from ideal behavior for the given environment at the current step.
"""
function d_ideal(env::Environment)
  return norm(env.state - env.ideal_state)
end

"""
    make_x0(sys)

A convenience function that returns an initial state with the correct dimensions for the plant.
"""
function make_x0(sys::StateSpace)
  return repeat([1.0], sys.nx)
end
