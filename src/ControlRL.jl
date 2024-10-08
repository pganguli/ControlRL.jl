module ControlRL

export Environment, step!, state, d_ideal, sim!
include("sim.jl")

export benchmarks, c2d
include("models.jl")

end
