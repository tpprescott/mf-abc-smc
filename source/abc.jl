export simulate, distance

function simulate(
    m::AbstractModel{S,L},
    theta::AbstractParameters{S},
    out::ModelOutput{S,LL},
)::ModelOutput{S,L} where {S} where {L} where {LL}
    return simulate(m, theta, out.u)
end
function simulate(
    m::AbstractModel{S,L},
    theta::AbstractParameters{S},
    u::AbstractNoise{S,LL} = IndependentSimulation{S,L}(),
)::ModelOutput{S,L} where {S} where {L} where {LL}
    return ModelOutput{S,L}(m(theta, u)...)
end

function distance(
    x1::ModelOutput{S,L},
    x2::ModelOutput{S,L},
) where {S} where {L}
    return distance(x1.y, x2.y)
end
function distance(
    x::ModelOutput{S,L},
    y::AbstractSummaryStat{S,L},
) where {S} where {L}
    return distance(x.y, y)
end
using LinearAlgebra
function distance(
    x::AbstractSummaryStat{S,L},
    y::AbstractSummaryStat{S,L},
) where {S} where {L}
    return norm(x-y)
end

# export ABCStep, ABCStepOut

struct ABCStep{S,L}
    m::AbstractModel{S,L}
    y_obs::AbstractSummaryStat{S,L}
    epsilon::Float64
end

struct ABCStepOut{S,L}
    sim::ModelOutput{S,L}
    d::Float64
    w::Float64

    function ABCStepOut(
        abc::ABCStep{S,L},
        theta::AbstractParameters{S},
        coupling = IndependentSimulation{S,L}(),
    ) where {S} where {L}
        sim::ModelOutput{S,L} = simulate(abc.m, theta, coupling)
        d::Float64 = distance(sim, abc.y_obs)
        (d <= abc.epsilon) ? (w = 1.0) : (w = 0.0)
        return new{S,L}(sim, d, w)
    end
end

# Could define an ABC problem here
