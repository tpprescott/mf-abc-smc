###### MULTIFIDELITY EXTENSION
export Particle, Cloud, MFABCProblem, LoFi, HiFi
using Distributed

# Only have two fidelities of interest for now
struct LoFi <: Label end
struct HiFi <: Label end

### Allow continuation probabilities
abstract type AbstractContinuationProbability{S<:System,L<:Label} end
function cont_prob(
    alpha::AbstractContinuationProbability{S,L},
    abc_out::ABCStepOut{S,L},
    theta::AbstractParameters{S},
)::Float64 where {S} where {L}
    return alpha(abc_out, theta)
end

# Allow for an "always continue" (and use as the default)
struct AlwaysContinue{S,L} <: AbstractContinuationProbability{S,L} end
function cont_prob(
    alpha::AlwaysContinue{S,L},
    y::ABCStepOut{S,L},
    theta::AbstractParameters{S},
)::Float64 where {S} where {L}
    return 1.0
end

# I will only deal with piecewise constant continuation probabilities
export Eta
struct Eta{S,L} <: AbstractContinuationProbability{S,L}
    eta1::Float64
    eta2::Float64
    function Eta{S,L}(x1::Real, x2::Real) where {S} where {L}
        if (0.0 < x1 <= 1.0) && (0.0 < x2 <= 1.0)
            new{S,L}(x1, x2)
        else
            error("Not valid continuation probabilities")
        end
    end
end
function (eta::Eta{S,L})(y::ABCStepOut{S,L}, theta::AbstractParameters{S})::Float64 where {S} where {L}
# Return eta1 if acceptance, or eta2 if rejection (independent of theta)
    return eta.eta2 + y.w * (eta.eta1 - eta.eta2)
end

struct MFABCStep{S<:System,L<:Label}
    abc::ABCStep{S,L}
    alpha::AbstractContinuationProbability{S,L}

    function MFABCStep(
        m::AbstractModel{S,L},
        y_obs::AbstractSummaryStat{S,L},
        epsilon::Float64,
        alpha::AbstractContinuationProbability{S,L} = AlwaysContinue{S,L}(),
    ) where {S} where {L}
        return new{S,L}(ABCStep(m, y_obs, epsilon), alpha)
    end
end

struct MFABCStepOut{S<:System,L<:Label}
    abc::ABCStepOut{S,L}
    alpha::Float64 # Continuation probability
    t::Float64 # Simulation time

    function MFABCStepOut(
        mfabc::MFABCStep{S,L},
        theta::AbstractParameters{S},
        coupling = IndependentSimulation{S,L}(),
    ) where {S} where {L}
        abc::ABCStepOut,
        t_tot::Float64,
        _,
        t_gc::Float64,
        _ = @timed ABCStepOut(mfabc.abc, theta, coupling)
        alpha::Float64 = cont_prob(mfabc.alpha, abc, theta)
        return new{S,L}(abc, alpha, t_tot - t_gc)
    end
end

mutable struct MFABCProblem{S,T}
    p::ParameterSampler{T}
    q::ParameterSampler{T}
    steps::NTuple{N,MFABCStep{S,L} where L} where {N}

    function MFABCProblem(
        p::ParameterSampler{T},
        q::ParameterSampler{T},
        steps::MFABCStep{S}...,
    ) where {T<:AbstractParameters{S}} where {S}
        return new{S,T}(p, q, steps)
    end
end

# Two step MFABC is the only type we're dealing with: LoFi followed by HiFi
function MFABCProblem(
    p::ParameterSampler{T},
    q::ParameterSampler{T},
    m::Tuple{AbstractModel{S,LoFi},AbstractModel{S,HiFi}},
    y::Tuple{AbstractSummaryStat{S,LoFi},AbstractSummaryStat{S,HiFi}},
    e::Tuple{Float64,Float64},
    α::AbstractContinuationProbability{S,LoFi} = AlwaysContinue{S,LoFi}(),
) where {S} where {T}
    return MFABCProblem(
        p,
        q,
        MFABCStep.(m, y, e, (α, AlwaysContinue{S,HiFi}()))...,
    )
end

struct Particle{S<:System,T<:AbstractParameters{S}}
    theta::T
    w::Float64 # Weight in sample
    p::Float64 # Prior Likelihood
    q::Float64 # Importance Likelihood
    sims::NTuple{N,MFABCStepOut{S,L} where L} where {N} # Simulations completed
end

function Particle(prob::MFABCProblem{
    S,
    T,
}) where {T<:AbstractParameters{S}} where {S}
    theta, q = generate(prob.q)
    p::Float64 = likelihood(theta, prob.p)

    sims = mf_simulation(theta, prob.steps...)
    w::Float64 = (p / q) * mf_weight(sims...)
    return Particle{S,T}(theta, w, p, q, sims)
end

function mf_simulation(
    theta::AbstractParameters{S},
    sim_1::MFABCStep{S,L};
    noise::AbstractNoise{S,LL} = IndependentSimulation{S,L}(),
)::Tuple{MFABCStepOut{S,L}} where {S} where {L} where {LL}
    return (MFABCStepOut(sim_1, theta, noise),)
end
function mf_simulation(
    theta::AbstractParameters{S},
    sim_1::MFABCStep{S,L},
    sim_2::Vararg{MFABCStep{S,LL} where LL,N};
    noise::AbstractNoise{S,LLL} = IndependentSimulation{S,L}(),
)::Tuple{Vararg{
    MFABCStepOut{S,LL} where LL,
    K,
} where K} where {N} where {S} where {L} where {LLL}

    (mfabc_out,) = mf_simulation(theta, sim_1; noise = noise)
    if rand() < mfabc_out.alpha
        return (
            mfabc_out,
            mf_simulation(theta, sim_2...; noise = mfabc_out.abc.sim.u)...,
        )
    else
        return (mfabc_out,)
    end
end

function mf_weight(sim_1::MFABCStepOut{S,L})::Float64 where {S} where {L}
    return sim_1.abc.w
end
function mf_weight(
    sim_1::MFABCStepOut{S,L},
    sim_2::Vararg{MFABCStepOut{S,LL} where LL,N},
)::Float64 where {N} where {S} where {L}
    w = mf_weight(sim_1)
    w += (mf_weight(sim_2...) - w) / sim_1.alpha
    return w
end

import Base.length
function length(p::Particle{S,T})::Int64 where {S} where {T}
    return length(p.sims)
end

struct Cloud{S,T}
# Cloud is an array of particles, and the problem that generated them
    particles::Vector{Particle{S,T}}
    abc::MFABCProblem{S,T}
#    q::ImportanceSampler # The resampling distribution (i.e. posterior) built out of those particles
#    qvec::Array{Float64, 1} # An array of the posterior likelihood of each particle's parameter value
end

function Cloud(prob::MFABCProblem{S,T}, counter::Function, stop_condition; batch_size::Int64=100) where {S} where {T}
# Counter is one of ESS, gettime, or length
    println("Creating cloud: epsilons = $(getepsilon(prob)) and eta = $(geteta(prob)).")
    println("Conditions are when $(counter) exceeds $(stop_condition).")
    prob_func(i::Int64) = Particle(prob)
    C = Cloud{S,T}(Array{Particle{S,T},1}([]), prob)
    while counter(C) < stop_condition
        print("$counter = $(counter(C)) from $(length(C)) simulations; \r")
        append!(C.particles, pmap(prob_func, 1:batch_size))
    end
    return C
end
