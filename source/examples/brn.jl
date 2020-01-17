module BiochemicalReactionNetwork

export BRN, BRNTraj, SummaryVector, GillespieModel, TauLeapModel
using ..MFABCSMC
using Distributions

#=
DEFINE A STOCHASTIC (DISCRETE STATE SPACE) BIOCHEMICAL REACTION NETWORK
=#
struct BRN{S<:System}
    nu::Matrix{Int64}
    x0::Vector{Int64}
    v::Function
end
struct BRNTraj{S<:System}
    t::Vector{Float64}
    x::Matrix{Int64}
end

# Define summary spaces
struct SummaryVector{S,L} <: AbstractSummaryStat{S,L}
    y::Array{Float64,1}
end
# The above is defined for each particular S, L by an external constructor
# function SummaryVector{S, L}(tx::BRNTraj{S})

# Defining distance in parameter space by subtraction and length
@eval Base.:-(a::SummaryVector{S,L}, b::SummaryVector{S,L}) where {S} where {L} =
    SummaryVector{S,L}(a.y - b.y)
import LinearAlgebra.norm
norm(a::SummaryVector)::Float64 = norm(a.y,2)

# Model types for BRNs
struct GillespieModel{S,L} <: AbstractModel{S,L}
    Sv::BRN{S}
    T::Float64
end
struct TauLeapModel{S,L} <: AbstractModel{S,L}
    Sv::BRN{S}
    T::Float64
    tau::Float64
end

### NOISE TYPES AND COUPLING TECHNIQUES

struct CoarsePP
    # Coarse-grained Poisson process
    d::Array{Float64,1} # Distance points
    r::Array{Int64,1}   # Total reactions by that distance
end
const PP = Array{Float64,1} # Full Poisson process

function PP(cpp::CoarsePP)::PP
    f(x, y, z)::Array{Float64,1} = z .+ y .* (sort!(rand(x)))
    return vcat(broadcast(f, diff(cpp.r), diff(cpp.d), cpp.d[1:end-1])...)
end

struct CoarsePPSet{S,L} <: AbstractNoise{S,L}
    cpp::Vector{CoarsePP}
end
struct PPSet{S,L} <: AbstractNoise{S,L}
    pp::Vector{PP}
end

function PPSet{S,L}(x::CoarsePPSet{S,M})::PPSet{
    S,
    L,
} where {S} where {L} where {M}
    return PPSet{S,L}(broadcast(PP, x.cpp))
end
function PPSet{S,L}(x::IndependentSimulation{S,M}, num_reactions::Integer)::PPSet{
    S,
    L,
} where {S} where {L} where {M}
    return PPSet{S,L}(fill(PP(), num_reactions))
end


########### SIMULATIONS

################ Helper functions
function intervaldistance!(
    delta_d::Array{Float64,1},
    v::Array{Float64,1},
    tau::Float64,
)
    delta_d[:] .= v .* tau
    return nothing
end

function intervalreactions!(delta_r::Array{Int64,1}, delta_d::Array{Float64,1})
    delta_r[:] .= rand.(Distributions.PoissonCountSampler.(delta_d))
    return nothing
end

isneg(x)::Bool = (x < zero(x))

function check!(
    delta_r::Array{Int64,1},
    delta_x::Array{Int64,1},
    S::Matrix{Int64},
    x::Array{Int64,1},
)::Int64
    i = 0
    delta_x[:] .= S * delta_r # Update delta_x
    while any(isneg, x + delta_x)
        # If any element will be made negative, half the time interval, and try again with only the reactions falling in the first half
        delta_r[:] .= rand.(Binomial.(delta_r, 0.5)) # Randomly half delta_r
        delta_x[:] .= S * delta_r # Equivalently halving delta_x
        i += 1 # Keep count of the halving index
    end
    return i
end

function findreaction(v::Array{Float64,1})::Int64
    r = rand()
    return 1 + count(r .> cumsum(v) ./ sum(v))
end

function _d_next!(
    d_next::Array{Float64,1},
    pp_out::Array{PP,1},
    pp::Array{PP,1},
    j::Int64,
)
    if length(pp[j]) > 0
        d_next[j] = popfirst!(pp[j])
    else
        d_next[j] += -log(rand())
        push!(pp_out[j], d_next[j])
    end
    return nothing
end


# REQUIRED: define independent simulations

### TAU LEAP MODEL
function (m::TauLeapModel{S,L})(
    theta::AbstractParameters{S},
    ind::IndependentSimulation{S,L},
)::Tuple{SummaryVector{S,L},CoarsePPSet{S,L}} where {S} where {L}

    N, M = size(m.Sv.nu)

    t = zero(Float64)
    d = zeros(Float64, M)
    r = zeros(Int64, M)
    v = zeros(Float64, M)
    x = copy(m.Sv.x0)

    delta_d = zeros(Float64, M)
    delta_r = zeros(Int64, M)
    delta_x = zeros(Int64, N)

    tvec = Array{Float64,1}()
    dvec = Array{Float64,1}()
    xvec = Array{Int64,1}()
    rvec = Array{Int64,1}()

    while t < m.T
        append!(tvec, t)
        append!(xvec, x)
        append!(dvec, d)
        append!(rvec, r)

        m.Sv.v(v, theta, x)
        intervaldistance!(delta_d, v, m.tau)
        intervalreactions!(delta_r, delta_d)
        i = check!(delta_r, delta_x, m.Sv.nu, x) # Note that this step updates delta_x

        t += m.tau * (0.5^i)
        d .+= delta_d * (0.5^i)
        x .+= delta_x
        r .+= delta_r
    end
    append!(tvec, t)
    append!(xvec, x)
    append!(dvec, d)
    append!(rvec, r)

    K = length(tvec)

    tx = BRNTraj{S}(tvec, reshape(xvec, N, K))
    darr = reshape(dvec, M, K)
    rarr = reshape(rvec, M, K)
    z = [CoarsePP(darr[m, :], rarr[m, :]) for m = 1:M]
    return SummaryVector{S,L}(tx), CoarsePPSet{S,L}(z)
end

## GILLESPIE MODEL
function (m::GillespieModel{S,L})(
    theta::AbstractParameters{S},
    ind::IndependentSimulation{S,L},
)::Tuple{SummaryVector{S,L},PPSet{S,L}} where {S} where {L}
    num_reactions = size(m.Sv.nu, 2)
    return m(theta, PPSet{S,L}(ind, num_reactions))
end

function (m::GillespieModel{S,L})(
    theta::AbstractParameters{S},
    cpp::CoarsePPSet{S,LL},
)::Tuple{SummaryVector{S,L},PPSet{S,L}} where {S} where {L} where {LL}
    return m(theta, PPSet{S,L}(cpp))
end

function (m::GillespieModel{S,L})(
    theta::AbstractParameters{S},
    pp::PPSet{S,L},
)::Tuple{SummaryVector{S,L},PPSet{S,L}} where {S} where {L}
    N, M = size(m.Sv.nu)
    T = m.T

    pp_out = PPSet{S,L}(deepcopy(pp.pp))

    t = zero(Float64)
    d = zeros(Float64, M)
    x = copy(m.Sv.x0)
    v = zeros(Float64, M)


    d_next = zeros(Float64, M)
    t_next = zeros(Float64, M)

    for j in eachindex(pp.pp)
        _d_next!(d_next, pp_out.pp, pp.pp, j)
    end

    tvec = Array{Float64,1}()
    xvec = Array{Int64,1}()

    while isless(t, T)
        append!(tvec, t)
        append!(xvec, x)

        m.Sv.v(v, theta, x)
        t_next .= (d_next .- d) ./ v
        (dt, j) = findmin(t_next)

        _d_next!(d_next, pp_out.pp, pp.pp, j)

        t += dt
        d .+= v .* dt
        x .+= m.Sv.nu[:, j]

    end
    append!(tvec, m.T)
    append!(xvec, xvec[end-N+1:end])
    K::Int64 = div(length(xvec), N)

    tx = BRNTraj{S}(tvec, reshape(xvec, N, K))
    return SummaryVector{S,L}(tx), pp_out

end

end

# include("brn-chain.jl")
