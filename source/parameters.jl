# Generic functions to export: generate and likelihood
export generate, likelihood

function generate(p::ParameterSampler{T})::Tuple{T,Float64} where {T}
    return p()
end
function generate(p::ParameterSampler{T}, N::Int64)::Tuple{
    Array{T,1},
    Array{Float64,1},
} where {T}
    sample = Array{T,1}(undef, N)
    lhvec = Array{Float64,1}(undef, N)
    for i = 1:N
        sample[i], lhvec[i] = generate(p)::Tuple{T,Float64}
    end
    return sample, lhvec
end

function likelihood(theta::T, p::ParameterSampler{T})::Float64 where {T}
    return p(theta)
end
likelihood(theta::Vector{T}, p::ParameterSampler{T}) where {T} =
    broadcast(likelihood, theta, [p])
likelihood(theta::T, p::Vector{<:ParameterSampler{T}}) where {T} =
    broadcast(likelihood, [theta], p)


# Set up parameter spaces:
## R^n
# Set up parameter samplers:
## n independent distributions
## Multivariate distribution
## Importance distribution out of weighted sample

#=
HELPER FUNCTIONS
=#

isnz(x)::Bool = ~iszero(x)
ispos(x)::Bool = (x > zero(x))
isneg(x)::Bool = (x < zero(x))
pospart(x::Float64)::Float64 = max(0.0, x)
negpart(x::Float64)::Float64 = pospart(-x)
sqr(x) = x^2

# In the below, we assume T is a vector of parameters (but it could be anything)
function weighted_mean(x::Vector{T}, w::Vector{Float64}) where {T}
    return sum(w .* x) / sum(w)
end
function weighted_mean(F, x::Vector{T}, w::Vector{Float64}) where {T}
    return weighted_mean(broadcast(F, x), w)
end


#### IMPORTANCE SAMPLERS FOR ANY PARAMETER SPACE
export Importance

using StatsBase
# Importance distribution formed from weighted sample
struct Importance{T} <: ParameterSampler{T}
    w::Weights
    K::Vector{<:ParameterSampler{T}}
    prior::ParameterSampler{T}
end

function (q::Importance{T})()::Tuple{T,Float64} where {T}

    K_rand = sample(q.K, q.w)
    theta::T, _ = generate(K_rand)
    prior_likelihood::Float64 = likelihood(theta, q.prior)
    prior_likelihood == 0.0 && (return q())
    
    lh = q(theta)
    
    return theta, lh
end

function importance_likelihood_component(
    w_i::Float64,
    K_i::ParameterSampler{T},
    theta::T,
)::Float64 where T
return w_i * likelihood(theta, K_i)
end

function (q::Importance{T})(theta::T)::Float64 where {T}
    prior_likelihood = likelihood(theta, q.prior)
    (prior_likelihood == 0.0) && (return 0.0)

    lh = 0.0
    for (w_i, K_i) in zip(q.w, q.K)
        lh += ispos(w_i) ? w_i*likelihood(theta, K_i) : zero(lh)
    end
    lh /= q.w.sum

    return lh
end

## PARAMETER SPACERS AND THEIR SAMPLERS
include("realparameter.jl")

# Maybe code plot some plots as ways of looking at parameters
