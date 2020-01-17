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

using Distributions
# Importance distribution formed from weighted sample
struct Importance{T} <: ParameterSampler{T}
    w::Vector{Float64}
    idx::Categorical
    K::Vector{<:ParameterSampler{T}}
    def::Float64
    prior::ParameterSampler{T}
    function Importance{T}(w, idx, K, def, prior) where {T}
        if (def < zero(def)) || (def > one(def))
            error("Incorrect defensive parameter (between 0 and 1 inclusive)")
        end
        return new{T}(w, idx, K, def, prior)
    end
end
function (q::Importance{T})()::Tuple{T,Float64} where {T}

    u = rand(2)
    use_prior_prob = (q.def*sum(q.w))/(q.def*sum(q.w) + (1-q.def)*sum(pospart, q.w))
    if u[1] < use_prior_prob
        theta, prior_likelihood = generate(q.prior)
    else
        theta::T, _ = generate(q.K[rand(q.idx)])
        prior_likelihood::Float64 = likelihood(theta, q.prior)
        prior_likelihood == 0.0 && (return q())
    end

    lhvec = likelihood(theta, q.K)
    lhvec .*= q.w ./ sum(q.w)

    F = q.def*prior_likelihood + (1.0-q.def)*sum(pospart, lhvec)
    G = (1.0-q.def)*sum(negpart, lhvec)
    alpha::Float64 = max(1.0 - (G/F), q.def * prior_likelihood / F)

    if u[2] < alpha
        return theta, alpha * F
    else
        return q()
    end
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

    importance_likelihood = 0.0
    for (w_i, K_i) in zip(q.w, q.K)
        importance_likelihood += importance_likelihood_component(w_i, K_i, theta)
    end
    importance_likelihood /= sum(q.w)

    lh::Float64 = max(q.def * prior_likelihood,
        q.def*prior_likelihood + (1-q.def)*importance_likelihood)
    return lh
end

## PARAMETER SPACERS AND THEIR SAMPLERS
include("realparameter.jl")

# Maybe code plot some plots as ways of looking at parameters
