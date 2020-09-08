#########################
# Standard parameter space is R^n
# Need to specify operations on parameters
# And types of samplers
#########################

export RealParameter

using StaticArrays
struct RealParameter{S,N} <: AbstractParameters{S}
    k::SVector{N,Float64}
end
function RealParameter{S,N}(k::Vararg{Float64,N}) where {S} where {N}
    return RealParameter{S,N}(SVector(k))
end

for op in (:+, :-, :*, :/)
    @eval Base.$op(
        x::RealParameter{S,N},
        y::RealParameter{S,N},
    ) where {S} where {N} = RealParameter{S,N}($op(x.k, y.k))
    @eval Base.$op(x::RealParameter{S,N}, lambda::Number) where {S} where {N} =
        RealParameter{S,N}($op(x.k, lambda))
    @eval Base.$op(lambda::Number, x::RealParameter{S,N}) where {S} where {N} =
        RealParameter{S,N}($op(lambda, x.k))
end
function sqr(x::RealParameter{S,N}) where {S} where {N}
    return RealParameter{S,N}(sqr.(x.k))
end
function Base.zero(::Type{T}) where T<:RealParameter{S,N} where S where N
    return T(zero(SVector{N, Float64}))
end
Base.zero(x::RealParameter{S,N}) where S where N = zero(RealParameter{S,N})

##### Samplers of R^n

export RealDist, CorrDist

# Independent distributions for each component
using Distributions
struct RealDist{S,N} <: ParameterSampler{RealParameter{S,N}}
    dist::SVector{N,UnivariateDistribution}
end
function RealDist{S,N}(d::Vararg{UnivariateDistribution,N}) where {S} where {N}
    return RealDist{S,N}(SVector(d))
end
function RealDist{S}(d::Vararg{UnivariateDistribution,N}) where {S} where {N}
    return RealDist{S,N}(d)
end
function (p::RealDist{S,N})()::Tuple{
    RealParameter{S,N},
    Float64,
} where {S} where {N}
    theta = RealParameter{S,N}(broadcast(rand, p.dist))
    return theta, p(theta)
end
function (p::RealDist{S,N})(t::RealParameter{S,N})::Float64 where {S} where {N}
    lh::Float64 = 1.0
    for (d_i, k_i) in zip(p.dist, t.k)
        lh *= pdf(d_i, k_i)
    end
    return lh
end

# Multivariate distribution across all components
struct CorrDist{S,N} <: ParameterSampler{RealParameter{S,N}}
    dist::MultivariateDistribution
    function CorrDist{S}(dist::MultivariateDistribution) where {S}
        return new{S,length(dist)}(dist)
    end
end
function (p::CorrDist{S,N})()::Tuple{
    RealParameter{S,N},
    Float64,
} where {S} where {N}
    theta = RealParameter{S,N}(rand(p.dist))
    return theta, p(theta)
end
function (p::CorrDist{S,N})(t::RealParameter{S,N})::Float64 where {S} where {N}
    return pdf(p.dist, t.k)
end

# How to construct an importance distribution from a weighted sample
# This needs to be made more flexible: I'm hard-coding a standard approach
# (common Gaussian perturbation kernel with common variance)
function Importance(
    theta::Vector{RealParameter{S,N}},
    w::Vector{Float64},
    prior::ParameterSampler{RealParameter{S,N}};
    positivity_fun = abs,
) where {S} where {N}

    ww = copy(w)
    broadcast!(positivity_fun, ww, ww)
    p = ispos.(ww)
    wt = Weights(ww[p])

    i=1
    arr = Array{Float64, 2}(undef, N, sum(p))
    for (i, theta_i) in enumerate(theta[p])
        view(arr, :, i) .= theta_i.k
    end

    twice_sample_var = 2.0 * var(arr, wt, 2)

    common_perturbation(t::RealParameter{S,N}) =
        RealDist{S,N}(Normal.(t.k, sqrt.(twice_sample_var)))

    K = broadcast(common_perturbation, theta[p])

    return Importance{RealParameter{S,N}}(wt, K, prior)
end
