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
# Common variance only uses the positive weights.
function Importance(
    theta::Vector{RealParameter{S,N}},
    w::Vector{Float64},
    def::Float64,
    prior::ParameterSampler{RealParameter{S,N}},
) where {S} where {N}

    nz = broadcast(isnz, w)
    w_pos = broadcast(pospart, w[nz])
    w_pos ./= sum(w_pos)
    idx = Categorical(w_pos)

    thetastar = sum(w_pos.*theta[nz])
    twice_sample_var = 2.0 * sum([w_i*sqr(t_i - thetastar) for (w_i, t_i) in zip(w_pos, theta[nz])])

    common_perturbation(t::RealParameter{S,N}) =
        RealDist{S,N}(Normal.(t.k, sqrt.(twice_sample_var.k)))

    K = broadcast(common_perturbation, theta[nz])

    return Importance{RealParameter{S,N}}(w[nz], idx, K, def, prior)
end
