export getdistance, getweight, gettheta, gettime, getalpha, getepsilon
export ESS, efficiency

###### Functions of clouds
ispos(p::Particle)::Bool = ispos(p.w)
isneg(p::Particle)::Bool = isneg(p.w)
isnz(p::Particle)::Bool = isnz(p.w)
islength(p::Particle, i::Integer) = (length(p) == i)
sqr(x) = x^2

# Distance
getdistance(s::MFABCStepOut) = s.abc.d
getdistance(p::Particle, j::Integer) = getdistance(p.sims[j])
getdistance(p::Particle) = broadcast(getdistance, p.sims)

# Weight
getweight(p::Particle) = p.w

# Theta
function gettheta(p::Particle{S,T})::T where {S} where {T}
    return p.theta
end

# Time
gettime(s::MFABCStepOut)::Float64 = s.t
function gettime(p::Particle)::Float64
    t::Float64 = zero(Float64)
    for s in p.sims
        t += gettime(s)
    end
    return t
end
gettime(p::Particle, i::Integer) = gettime(p.sims[i])

# Alpha
function getalpha(step::MFABCStep{S,L})::AbstractContinuationProbability{
    S,
    L,
} where {S} where {L}
    return step.alpha
end
function getalpha(mfabc::MFABCProblem{S,T}) where {S} where {T}
    return broadcast(getalpha, mfabc.steps)
end
getalpha(c::Cloud) = getalpha(c.abc)

# Epsilons
getepsilon(abc_step::ABCStep) = abc_step.epsilon
getepsilon(step::MFABCStep) = getepsilon(step.abc)
getepsilon(mfabc::MFABCProblem) = broadcast(getepsilon, mfabc.steps)
getepsilon(c::Cloud) = getepsilon(c.abc)

function ESS(w::AbstractArray{Float64,1})::Float64
    sum(w)<=0.0 && (return 0.0)
    return sum(w)^2/sum(w.^2)
end
function ESS(c::Cloud)::Float64
     length(c)==0 && (return 0.0)
     sum(getweight, c.particles) <= 0 && (return 0.0)
     return (sum(getweight, c.particles))^2 / sum(sqr ∘ getweight, c.particles)
end
function gettime(c::Cloud)::Float64
    length(c)==0 && (return 0.0)
    return sum(gettime, c)
end
function efficiency(c)::Float64
    T = gettime(c)
    T==0.0 && (return 0.0)
    return ESS(c)/T
end


# Importance distribution comes from array of particles in c
# Assume default of taking absolute weight to make the importance distribution
function Importance(c::Cloud{S,T})::Importance{T} where {T} where {S}
    θ = [gettheta(p) for p in c]
    return Importance(
        gettheta.(c),
        getweight.(c),
        c.abc.p,
    )
end
function likelihood(c::Cloud{S,T}, q::ParameterSampler{T}) where {S} where {T}
    return broadcast(likelihood, gettheta.(c), Ref(q))
end

Base.getindex(c::Cloud, i...) = getindex(c.particles, i...)
Base.firstindex(c::Cloud) = 1
Base.lastindex(c::Cloud) = length(c.particles)
Base.iterate(c::Cloud) = iterate(c.particles)
Base.iterate(c::Cloud, i::Int64) = iterate(c.particles, i)
Base.length(c::Cloud) = length(c.particles)

# Following function is a fudge:
# The abc entry in the resulting cloud is
# NOT
# the ABC problem generating all particles.
function Base.vcat(C1::Cloud{S,T}, C::Cloud{S,T}...)::Cloud{S,T} where S where T
    return Cloud{S,T}(vcat(C1.particles, (c.particles for c in C)...), C1.abc)
end
