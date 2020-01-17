export EtaSpace, Eta, Phi
export geteta
using Statistics
import Statistics.mean

struct Phi
    W::Float64
    W_fp::Float64
    W_fn::Float64
    T_lo::Float64
    T_hi_p::Float64
    T_hi_n::Float64

    function Phi(
        W::Float64,
        W_fp::Float64,
        W_fn::Float64,
        T_lo::Float64,
        T_hi_p::Float64,
        T_hi_n::Float64,
)
        args = (W_fp, W_fn, T_lo, T_hi_p, T_hi_n)
        if any(isneg, args)
            error("$args contains negative value for constructing Phi")
        end
        return new(W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n)
    end
end
function (phi::Phi)(e1::Float64, e2::Float64)::Float64
    e1 += eps()
    e2 += eps()
    return (phi.W + phi.W_fp *((1.0/e1) - 1.0) + phi.W_fn*((1.0/e2) - 1.0)) *
           (phi.T_lo + phi.T_hi_p * e1 + phi.T_hi_n * e2)
end
function (phi::Phi)(e::Tuple{Float64,Float64})::Float64
    return phi(e...)
end
function (phi::Phi)(eta::Eta)::Float64
    return phi(eta.eta1, eta.eta2)
end

# Combine Phi from different clouds with weighting vector
import Base.+, Base.*
+(ϕ::Phi, ψ::Phi) = Phi(
    ϕ.W + ψ.W,
    ϕ.W_fp + ψ.W_fp,
    ϕ.W_fn + ψ.W_fn,
    ϕ.T_lo + ψ.T_lo,
    ϕ.T_hi_p + ψ.T_hi_p,
    ϕ.T_hi_n + ψ.T_hi_n,
    )
*(w::S, ϕ::Phi) where S<:Number = Phi(
    w*ϕ.W,
    w*ϕ.W_fp,
    w*ϕ.W_fn,
    w*ϕ.T_lo,
    w*ϕ.T_hi_p,
    w*ϕ.T_hi_n,
)

function mean(ϕs::Array{Phi, 1}, lambda::Array{Float64,1}=fill(1.0/length(ϕs), length(ϕs)))::Phi
    return sum(lambda.*ϕs)
end

# Find overall minimum of phi on non-negative orthant
function eta_bar(phi::Phi)::NTuple{2,Float64}
    (phi.W > (phi.W_fp + phi.W_fn)) || (return Inf64, Inf64)
    return sqrt(phi.T_lo * phi.W_fp / ((phi.W - phi.W_fp - phi.W_fn) * phi.T_hi_p)),
        sqrt(phi.T_lo * phi.W_fn / ((phi.W - phi.W_fp - phi.W_fn) * phi.T_hi_n))
end
function phi_bar(phi::Phi)::Float64
    (phi.W > (phi.W_fp + phi.W_fn)) || (return -Inf64)
    return (sqrt((phi.W - phi.W_fp - phi.W_fn) * phi.T_lo) +
            sqrt(phi.W_fp * phi.T_hi_p) + sqrt(phi.W_fn * phi.T_hi_n))^2
end

function _bounded(x::T, lo::T, hi::T)::T where {T}
    return max(lo, min(x, hi))
end
function eta1(phi::Phi, x::Float64, rho1::Float64 = 0.0)::Float64
    (~iszero(phi.T_hi_p) &
     ((phi.W - phi.W_fp - (1.0 - 1.0/x)*phi.W_fn)>0.0)) || (return 1.0)
    eta1_bar::Float64 = sqrt((phi.T_lo + (x * phi.T_hi_n)) * phi.W_fp /
                             (phi.T_hi_p *
                              (phi.W - phi.W_fp - (1.0 - 1.0/x)*phi.W_fn)))
    return _bounded(eta1_bar, rho1, 1.0)
end
function eta2(phi::Phi, x::Float64, rho2::Float64 = 0.0)::Float64
    (~iszero(phi.T_hi_n) &
     ((phi.W - (1.0 - 1.0/x)*phi.W_fp - phi.W_fn)>0.0)) || (return 1.0)
    eta2_bar::Float64 = sqrt((phi.T_lo + x * phi.T_hi_p) * phi.W_fn /
                             (phi.T_hi_n *
                              (phi.W - (1.0 - 1.0/x)*phi.W_fp - phi.W_fn)))
    return _bounded(eta2_bar, rho2, 1.0)
end

isprob(x) = (x >= zero(x)) & (x <= one(x))
struct EtaSpace
    rho1::Float64
    rho2::Float64
    function EtaSpace(rho1::Float64, rho2::Float64)
        all(isprob, (rho1, rho2)) || error("Invalid bounds")
        return new(rho1, rho2)
    end
    function EtaSpace()
        return EtaSpace(0.0, 0.0)
    end
    function EtaSpace(rho::Float64)
        return EtaSpace(rho, rho)
    end
end
import Base.in
function in(eta_val::Tuple{Float64,Float64}, H::EtaSpace)::Bool
    return (all(isprob, eta_val) && ((eta_val[1] >= H.rho1) & (eta_val[2] >= H.rho2)))
end

import Base.findmin
function findmin(phi::Phi, H::EtaSpace = EtaSpace())::Tuple{
    Float64,
    Tuple{Float64,Float64},
}
    eta_val = eta_bar(phi)
    in(eta_val, H) && (return (phi_bar(phi), eta_val))
    # Look on the border if eta_val doesn't live in H
    eta_list = [
        (1.0, eta2(phi, 1.0, H.rho2)),
        (eta1(phi, 1.0, H.rho1), 1.0),
        (H.rho1, eta2(phi, H.rho1, H.rho2)),
        (eta1(phi, H.rho2, H.rho1), H.rho2),
    ]
    phi_list = phi.(eta_list)
    phi_val, j = findmin(phi_list)
    return phi_val, eta_list[j]
end

# FINALLY
# Get an optimal Eta from phi, given (optional) space where Eta can live
function Eta{S,L}(phi::Phi, H::EtaSpace = EtaSpace()) where {S} where {L}
    phi_min, eta_min = findmin(phi, H)
    return Eta{S,L}(eta_min...)
end

###########
# Now need to construct Phi from a vector of particles


# FUNCTIONS TO IDENTIFY ROC (true/false positive/negatives)

getweight(s::MFABCStepOut) = s.abc.w
function getweight(sims::NTuple{N,MFABCStepOut{S,L} where L})::NTuple{
    N,
    Float64,
} where {N} where {S}

    return broadcast(getweight, sims)
end

# According to new epsilon pair
function _xx(sim1::MFABCStepOut, epsilons::Tuple{Float64,Float64})::Tuple{Bool}
    return (sim1.abc.d < epsilons[1],)
end
function _xx(
    sim1::MFABCStepOut{S,L} where {L},
    sim2::MFABCStepOut{S,L} where {L},
    epsilons::Tuple{Float64,Float64},
)::Tuple{Bool,Bool} where {S}
    return (sim1.abc.d < epsilons[1], sim2.abc.d < epsilons[2])
end
function _xx(pl::Particle, epsilons::Tuple{Float64,Float64})
    return _xx(pl.sims..., epsilons)
end

# FP/FN with new epsilon(s)
function _fp(pl::Particle, epsilons::Tuple{Float64,Float64})::Bool
    _xx(pl, epsilons) == (true, false)
end
function _fn(pl::Particle, epsilons::Tuple{Float64,Float64})::Bool
    _xx(pl, epsilons) == (false, true)
end
function _p(pl::Particle, epsilons::Tuple{Float64,Float64})::Bool
    _xx(pl, epsilons)[1]
end

# Put into estimates, involving the likelihood of the particle under the next importance distribution
function W(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    xx = _xx(pl, epsilons)
    w::Float64 = pl.p^2 / (pl.q * q_next)
    if length(xx)==1
        xx[1] ? (return w) : (return 0.0)
    else
        (xx == (false, false)) && (return 0.0)
        (xx == (true, true)) && (return w)
        (xx == (true, false)) && (return w*(1.0 - (1.0/pl.sims[1].alpha)))
        (xx == (false, true)) && (return w*(1.0/pl.sims[1].alpha))
        error("Something odd just happened")
    end
end

function W_xx(pl::Particle, q_next::Float64)::Float64
    return (pl.p)^2 / (pl.q * pl.sims[1].alpha * q_next)
end

function W_fp(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    _fp(pl, epsilons) && (return W_xx(pl, q_next))
    return 0.0
end
function W_fn(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    _fn(pl, epsilons) && (return W_xx(pl, q_next))
    return 0.0
end

function T_lo(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    return q_next * gettime(pl, 1) / pl.q
end

function T_hi_p(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    length(pl)==2 || (return 0.0)
    _p(pl, epsilons) && (return q_next * gettime(pl, 2) / (pl.q * pl.sims[1].alpha))
    return 0.0
end

function T_hi_n(pl::Particle, q_next::Float64, epsilons::Tuple{Float64,Float64})::Float64
    length(pl)==2 || (return 0.0)
    ~_p(pl, epsilons) && (return q_next * gettime(pl, 2) / (pl.q * pl.sims[1].alpha))
    return 0.0
end

function Phi(c::Cloud, qvec::Vector{Float64}, epsilons::Tuple{Float64, Float64}=getepsilon(c))
    get_estimate(f) = mean(x->f(x...,epsilons), zip(c.particles, qvec))
    return Phi(broadcast(
        get_estimate,
        (W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n),
    )...)
end

count_sim_pair(c::Cloud)::Int64 = count(p->(length(p)==2), c)

function Phi(
    C::Vector{Cloud{S,T}},
    qvec::Vector{Vector{Float64}},
    epsilons::Tuple{Float64, Float64}...
) where S where T

    Φ = map(x->Phi(x..., epsilons...), zip(C, qvec))
    λ = map(c_i->count_sim_pair(c_i)/length(c_i), C)
    return mean(Φ, λ)
end

function Phi(c::Cloud{S,T}, q::ParameterSampler{T}, eps...) where S where T
    qvec = likelihood(c,q)
    return Phi(c, qvec, eps...)
end
function Phi(c::Vector{Cloud{S,T}}, q::ParameterSampler{T}, eps...) where S where T
    qvec = pmap(c_i->likelihood(c_i, q), c)
    return Phi(c, qvec, eps...)
end

function Eta(
    c::Cloud{S,T},
    q,
    epsilons::Tuple{Float64, Float64}...;
    H::EtaSpace=EtaSpace(eps()),
) where {S} where {T}
    phi_val, eta_val = findmin(Phi(c, q, epsilons...), H)
    return Eta{S, LoFi}(eta_val...)
end

function Eta(
    c::Vector{Cloud{S,T}},
    q,
    epsilons::Tuple{Float64, Float64}...;
    H::EtaSpace=EtaSpace(eps()),
) where {S} where {T}
    phi_val, eta_val = findmin(Phi(c, q, epsilons...), H)
    return Eta{S, LoFi}(eta_val...)
end

function geteta(prob::MFABCProblem)
    return prob.steps[1].alpha
end
function geteta(c::Cloud)
    return geteta(c.abc)
end
