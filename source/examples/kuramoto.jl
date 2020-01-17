module KuramotoOscillators

export Kuramoto, KuramotoSummary
export Oscillators, mAnsatz
export prior, y_lo, y_hi, m_lo, m_hi

using DifferentialEquations
using Distributions
using LinearAlgebra
using ..MFABCSMC
#using LightGraphs
#using SparseArrays

struct Kuramoto <: System
    N::Int64
end

#########
# PARAMETERS

# Coupling strength; median frequency; frequency spread
prior = RealDist{Kuramoto}(
    Uniform(1.0, 3.0), # Coupling strength
    Uniform(-2π, 2π),   # Mean angular velocity
    Uniform(0.0, 1.0), # Dispersion parameter of (Cauchy) frequency distribution
)

#########
# MODELS

struct Oscillators <: AbstractModel{Kuramoto, HiFi}
    nw::Kuramoto
    x0::Vector{Float64}
    T::Float64
    function Oscillators(nw::Kuramoto, x0::Vector{Float64}, T::Float64)
        (length(x0)==nw.N) && return new(nw, x0, T)
        error("Initial condition isn't the right length")
    end
end

struct mAnsatz <: AbstractModel{Kuramoto, LoFi}
    nw::Kuramoto
    x0::Vector{Float64}
    T::Float64
    function mAnsatz(nw::Kuramoto, x0::Vector{Float64}, T::Float64)
        (length(x0)==nw.N) && return new(nw, x0, T)
        error("Initial condition isn't the right length")
    end
end

num_nodes = 256
m_hi = Oscillators(Kuramoto(num_nodes), zeros(num_nodes), 30.0)
m_lo = mAnsatz(m_hi.nw, m_hi.x0, m_hi.T)


#=
struct m2Ansatz <: AbstractModel{Kuramoto, LoFi}
    nw::Kuramoto
    x0::Vector{Float64}
    T::Float64
    function m2Ansatz(nw::Kuramoto, x0::Vector{Float64}, T::Float64)
        (length(x0)==nw.N) && return new(nw, x0, T)
        error("Initial condition isn't the right length")
    end
end
=#

########
# SUMMARY SPACE IS DETERMINED BY THE DATA

# This struct is to pick a timepoint to get an idea of the transient
struct SSGenerator
    t::Float64
end

struct DaidoTrajectory{S,L} <: AbstractSummaryStat{S,L}
    t::Vector{Float64}
    R::Vector{Float64}
    ϕ::Vector{Float64}
    function DaidoTrajectory{S,L}(t,R,ϕ) where S where L
        if issorted(t)
            if (length(R)==length(t))&&(length(ϕ)==length(t))
                return new{S,L}(t,R,ϕ)
            else
                error("Vectors of unequal length")
            end
        else
            error("Unsorted times")
        end
    end
end

function SSGenerator(traj::DaidoTrajectory)
    Rstar = trap_int(traj.R, traj.t)./traj.t[end]
    j = findfirst(abs.(traj.R .- Rstar) .<= abs(traj.R[1] - Rstar)/2.0)
    return SSGenerator(traj.t[j])
end

struct KuramotoSummary{S,L} <: AbstractSummaryStat{S,L}
    Rstar::Float64
    RHalf::Float64
    omega::Float64
end

function trap_int(u::Vector{T}, t::Vector{Float64}) where T
    return sum( (u[1:end-1]+u[2:end]).*diff(t) )/2.0
end
function KuramotoSummary(u::DaidoTrajectory{S,L}, g::SSGenerator) where S where L
    Rstar = trap_int(u.R, u.t)./u.t[end]
    j = findfirst(u.t .>= g.t)
    RHalf = u.R[j]
    omega = (u.ϕ[end] - u.ϕ[1])/u.t[end]
    return KuramotoSummary{S,L}(Rstar, RHalf, omega)
end

import .MFABCSMC.distance
function distance(z1::KuramotoSummary{S,L},z2::KuramotoSummary{S,L})::Float64 where S where L
    dRstar2 = (z1.Rstar^2 - z2.Rstar^2)^2
    dRHalf = (z1.RHalf - z2.RHalf)^2
    domega = (z1.omega - z2.omega)^2
    return sqrt(4.0*dRstar2 + dRHalf + domega)
end

############ SIMULATIONS

## HIGH FIDELITY

function (m::Oscillators)(theta::AbstractParameters{S},
          i::IndependentSimulation{S};
)::Tuple{KuramotoSummary{S,HiFi}, AbstractNoise{S,HiFi}} where S<:Kuramoto where L

    traj = _get_trajectory(m, theta)
    return KuramotoSummary(traj, G_data), IndependentSimulation{S,HiFi}()

end

function _get_trajectory(m::Oscillators, theta::AbstractParameters{S}) where S<:Kuramoto
    ω_vec = rand(Cauchy(theta.k[2], theta.k[3]), m.nw.N)
    prob = ODEProblem(_dphase, m.x0, (0.0, m.T), (ω_vec, theta))
    u = solve(prob)

    ts = 0:0.01:m.T
    Zvec::Vector{Complex} = _Z.(map(u, ts))

    return DaidoTrajectory{S,HiFi}(collect(ts), abs.(Zvec), unwrap!(angle.(Zvec)))
end

function _dphase(du::Vector{Float64}, u::Vector{Float64}, (ω_vec, theta), t)
    N = length(u)
    K = theta.k[1]/N
    du[:] = ω_vec
    for i in 1:N-1
        for j in i+1:N
            _ij = K*sin(u[j]-u[i])
            du[i] += _ij
            du[j] -= _ij
        end
    end
    return nothing
end

_Z(u::Vector{Float64})::Complex = mean(exp.(im .* u))
_R(u::Vector{Float64})::Float64 = abs(_Z(u))
_Ω(u::Vector{Float64})::Float64 = angle(_Z(u))


function unwrap(v, inplace=false)
  # currently assuming an array
  unwrapped = inplace ? v : copy(v)
  for i in 2:length(v)
    while unwrapped[i] - unwrapped[i-1] >= pi
      unwrapped[i] -= 2pi
    end
    while unwrapped[i] - unwrapped[i-1] <= -pi
      unwrapped[i] += 2pi
    end
  end
  return unwrapped
end
unwrap!(v) = unwrap(v, true)

## LOW FIDELITY
function (m::mAnsatz)(theta::AbstractParameters{S},
                          i::IndependentSimulation{S},
)::Tuple{KuramotoSummary{S,LoFi}, AbstractNoise{S,LoFi}} where S<:Kuramoto

    traj = _get_trajectory(m, theta)
    return KuramotoSummary(traj, G_data), IndependentSimulation{S, LoFi}()
end

function _get_trajectory(m::mAnsatz, theta::AbstractParameters{S}) where S<:Kuramoto
    Z0 = _Z(m.x0)
    R_0 = abs(Z0)
    ϕ_0 = angle(Z0)

    β = theta.k[1]/2
    α = β - theta.k[3]

    prob = ODEProblem(_dRϕ_m, [R_0], (0.0, m.T), (α, β))
    u = solve(prob)

    ts = 0:0.01:m.T
    return DaidoTrajectory{S, LoFi}(collect(ts), map(t->u(t)[1], ts), map(t->ϕ_0 + theta.k[2]*t, ts))
end

function _dRϕ_m(du, u, (α, β), t)
    du[1] = α*u[1] - β*u[1]^3
end

########### SYNTHETIC DATA
# Generate some synthetic data (and the summary statistic that will be used to compare)
function synthetic_data(theta::RealParameter{S}) where S <: Kuramoto
    traj = _get_trajectory(m_hi, theta)
    G = SSGenerator(traj)
    y_hi = KuramotoSummary(traj, G)
    y_lo = KuramotoSummary{S, LoFi}(y_hi.Rstar, y_hi.RHalf, y_hi.omega)
    return y_lo, y_hi, G
end

const theta = RealParameter{Kuramoto,3}([2.0, π/3, 0.1])

using Random
Random.seed!(1457)
const y_lo, y_hi, G_data = synthetic_data(theta)
Random.seed!()

end
# smc = SMCProblem(prior, (y_lo, y_hi), (m_lo, m_hi), [1.0, 0.75, 0.5])
