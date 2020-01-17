struct Chain <: System end

# Define parameter space
ChainPars = RealParameter{Chain,3}
prior = RealDist{Chain,3}(Uniform(1, 20), Uniform(1, 10), Uniform(1, 10))

# Define summary space
using LinearAlgebra
function xbar(t::Array{Float64,1}, x::Array{<:Real,1})::Float64
    return sum(diff(t) .* x[1:end-1]) / (t[end] - t[1])
end
function xvar(t::Array{Float64,1}, x::Array{<:Real,1})::Float64
    return xbar(t, (x .- xbar(t, x)) .^ 2)
end
function summ_stat(tx::BRNTraj{Chain}, horizon::Float64)::Array{Float64,1}
    t_start::Float64 = tx.t[end] - horizon
    j = searchsortedlast(tx.t, t_start)
    j == 0 && error("Looking back too far.")

    t_trunc = tx.t[j+1:end]
    pushfirst!(t_trunc, t_start)
    x_trunc = tx.x[:, j:end]

    return [
        xbar(t_trunc, x_trunc[1, :]),
        xbar(t_trunc, x_trunc[2, :]),
        xvar(t_trunc, x_trunc[1, :]),
        xvar(t_trunc, x_trunc[2, :]),
    ]
end

function SummaryVector{Chain,L}(tx::BRNTraj{Chain}) where {L}
    return SummaryVector{Chain,L}(summ_stat(tx, 20.0))
end


# Define the BRN associated with this system
const S = [1 -1 0; 0 1 -1]

function propensity!(v, theta::ChainPars, x)
    v[1] = theta.k[1]
    v[2] = theta.k[2] * x[1]
    v[3] = theta.k[3] * x[2]
    return nothing
end
const x0 = [0, 0]

const brn = BRN{Chain}(S, x0, propensity!)

# STEADY STATE MEAN IS (k[0]/k[1], k[0]/k[2])
#

const T = 100.0
const tau = 0.1

# Nominal parameters
const theta = ChainPars((10.0, 3.0, 2.0))

# Define high and low fidelity models
const c_hi = GillespieModel{Chain,HiFi}(brn, T)
const c_lo = TauLeapModel{Chain,LoFi}(brn, T, tau)


# Reproducible "data"
using Random
Random.seed!(100)
y_obs_hi = simulate(c_hi, theta).y
y_obs_lo = SummaryVector{Chain,LoFi}(y_obs_hi.y)
Random.seed!()
