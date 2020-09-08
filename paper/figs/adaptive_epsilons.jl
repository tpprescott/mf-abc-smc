@everywhere include("../../source/mf-abc-smc.jl")

using .KuramotoOscillators
using StatsPlots, LaTeXStrings, Roots
using JLD

pvec(C) = [p.p for p in C]
qvec(C) = [p.q for p in C]
rvec(C, q) = likelihood(C,q)
αvec(C) = [p.sims[1].alpha for p in C]
tvec(C) = [p.sims[1].t for p in C]
dvec(C) = [p.sims[1].abc.d for p in C]
function Tvec(C)
    Tvec = zeros(Float64, length(C))
    for (i,p) in enumerate(C)
        length(p)==2 && (Tvec[i]=p.sims[2].t)
    end
    return Tvec
end
function Dvec(C)
    Dvec = zeros(Float64, length(C))
    for (i,p) in enumerate(C)
        length(p)==2 && (Dvec[i]=p.sims[2].abc.d)
    end
    return Dvec
end
_M(C) = (length.(C).==2)

struct FlexiPhi
    pvec::Vector{Float64}
    qvec::Vector{Float64}
    rvec::Vector{Float64}
    αvec::Vector{Float64}
    tvec::Vector{Float64}
    dvec::Vector{Float64}
    Tvec::Vector{Float64}
    Dvec::Vector{Float64}
    M::BitArray{1}
    function FlexiPhi(C::Cloud{S,T},q::ParameterSampler{T}) where S where T
        return new(pvec(C), qvec(C), rvec(C,q), αvec(C), tvec(C), dvec(C), Tvec(C), Dvec(C), _M(C))
    end
end

function get_coeffs(Φ::FlexiPhi, (tϵ, ϵ)::Tuple{Float64, Float64})
    Z = mean( (Φ.pvec ./ Φ.qvec) .* ((Φ.dvec .< tϵ) .+ Φ.M .* ((Φ.Dvec .< ϵ) .- (Φ.dvec .< tϵ)) ./ Φ.αvec ))
    W = mean( ((Φ.pvec .^ 2)./(Φ.rvec .* Φ.qvec)) .* ((Φ.dvec .< tϵ) .+ Φ.M .* ((Φ.Dvec .< ϵ) .- (Φ.dvec .< tϵ)) ./ Φ.αvec )) + eps()
    W_fp = mean( ((Φ.pvec .^ 2)./(Φ.rvec .* Φ.qvec)) .* ((Φ.dvec .< tϵ) .* (Φ.Dvec .>= ϵ)).*(Φ.M ./ Φ.αvec)) + eps()
    W_fn = mean( ((Φ.pvec .^ 2)./(Φ.rvec .* Φ.qvec)) .* ((Φ.dvec .>= tϵ) .* (Φ.Dvec .< ϵ)).*(Φ.M ./ Φ.αvec)) + eps()
    T_lo = mean( (Φ.rvec ./ Φ.qvec) .* Φ.tvec )
    T_hi_p = mean( (Φ.rvec ./ (Φ.qvec .* Φ.αvec)) .* (Φ.dvec .< tϵ) .* Φ.Tvec)
    T_hi_n = mean( (Φ.rvec ./ (Φ.qvec .* Φ.αvec)) .* (Φ.dvec .>= tϵ) .* Φ.Tvec)
    return Z, W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n
end

import .MFABCSMC.efficiency
function efficiency(Φ::FlexiPhi, (tϵ, ϵ)::Tuple{Float64,Float64}, (η1, η2)::Tuple{Float64, Float64}=(1.0, 1.0))
    Z, W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n = get_coeffs(Φ, (tϵ, ϵ))
    iszero(Z) && (return 0.0)
    return Z^2 / ((W + ((1/η1) - 1)*W_fp + ((1/η2)-1)*W_fn) * (T_lo + η1*T_hi_p + η2*T_hi_n))
end

import .MFABCSMC.Eta
function Eta(Φ::FlexiPhi, (tϵ, ϵ)::Tuple{Float64, Float64}, H::EtaSpace=EtaSpace(eps()))
    Z, W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n = get_coeffs(Φ, (tϵ, ϵ))
    return findmin(Phi(W, W_fp, W_fn, T_lo, T_hi_p, T_hi_n), H)[2]
end

# Now run it!


##########
# ABC-SMC


function step_SMC(C::Cloud{S,T}, ϵ::Float64, target::Float64) where S where T
    q = Importance(C)
    Φ = FlexiPhi(C, q)
    xmax = getepsilon(C)[1]

    f(x::Float64) = efficiency(Φ, (x, x), (1.0, 1.0))-target
    if f(eps())*f(xmax)<0.0
        ϵ_next = fzero(f, eps(), xmax)
    else
        ϵ_next = xmax
    end
    ϵ_next = max(eps(), ϵ_next)
    prob = MFABCProblem(
        C.abc.p,
        q,
        (C.abc.steps[1].abc.m, C.abc.steps[2].abc.m),
        (C.abc.steps[1].abc.y_obs, C.abc.steps[2].abc.y_obs),
        (ϵ_next, ϵ_next),
    )
    C_next = Cloud(prob, ESS, 400)
    return C_next, round(ϵ_next, digits=2), Φ
end

function DataAdaptiveEpsilon()

ϵ = [2.0]
max_T = 4
prob = MFABCProblem(prior, prior, (m_lo, m_hi), (y_lo, y_hi), (ϵ[end], ϵ[end]))

C_arr = [Cloud(prob, ESS, 400)]
target = efficiency(C_arr[end])
Φvec = [FlexiPhi(C_arr[end], prior)]

while (length(ϵ)<max_T)&(ϵ[end]>eps())
    C_next, ϵ_next, Φ_next = step_SMC(C_arr[end], ϵ[end], target)
    push!(C_arr, C_next)
    push!(ϵ, ϵ_next)
    push!(Φvec, Φ_next)
end

ϵ_range_prior = collect(0.0:0.01:ϵ[1][end])
predicted_efficiencies_prior = map(x->efficiency(Φvec[1], (x,x)), ϵ_range_prior)

ϵ_ranges = [collect(0.0:0.01:ϵ_i[end]) for ϵ_i in ϵ[1:end-1]]
predicted_efficiencies = [map(x->efficiency(Φ_i, (x,x)), ϵ_range_i) for (Φ_i, ϵ_range_i) in zip(Φvec[2:end], ϵ_ranges)]
actual_efficiencies = broadcast(efficiency, C_arr)

save("./data/adaptive_epsilon.jld",
"ϵ", ϵ,
"ϵ_range_prior", ϵ_range_prior,
"predicted_efficiencies_prior", predicted_efficiencies_prior,
"ϵ_ranges", ϵ_ranges,
"predicted_efficiencies", predicted_efficiencies,
"actual_efficiencies", actual_efficiencies,
)

param_ranges = ((1.,3.), (-2π, 2π), (0.,1.))
param_labels = [L"K", L"\omega_0", L"\gamma"]

fig_abc = plot_densities(C_arr[end], param_ranges,
    param_labels=param_labels,
    fn="/figs/data_abc_smc_adaptive",
    )
save("./out/pdf/posterior_abc_smc_adaptive.pdf", fig_abc)

end

function FigAdaptiveEpsilon(fn::String)

data = load(fn)
ϵ = data["ϵ"]
ϵ_range_prior = data["ϵ_range_prior"]
predicted_efficiencies_prior = data["predicted_efficiencies_prior"]
ϵ_ranges = data["ϵ_ranges"]
predicted_efficiencies = data["predicted_efficiencies"]
actual_efficiencies = data["actual_efficiencies"]

target = actual_efficiencies[1]
xmax = ϵ[1]
fig = hline([target], label="", linestyle=:dot, linecolor=:black)

plot!(fig; xlim=[0,xmax])
plot!(fig, ϵ_range_prior, predicted_efficiencies_prior;
    label=L"\hat q_1 = \pi",
    linecolor=1,
)

for (i,(ϵ_range_i, predicted_efficiencies_i)) in enumerate(zip(ϵ_ranges, predicted_efficiencies))
    plot!(fig, ϵ_range_i, predicted_efficiencies_i;
        label = latexstring("\\hat q_$(i+1)"),
        linecolor=i+1,
    )
end

ϵ_labels = string.(ϵ)

scatter!(fig, ϵ, actual_efficiencies;
 group=1:length(ϵ),
 seriescolor=1:length(ϵ),
 markershape=:star5,
 label="",
 markersize=5)

plot!(fig;
    xticks=(ϵ,ϵ_labels),
    xlabel=L"\epsilon",
    yticks=0.0:0.25:1.5,
    ylim=(0.0,1.5),
    ylabel="Efficiency (ESS/sec)",
    size=(300,200),
    legend=:topleft,
    legendfontsize=8,
)

save("./out/pdf/adaptive_epsilon.pdf", fig)
return nothing

end

############# MF-ABC-SMC

function step_MFABCSMC(C::Cloud{S,T}, ϵ::Float64, target::Float64; max_loops::Int64=5) where S where T
    q = Importance(C)
    Φ = FlexiPhi(C, q)

    xmax = getepsilon(C)[1]
    ϵ_steps = [xmax]

    η_steps = Vector{Tuple{Float64, Float64}}()

    for i in 1:max_loops
        # Get η to optimise efficiency at current ϵ value
        η_steps_next = Eta(Φ, (ϵ_steps[end], ϵ_steps[end]), EtaSpace(0.01))
        push!(η_steps, η_steps_next)

        f(x::Float64) = efficiency(Φ, (x, x), η_steps[end]) - target

        # Find ϵ that brings efficiency down to target - stop condition where epsilon doesn't decrease
        f(eps())*f(ϵ_steps[end])<0.0 || break
        ϵ_steps_next = round(max(eps(), fzero(f, eps(), ϵ_steps[end])), digits=2, RoundUp)
        (ϵ_steps_next >= ϵ_steps[end]) && break
        push!(ϵ_steps, ϵ_steps_next)

    end

    prob = MFABCProblem(
        C.abc.p,
        q,
        (C.abc.steps[1].abc.m, C.abc.steps[2].abc.m),
        (C.abc.steps[1].abc.y_obs, C.abc.steps[2].abc.y_obs),
        (ϵ_steps[end], ϵ_steps[end]),
        Eta{Kuramoto, LoFi}(η_steps[end]...),
    )
    C_next = Cloud(prob, ESS, 400)
    return C_next, ϵ_steps, η_steps, Φ
end


function DataAdaptiveEpsilonEta()

ϵ = [[2.0]]
η = [[(1.0, 1.0)]]
max_T = 4
prob = MFABCProblem(prior, prior, (m_lo, m_hi), (y_lo, y_hi), (ϵ[end][end], ϵ[end][end]))
C_arr = [Cloud(prob, ESS, 400)]
target = efficiency(C_arr[end])
Φvec = [FlexiPhi(C_arr[end], prior)]

while (length(ϵ)<max_T)&(ϵ[end][end]>eps())
    C_next, ϵ_steps, η_steps, Φ_next = step_MFABCSMC(C_arr[end], ϵ[end][end], target)
    push!(C_arr, C_next)
    push!(ϵ, ϵ_steps)
    push!(η, η_steps)
    push!(Φvec, Φ_next)
end

ϵ_range_prior = collect(0.0:0.01:ϵ[1][1])
predicted_efficiencies_prior = map(x->efficiency(Φvec[1], (x,x)), ϵ_range_prior)

ϵ_ranges = [collect(0.0:0.01:ϵ_i[end]) for ϵ_i in ϵ[1:end-1]]
predicted_efficiencies = [[map(x->efficiency(Φ_i, (x,x), η_ij), ϵ_range_i) for η_ij in η_i] for (Φ_i, ϵ_range_i, η_i) in zip(Φvec[2:end], ϵ_ranges, η[2:end])]
actual_efficiencies = broadcast(efficiency, C_arr)

save("./data/adaptive_epsilon_eta.jld",
"ϵ", ϵ,
"η", η,
"ϵ_range_prior", ϵ_range_prior,
"predicted_efficiencies_prior", predicted_efficiencies_prior,
"ϵ_ranges", ϵ_ranges,
"predicted_efficiencies", predicted_efficiencies,
"actual_efficiencies", actual_efficiencies,
)

param_ranges = ((1.,3.), (-2π, 2π), (0.,1.))
param_labels = [L"K", L"\omega_0", L"\gamma"]

fig_mfabc = plot_densities(C_arr[end], param_ranges,
    param_labels=param_labels,
    fn="/figs/data_mfabc_smc_adaptive",
    )
save("./out/pdf/posterior_mfabc_smc_adaptive.pdf", fig_mfabc)

end

function FigAdaptiveEpsilonEta(fn::String)

data = load(fn)
ϵ = data["ϵ"]
η = data["η"]
ϵ_range_prior = data["ϵ_range_prior"]
predicted_efficiencies_prior = data["predicted_efficiencies_prior"]
ϵ_ranges = data["ϵ_ranges"]
predicted_efficiencies = data["predicted_efficiencies"]
actual_efficiencies = data["actual_efficiencies"]

target = actual_efficiencies[1]
xmax = ϵ[1][end]

fig = hline([target];
    label="",
    linestyle=:dot,
    linecolor=:black,
    xlim=[0,xmax],
)
plot!(fig, ϵ_range_prior, predicted_efficiencies_prior;
    label=L"\hat r_1 = \pi; (\eta_1,\eta_2)=(1.0,1.0)",
    linecolor=1,
)

for (i, (ϵ_range_i, predicted_efficiencies_i, η_i)) in enumerate(zip(ϵ_ranges, predicted_efficiencies, η[2:end]))
    for predicted_efficiencies_ij in predicted_efficiencies_i[1:end-1]
        plot!(fig, ϵ_range_i, predicted_efficiencies_ij;
         label = "",
         linecolor=i+1,
         linestyle=:dot,
        )
    end
    plot!(fig, ϵ_range_i, predicted_efficiencies_i[end];
     label=latexstring("\\hat r_$(i+1); (\\eta_1, \\eta_2)=$(round.((η_i[end]),digits=2))"),
     linecolor=i+1,
     linestyle=:solid,
    )
end

epsilons = [ϵ_i[end] for ϵ_i in ϵ]
epsilon_labels = string.(epsilons)
epsilon_labels[3] = ""

scatter!(
    fig, epsilons, actual_efficiencies;
    group=1:length(ϵ),
    seriescolor=1:length(ϵ),
    markershape=:star5,
    label="",
    markersize=5,
)
plot!(fig;
    xticks=(epsilons, epsilon_labels),
    xlabel=L"\epsilon",
    yticks=0:2.5:10.0,
    ylims=(0.0,12.0),
    ylabel="Efficiency (ESS/sec)",
    size=(300,200),
    legend=:topleft,
    legendfontsize=8,
)
save("./out/pdf/adaptive_epsilon_eta.pdf", fig)
return nothing

end
