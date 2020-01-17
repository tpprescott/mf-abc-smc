@everywhere include("../../source/mf-abc-smc.jl")

using .KuramotoOscillators
using JLD, Plots, LaTeXStrings

sch = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
Ngen = length(sch)
Nclouds = 1
ESSgen = 400.0

smc = SMCProblem(prior, (y_lo, y_hi), (m_lo, m_hi), [2.0, 1.9])
Cloud(smc, length, 100, defensive=0.01, eta_min=0.01)
smc = SMCProblem(prior, (y_lo, y_hi), (m_lo, m_hi), sch)

F(p::Particle{S,T}) where S where T = p.w * p.theta.k
W(p::Particle{S,T}) where S where T = p.w
F(c::Cloud{S,T}) where S where T = sum(F, c)/sum(W, c)

T_abc = zeros(Ngen, Nclouds)
ESS_abc = zeros(Ngen, Nclouds)
length_abc = zeros(Ngen, Nclouds)
F_abc = zeros(3, Nclouds)

T_mfabc = zeros(Ngen, Nclouds)
ESS_mfabc = zeros(Ngen, Nclouds)
length_mfabc = zeros(Ngen, Nclouds)
F_mfabc = zeros(3, Nclouds)

eta_mfabc = zeros(2, Ngen-1, Nclouds)

using Random
Random.seed!(777)

for N in 1:Nclouds
    abc = Cloud(smc, ESS, ESSgen, defensive=0.0, eta_min=1.0)
    mfabc = Cloud(smc, ESS, ESSgen, defensive=0.01, eta_min=0.01)

    T_abc[:,N] = gettime.(abc)
    ESS_abc[:,N] = ESS.(abc)
    length_abc[:,N] = length.(abc)
    F_abc[:,N] = Vector(F(abc[end]))

    T_mfabc[:,N] = gettime.(mfabc)
    ESS_mfabc[:,N] = ESS.(mfabc)
    length_mfabc[:,N] = length.(mfabc)
    F_mfabc[:,N] = Vector(F(mfabc[end]))

    for (j, eta) in enumerate(geteta.(mfabc[2:end]))
        eta_mfabc[:,j,N] = [eta.eta1, eta.eta2]
    end

    if N==Nclouds
        param_ranges = ((1.,3.), (-2π, 2π), (0.,1.))
        param_labels = [L"K", L"\omega_0", L"\gamma"]

        fig_abc = plot_densities(abc[end], param_ranges,
            param_labels=param_labels,
            fn="abc_smc_ESS400",
            )
        fig_mfabc = plot_densities(mfabc[end], param_ranges,
            param_labels=param_labels,
            fn="mfabc_smc_ESS400"
            )

        save("./out/pdf/posterior_abc_smc_ESS400.pdf", fig_abc)
        save("./out/pdf/posterior_mfabc_smc_ESS400.pdf", fig_mfabc)

    end

end
Random.seed!()
#=
save("./data/common_schedule_smc.jld",
"T_abc", T_abc,
"ESS_abc", ESS_abc,
"length_abc", length_abc,
"F_abc", F_abc,
"T_mfabc", T_mfabc,
"ESS_mfabc", ESS_mfabc,
"length_mfabc", length_mfabc,
"F_mfabc", F_mfabc,
"eta_mfabc", eta_mfabc,
)
=#
