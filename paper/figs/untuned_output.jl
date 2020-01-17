@everywhere include("../../source/mf-abc-smc.jl")
using .KuramotoOscillators

using LaTeXStrings
using Plots
using JLD

abc_rs = MFABCProblem(prior, prior, (m_lo, m_hi), (y_lo, y_hi), (0.5, 0.5));
abc_smc = SMCProblem(prior, (y_lo, y_hi), (m_lo, m_hi), [2.0, 1.5, 1.0, 0.5]);
mfabc_rs = MFABCProblem(prior, prior, (m_lo, m_hi), (y_lo, y_hi), (0.5, 0.5), Eta{Kuramoto,LoFi}(0.5, 0.5))

# Initial compilation
Cloud(abc_rs, length, 100)
Cloud(abc_smc, length, 100, defensive=0.0, eta_min=1.0)
Cloud(mfabc_rs, length, 100)

get_results(C::Cloud) = (ESS(C), gettime(C))
get_results(Cvec::Array{Cloud{T,S},1}) where T where S = (ESS(Cvec[end]), sum(gettime.(Cvec)))

using Random
Random.seed!(666)
C_abc_rs = Cloud(abc_rs, length, 6000)
C_abc_smc = Cloud(abc_smc, length, 1500, defensive=0.0, eta_min=1.0)
C_mfabc_rs = Cloud(mfabc_rs, length, 6000)

results_abc_rs = get_results(C_abc_rs)
results_abc_smc = get_results(C_abc_smc)
results_mfabc_rs = get_results(C_mfabc_rs)

param_ranges = ((1.,3.), (-2π, 2π), (0.,1.))
param_labels = [L"K", L"\omega_0", L"\gamma"]
fig_abc_rs = plot_densities(C_abc_rs, param_ranges;
    param_labels=param_labels,
    fn="data_abc_rs",
)
fig_abc_smc = plot_densities(C_abc_smc[end], param_ranges;
    param_labels=param_labels,
    fn="data_abc_smc",
)
fig_mfabc_rs = plot_densities(C_mfabc_rs, param_ranges;
    param_labels=param_labels,
    fn="data_mfabc_rs",
)

save("./out/pdf/posterior_abc_rs.pdf", fig_abc_rs)
save("./out/pdf/posterior_abc_smc.pdf", fig_abc_smc)
save("./out/pdf/posterior_mfabc_rs.pdf", fig_mfabc_rs)

save("./data/untuned_output.jld",
"results_abc_rs", results_abc_rs,
"results_abc_smc", results_abc_smc,
"results_mfabc_rs", results_mfabc_rs,
)
