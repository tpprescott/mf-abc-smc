@everywhere include("../../source/mf-abc-smc.jl")
using .KuramotoOscillators

using JLD
using LaTeXStrings
using Statistics

data = load("./data/common_schedule_smc.jld")
epsvec = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
gen_labels = [latexstring("\\epsilon_$i = $(epsvec[i])") for i in 1:8]

# Times went awry due to sharing cluster
# First take: only use 1:40
# Second take: only use 10:50
# Third take: use everything
idx_take = 1:50
K = length(idx_take)

T_abc = data["T_abc"][:,idx_take]
ESS_abc = data["ESS_abc"][:,idx_take]
F_abc = data["F_abc"][:,idx_take]
eff_abc = ESS_abc./T_abc

T_mfabc = data["T_mfabc"][:,idx_take]
ESS_mfabc = data["ESS_mfabc"][:,idx_take]
F_mfabc = data["F_mfabc"][:,idx_take]
eff_mfabc = ESS_mfabc./T_mfabc

eta_mfabc = data["eta_mfabc"][:,:,idx_take]

using StatsPlots

# Efficiencies

# This takes the ESS of the last generation divided by simulation time of all generations
fig1a = plot(;
title="Efficiencies: ESS of last generation to total simulation time",
ylabel="Efficiency (ESS/sec)",
xtick=([1,2],["ABC-SMC", "MF-ABC-SMC"]),
legend=:none,
)
boxplot!(fig1a, fill(1,K), ESS_abc[end,:] ./ sum(T_abc,dims=1)')
boxplot!(fig1a, fill(2,K), ESS_mfabc[end,:] ./ sum(T_mfabc,dims=1)')
save("./out/pdf/efficiencies_smc.pdf", fig1a)

# See generations
fig2a = plot(;
title="Simulation time per generation",
ylabel="Simulation time (s)",
xlabel="Generation",
xtick=([1,2,3,4,5,6,7,8],gen_labels),
)
boxplot!(fig2a, repeat(1:8, K).-0.05, reshape(T_abc, 8*K, 1), label="ABC-SMC")
boxplot!(fig2a, repeat(1:8, K).+0.05, reshape(T_mfabc, 8*K, 1), label="MF-ABC-SMC")
save("./out/pdf/time_per_gen.pdf", fig2a)

fig2b = plot(;
title="ESS per generation",
ylabel="ESS",
xlabel="Generation",
xtick=([1,2,3,4,5,6,7,8],gen_labels),
)
boxplot!(fig2b, repeat(1:8, K).-0.05, reshape(ESS_abc, 8*K, 1), label="ABC-SMC")
boxplot!(fig2b, repeat(1:8, K).+0.05, reshape(ESS_mfabc, 8*K, 1), label="MF-ABC-SMC")
save("./out/pdf/ess_per_gen.pdf", fig2b)

fig2c = plot(;
       title="Efficiency per generation",
       ylabel="Efficiency (ESS/sec)",
       xlabel="Generation",
       xtick=([1,2,3,4,5,6,7,8], gen_labels)
       )
boxplot!(fig2c, repeat(1:8, 50).-0.05, reshape(eff_abc,8*50,1), label="ABC-SMC")
boxplot!(fig2c, repeat(1:8, 50).+0.05, reshape(eff_mfabc,8*50,1), label="MF-ABC-SMC")
plot!(fig2c; yscale=:log10, yticks=(2.0.^(-1:4),2.0.^(-1:4)))
save("./out/pdf/eff_per_gen.pdf", fig2c)


# Estimates and simulation times
fig3a = plot(;
xlabel=L"K",
ylabel="Simulation time (h)",
legend=:none,
)
fig3b = plot(;
xlabel=L"\omega_0",
ylabel="",
legend=:none,
title="Empirical posterior means of parameters"
)
fig3c = plot(;
xlabel=L"\gamma",
ylabel="",
legend=:right,
)

T_tot_abc = sum(T_abc, dims=1)'
T_tot_mfabc = sum(T_mfabc, dims=1)'

scatter!(fig3a, F_abc[1,:], T_tot_abc./3600)
scatter!(fig3a, F_mfabc[1,:], T_tot_mfabc./3600)
scatter!(fig3b, F_abc[2,:], T_tot_abc./3600)
scatter!(fig3b, F_mfabc[2,:], T_tot_mfabc./3600)
scatter!(fig3c, F_abc[3,:], T_tot_abc./3600, label="ABC-SMC")
scatter!(fig3c, F_mfabc[3,:], T_tot_mfabc./3600, label="MF-ABC-SMC")

plot!(
       fig3a;
       xticks = [round(minimum(vcat(F_abc[1,:], F_mfabc[1,:])), RoundDown, digits=2),
                 round(maximum(vcat(F_abc[1,:], F_mfabc[1,:])), RoundUp, digits=2),
                 round(median(vcat(F_abc[1,:], F_mfabc[1,:])), digits=2),
               ],
       xlim = (round(minimum(vcat(F_abc[1,:], F_mfabc[1,:])), RoundDown, digits=2),
           round(maximum(vcat(F_abc[1,:], F_mfabc[1,:])), RoundUp, digits=2),
           ),
       yticks = round.([median(T_tot_abc)./3600, median(T_tot_mfabc)./3600], digits=2),
)
plot!(
       fig3b;
       xticks = [round(minimum(vcat(F_abc[2,:], F_mfabc[2,:])), RoundDown, digits=3),
                 round(maximum(vcat(F_abc[2,:], F_mfabc[2,:])), RoundUp, digits=3),
                 round(median(vcat(F_abc[2,:], F_mfabc[2,:])), digits=3),
               ],
       xlim = (round(minimum(vcat(F_abc[2,:], F_mfabc[2,:])), RoundDown, digits=3),
           round(maximum(vcat(F_abc[2,:], F_mfabc[2,:])), RoundUp, digits=3),
           ),
       yticks = round.([median(T_tot_abc)./3600, median(T_tot_mfabc)./3600], digits=2),
)
plot!(
       fig3c;
       xticks = [round(minimum(vcat(F_abc[3,:], F_mfabc[3,:])), RoundDown, digits=3),
                 round(maximum(vcat(F_abc[3,:], F_mfabc[3,:])), RoundUp, digits=3),
                 round(median(vcat(F_abc[3,:], F_mfabc[3,:])), digits=3),
               ],
       xlim = (round(minimum(vcat(F_abc[3,:], F_mfabc[3,:])), RoundDown, digits=3),
           round(maximum(vcat(F_abc[3,:], F_mfabc[3,:])), RoundUp, digits=3),
           ),
       yticks = round.([median(T_tot_abc)./3600, median(T_tot_mfabc)./3600], digits=2),
)

using Plots.PlotMeasures
fig3abc = plot(fig3a,fig3b,fig3c;
layout=(1,3),
plot_title="Estimates and simulation times: generation 8",
size=(1200,400),
margin=25px,
)
save("./out/pdf/estimates_gen8.pdf", fig3abc)

# All of the eta! From every other generation, though.
fig5 = plot(;
title = "Continuation probabilities by generation",
xlabel=L"\eta_1",
ylabel=L"\eta_2",
xlim=(0,1),
ylim=(0,1),
)
for i in 1:2:7
scatter!(fig5, eta_mfabc[1,i,:], eta_mfabc[2,i,:];
 label=gen_labels[i+1],
 markersize=4,
 shape=:auto,
 )
end
save("./out/pdf/etas_gen.pdf", fig5)
