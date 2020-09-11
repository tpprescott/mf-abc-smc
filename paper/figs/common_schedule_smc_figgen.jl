include("../../source/mf-abc-smc.jl")
using .KuramotoOscillators

using JLD
using LaTeXStrings
using Statistics

data = load("./data/common_schedule_smc.jld")
epsvec = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
gen_labels = [latexstring("\\epsilon_$i = $(epsvec[i])") for i in 1:8]
gen_labels = 1:8

# Times went awry due to sharing cluster
# First take: only use 1:40
# Second take: only use 10:50
# Third take: use everything
idx_take = 1:50
K = length(idx_take)

T_abc = data["T_abc"][:,idx_take]
ESS_abc = data["ESS_abc"][:,idx_take]
F_abc = data["F_abc"][:,idx_take]
N_abc = data["length_abc"][:,idx_take]
eff_abc = ESS_abc./T_abc
T_per_N_abc = T_abc./N_abc

T_mfabc = data["T_mfabc"][:,idx_take]
ESS_mfabc = data["ESS_mfabc"][:,idx_take]
F_mfabc = data["F_mfabc"][:,idx_take]
N_mfabc = data["length_mfabc"][:,idx_take]
eff_mfabc = ESS_mfabc./T_mfabc
T_per_N_mfabc = T_mfabc./N_mfabc

eta_mfabc = data["eta_mfabc"][:,:,idx_take]

using StatsPlots

# Efficiencies

# This takes the ESS of the last generation divided by simulation time of all generations
#=
fig1a = plot(;
title="Efficiencies: ESS of last generation to total simulation time",
ylabel="Efficiency (ESS/sec)",
xtick=([1,2],["ABC-SMC", "MF-ABC-SMC"]),
legend=:none,
)
boxplot!(fig1a, fill(1,K), ESS_abc[end,:] ./ sum(T_abc,dims=1)')
boxplot!(fig1a, fill(2,K), ESS_mfabc[end,:] ./ sum(T_mfabc,dims=1)')
plot!(fig1a; ylim=[0,Inf])
save("./out/pdf/efficiencies_smc.pdf", fig1a)
=#


# See generations

L = @layout [
       a{0.05h};
       [b{0.7w} c];
       d{0.05h};
       [e{0.7w} f];
       g{0.05h};
       [h{0.7w} i]
]

fig2 = plot(; layout=L, ylim=[0, Inf], size=(750,900))
plot!(fig2; subplot=1, title = "Simulation time per proposal", grid = false, showaxis = false, bottom_margin = -50Plots.px)
plot!(fig2; subplot=4, title = "Total simulation time", grid = false, showaxis = false, bottom_margin = -50Plots.px)
plot!(fig2; subplot=7, title = "Efficiency", grid = false, showaxis = false, bottom_margin = -50Plots.px)


plot!(fig2; subplot=2, ylabel="μs", xlabel="Generation", xtick=([1,2,3,4,5,6,7,8], gen_labels))
boxplot!(fig2, repeat(1:8, K).-0.05, reshape(1000 .* T_per_N_abc, 8*K, 1); subplot=2, label="ABC-SMC")
boxplot!(fig2, repeat(1:8, K).+0.05, reshape(1000 .* T_per_N_mfabc, 8*K, 1); subplot=2, label="MF-ABC-SMC")
# save("./out/pdf/time_per_proposal_per_gen.pdf", fig2a)

plot!(fig2; subplot=5, legend=:none, ylabel="min", xlabel="Generation", xtick=([1,2,3,4,5,6,7,8], gen_labels)),
boxplot!(fig2, repeat(1:8, K).-0.05, reshape(T_abc./60, 8*K, 1); subplot=5, label="ABC-SMC")
boxplot!(fig2, repeat(1:8, K).+0.05, reshape(T_mfabc./60, 8*K, 1); subplot=5, label="MF-ABC-SMC")
# save("./out/pdf/length_per_gen.pdf", fig2b)

plot!(fig2; subplot=8, legend=:none, ylabel="ESS / s", xlabel="Generation", xtick=([1,2,3,4,5,6,7,8], gen_labels))
boxplot!(fig2, repeat(1:8, 50).-0.05, reshape(eff_abc,8*50,1); subplot=8, label="ABC-SMC")
boxplot!(fig2, repeat(1:8, 50).+0.05, reshape(eff_mfabc,8*50,1); subplot=8, label="MF-ABC-SMC")
# save("./out/pdf/eff_per_gen.pdf", fig2c)

boxplot!(fig2, fill(1,K), (sum(1000 .*T_abc, dims=1)./sum(N_abc, dims=1))'; subplot=3)
boxplot!(fig2, fill(2,K), (sum(1000 .*T_mfabc, dims=1)./sum(N_mfabc, dims=1))'; subplot=3)
plot!(fig2; subplot=3, ylabel="μs", xlabel="Algorithm", xtick=([1,2], ["ABC-SMC" "MF-ABC-SMC"]), legend=:none)

boxplot!(fig2, fill(1,K), sum(T_abc./60, dims=1)'; subplot=6)
boxplot!(fig2, fill(2,K), sum(T_mfabc./60, dims=1)'; subplot=6)
plot!(fig2; subplot=6, ylabel="min", xlabel="Algorithm", xtick=([1,2], ["ABC-SMC" "MF-ABC-SMC"]), legend=:none)

boxplot!(fig2, fill(1,K), ESS_abc[end,:]./(sum(T_abc, dims=1))'; subplot=9)
boxplot!(fig2, fill(2,K), ESS_mfabc[end,:]./(sum(T_mfabc, dims=1))'; subplot=9)
plot!(fig2; subplot=9, ylabel="ESS / s", xlabel="Algorithm", xtick=([1,2], ["ABC-SMC" "MF-ABC-SMC"]), legend=:none)

save("./out/pdf/generations.pdf", fig2)



# Estimates and simulation times
fig3a = plot(;
xlabel=L"K",
ylabel="Simulation time (min)",
legend=:none,
)
fig3b = plot(;
xlabel=L"\omega_0",
ylabel="",
legend=:none,
title="Empirical posterior parameter means"
)
fig3c = plot(;
xlabel=L"\gamma",
ylabel="",
legend=:right,
)

T_tot_abc = sum(T_abc, dims=1)'
T_tot_mfabc = sum(T_mfabc, dims=1)'

scatter!(fig3a, F_abc[1,:], T_tot_abc./60)
scatter!(fig3a, F_mfabc[1,:], T_tot_mfabc./60)
scatter!(fig3b, F_abc[2,:], T_tot_abc./60)
scatter!(fig3b, F_mfabc[2,:], T_tot_mfabc./60)
scatter!(fig3c, F_abc[3,:], T_tot_abc./60, label="ABC-SMC")
scatter!(fig3c, F_mfabc[3,:], T_tot_mfabc./60, label="MF-ABC-SMC")

μ = mean(hcat(F_abc, F_mfabc), dims=2)
σ = std(hcat(F_abc,F_mfabc), dims=2)

mtick = broadcast((μ_i, d) -> round(μ_i, digits=d), μ, (2,3,3))
gap = broadcast((σ_i, d) -> round(σ_i, digits=d), σ, (3,4,4))

llim  = mtick .- (3 .* gap)
ltick = mtick .- (2 .* gap)
utick = mtick .+ (2 .* gap)
ulim  = mtick .+ (3 .* gap)

Tticks = round.([mean(T_tot_abc), mean(T_tot_mfabc)]./60, digits=1)

figvec = [fig3a, fig3b, fig3c]

for i in 1:3
plot!(
       figvec[i];
       xticks = [ltick[i], mtick[i], utick[i]],
       xlim = [llim[i], ulim[i]],
       yticks = Tticks,
       ylim = [0, Inf],
)
end

using Plots.PlotMeasures
fig3abc = plot(fig3a,fig3b,fig3c;
layout=(1,3),
size=(1000,300),
margin=25px,
)
save("./out/pdf/estimates.pdf", fig3abc)

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
save("./out/pdf/etas.pdf", fig5)
