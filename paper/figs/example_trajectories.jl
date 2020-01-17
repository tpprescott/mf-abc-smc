@everywhere include("../../source/mf-abc-smc.jl")

using .KuramotoOscillators
using Plots
using JLD
using LaTeXStrings

example_lo_traj() = KuramotoOscillators._get_trajectory(m_lo, KuramotoOscillators.theta)
example_hi_traj() = KuramotoOscillators._get_trajectory(m_hi, KuramotoOscillators.theta)

function example_kuramoto_dynamics(N::Integer=1)
    Rfig = plot(; xlabel="Time", ylabel=L"R(t)",
    title="Kuramoto parameter: magnitude", titlefontsize=10)
    Φfig = plot(; xlabel="Time", ylabel=L"\Phi(t)",
    title="Kuramoto parameter: phase", titlefontsize=10)

    for n in 1:N-1
        traj_hi = example_hi_traj()
        plot!(Rfig, traj_hi.t, traj_hi.R; linealpha=0.5, label="")
        plot!(Φfig, traj_hi.t, traj_hi.ϕ; linealpha=0.5, label="")
    end
    traj_hi = example_hi_traj()
    plot!(Rfig, traj_hi.t, traj_hi.R; linealpha=0.5, label="")
    plot!(Φfig, traj_hi.t, traj_hi.ϕ; linealpha=0.5, label="")

    traj_lo = example_lo_traj()
    plot!(Rfig, traj_lo.t, traj_lo.R; linewidth=2, linecolor=:black, label="Low fidelity")
    plot!(Φfig, traj_lo.t, traj_lo.ϕ; linewidth=2, linecolor=:black, label="Low fidelity")

    plot!(Rfig; size=(300,200))
    plot!(Φfig; size=(300,200), legend=:none)

    save("./out/pdf/exampletraj_R.pdf", Rfig)
    save("./out/pdf/exampletraj_Phi.pdf", Φfig)
    return nothing
end
