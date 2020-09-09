using StatsPlots
using LaTeXStrings
using JLD
using Plots.Measures
export plot_densities, volume

volume(xlim::Tuple{Float64,Float64}) = xlim[2]-xlim[1]
volume(xlim::Tuple{Float64,Float64}...) = prod(volume.(xlim))
volume(xlim::NTuple{N, Tuple{Float64,Float64}}) where N = volume(xlim...)

function _in(p::P, xlim::Tuple{Float64,Float64}, dim::Int64)::Bool where P<:RealParameter
    return xlim[1]<p.k[dim]<xlim[2]
end
function _in(p::P, xlim::NTuple{N,Tuple{Float64,Float64}}, dim::NTuple{N,Int64}) where P<:RealParameter where N
    return all([_in(p, xlim_i, dim_i) for (xlim_i, dim_i) in zip(xlim, dim)])
end
function _in(p::Particle{S,P}, xlim, dim)::Bool where S where P<:RealParameter
    return _in(p.theta, xlim, dim)
end

function _in(theta_i::AbstractArray{Float64,1}, xlim::Tuple{Float64,Float64}, dim::Int64)::Bool
    return xlim[1]<theta_i[dim]<xlim[2]
end
function _in(theta_i::AbstractArray{Float64,1}, xlim::NTuple{N,Tuple{Float64,Float64}}, dim::NTuple{N,Int64}) where N
    return all([_in(theta_i, xlim_j, dim_j) for (xlim_j, dim_j) in zip(xlim, dim)])
end


function density(theta::AbstractArray{Float64,2}, w::AbstractArray{Float64,1}, xlim, dim) where S where T
    d = 0.0
    for (i, w_i) in enumerate(w)
        iszero(w_i) || (_in(theta[:,i], xlim, dim) && (d += w_i))
    end
    d /= sum(w)*volume(xlim)
    return d
end

function get_marginal(theta::AbstractArray{Float64,2}, w::AbstractArray{Float64,1}, dim::Int64, dx::Float64) where S where T
    function f(x::Float64)
        xlim = (x-dx/2, x+dx/2)
        return density(theta, w, xlim, dim)
    end
    return f
end
function get_marginal(theta::AbstractArray{Float64,2}, w::AbstractArray{Float64,1}, dims::Tuple{Int64,Int64}, dxdy::Tuple{Float64,Float64}) where S where T
    function f(x::Float64, y::Float64)
        xlim = (x-dxdy[1]/2, x+dxdy[1]/2)
        ylim = (y-dxdy[2]/2, y+dxdy[2]/2)
        return density(theta, w, (xlim,ylim), dims)
    end
    return f
end

tri(N::Int64)::Int64 = N*(N+1)//2

function plot_densities(
    C::Cloud{S,T},
    axislims::NTuple{N,Tuple{Float64,Float64}},
    resolution::Int64=Int64(round(sqrt(ESS(C))));
    param_labels::Vector{Str}=[latexstring("\\theta_$i") for i in 1:N],
    fn::AbstractString = "figdata",
    ) where T<:RealParameter{S,N} where S where N where Str<:AbstractString

    ϵ = getepsilon(C)[2]
    theta = hcat([Vector{Float64}(p.theta.k) for p in C if !iszero(p.w)]...)
    w = getweight.(C)
    filter!(!iszero, w)

    save("./data/"*fn*".jld",
    "theta", theta,
    "w", w,
    "ϵ", ϵ,
    "axislims", axislims,
    "param_labels", param_labels,
    )
    return plot_densities(theta, w, ϵ, axislims, resolution, param_labels=param_labels)
end

function plot_densities(fn::AbstractString, resolution::Int64...)
    data = load("./data/"*fn*".jld")
    theta = data["theta"]
    w = data["w"]
    ϵ = data["ϵ"]
    axislims = data["axislims"]
    param_labels = data["param_labels"]

    return plot_densities(theta, w, ϵ, axislims, resolution...; param_labels=param_labels)
end

function plot_densities(
    theta::AbstractArray{Float64,2},
    w::AbstractArray{Float64,1},
    ϵ::Float64,
    axislims::NTuple{N,Tuple{Float64,Float64}},
    resolution::Int64=Int64(round(sqrt(2.0*ESS(w))));
    param_labels::Vector{Str}=[latexstring("\\theta_$i") for i in 1:N],
    ) where T<:RealParameter{S,N} where S where N where Str<:AbstractString

    smoothing = volume.(axislims)./resolution
    axis_range = [axislims_i[1]:daxis_i:axislims_i[2] for (axislims_i, daxis_i) in zip(axislims, smoothing)]
    bigW = maximum(abs.(w))

    F = plot(layout = grid(N,N),
        xlabel="Placeholder",
        ylabel="Placeholder",
        grid=:none,
        margin=2mm,
        )

    for j in 1:N
        for i in 1:j-1
            plot!(F,
                theta[i,:],
                theta[j,:],
                marker_z = w./bigW,
#                markercolor = :pu_or_r,
                seriestype = :scatter,
                markercolor = cgrad(:balance,rev=true),
                markershape = :xcross,
                markerstrokewidth = 0,
                markersize = 1.5,
                xlim = axislims[i] .+ (0.5.*(-smoothing[i], smoothing[i])),
                ylim = axislims[j] .+ (0.5.*(-smoothing[j], smoothing[j])),
                xlabel=param_labels[i],
                ylabel=param_labels[j],
                legend=:none,
                clims=(-1,1),
                colorbar=:right,
                colorbar_title=L"w(\theta)",
                subplot = i+(j-1)*N,
                )
        end
        plot!(F,
            axis_range[j],
            get_marginal(theta, w, j, smoothing[j]);
            legend=:none,
            yticks=[],
            xlabel=param_labels[j],
            xlim = axislims[j] .+ (0.5.*(-smoothing[j], smoothing[j])),
            ylabel="",
            color=:black,
            subplot = j+(j-1)*N,
            )
        for i in j+1:N
            xx = axis_range[i]
            yy = axis_range[j]
            f = get_marginal(theta, w, (i,j), (smoothing[i], smoothing[j]))
            ff(xy) = abs(f(xy...))
            bigZ = maximum(ff, Iterators.product(xx,yy))
            plot!(F,
                xx,
                yy,
                f;
#                fillcolor=:pu_or_r,
                seriestype=:heatmap,
                seriescolor=cgrad(:balance, rev=true),
                legend=:none,
                colorbar=:none,
                clims=(-bigZ, bigZ),
                xlabel=param_labels[i],
                ylabel=param_labels[j],
                xtick_direction=:out,
                ytick_direction=:out,
                subplot = i + (j-1)*N,
                )
        end
    end

    l = @layout [a{.05h}; b]
    full_fig = plot(
        plot(annotation=(0.5,0.5, latexstring("\\epsilon = $(round(ϵ, digits=3)) \\quad{} \\mathrm{ESS} = $(round(ESS(w), digits=1))")), framestyle=:none),
        F,
        size = (900,500),
        layout = l,
        )
    return full_fig

end
