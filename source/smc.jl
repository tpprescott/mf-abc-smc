export SMCProblem

struct SMCProblem{S<:System,T<:AbstractParameters{S},K}
    prior::ParameterSampler{T}
    data::Tuple{AbstractSummaryStat{S,LoFi},AbstractSummaryStat{S,HiFi}}
    models::Tuple{AbstractModel{S,LoFi},AbstractModel{S,HiFi}}
    epsilons::Tuple{SVector{K,Float64},SVector{K,Float64}}

    function SMCProblem(
        prior::ParameterSampler{T},
        data::Tuple{AbstractSummaryStat{S,LoFi},AbstractSummaryStat{S,HiFi}},
        models::Tuple{AbstractModel{S,LoFi},AbstractModel{S,HiFi}},
        epsilons::Vector{Float64},
    ) where {T<:AbstractParameters{S}} where S

        K = length(epsilons)
        return new{S,T,K}(
            prior,
            data,
            models,
            (SVector{K}(epsilons), SVector{K}(epsilons)),
        )
    end
    function SMCProblem(
        prior::ParameterSampler{T},
        data::Tuple{AbstractSummaryStat{S,LoFi},AbstractSummaryStat{S,HiFi}},
        models::Tuple{AbstractModel{S,LoFi},AbstractModel{S,HiFi}},
        epsilons::Vector{Float64},
        epsilons_tilde::Vector{Float64},
    ) where {T<:AbstractParameters{S}} where S

        K = length(epsilons)
        return new{S,T,K}(
            prior,
            data,
            models,
            (SVector{K}(epsilons_tilde), SVector{K}(epsilons)),
        )
    end
end

function epsilon_i(
    epsilons::Tuple{SVector{K,Float64},SVector{K,Float64}},
    i::Int64,
)::Tuple{Float64, Float64} where {K}
    if i > K
        error("Asking for the wrong pair of epsilons")
    else
        return (epsilons[1][i], epsilons[2][i])
    end
end

function Cloud(
    smc::SMCProblem{S,T,K},
    counter::Function,
    stop_condition::X;
    kwargs...
) where S where T where K where X<:Number
    return Cloud(smc, counter, fill(stop_condition, K); kwargs...)
end

function Cloud(
    smc::SMCProblem{S,T,K},
    counter::Function,
    stop_condition::Array{X,1};
    eta_min::Float64 = 0.01,
    batch_size::Int64 = 100,
) where {S} where {T} where {K} where X<:Number

    length(stop_condition)==K || error("Stop conditions don't match generations one-to-one.")

    C = Vector{Cloud{S,T}}(undef, K)
    mfabc = MFABCProblem(
        smc.prior,
        smc.prior,
        smc.models,
        smc.data,
        epsilon_i(smc.epsilons, 1),
    )
    C[1] = Cloud(mfabc, counter, stop_condition[1]; batch_size=batch_size)
    for k = 2:K
        q = Importance(C[k-1])
        e = epsilon_i(smc.epsilons, k)
        mfabc = MFABCProblem(
            smc.prior,
            q,
            smc.models,
            smc.data,
            e,
            Eta(C[1:k-1], q, e, H = EtaSpace(max(eps(),eta_min))),
        )
        C[k] = Cloud(mfabc, counter, stop_condition[k]; batch_size=batch_size)
    end
    return C
end

function efficiency(Cvec::Vector{Cloud{S,T}}) where S where T
    return ESS(Cvec[end])/sum(gettime.(Cvec))
end
