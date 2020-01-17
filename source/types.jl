#=
DEFINE SYSTEM AND FIDELITY AND A MODEL OF A PARTICULAR SYSTEM AT A PARTICULAR FIDELITY
DEFINE DATA FOR THE SYSTEM
DEFINE THE OUTPUT OF A MODEL SIMULATION
=#
#
export System, AbstractModel, AbstractParameters, ParameterSampler

# MODELS ARE DEFINED FOR A SYSTEM AND HAVE A LABEL
abstract type System end
abstract type Label end
abstract type AbstractModel{S<:System,L<:Label} end

# Assuming all models for a given system have the same parameter space
abstract type AbstractParameters{S<:System} end
abstract type ParameterSampler{T<:AbstractParameters} end

export AbstractSummaryStat, AbstractNoise, IndependentSimulation, ModelOutput

abstract type AbstractSummaryStat{S<:System,L<:Label} end
abstract type AbstractNoise{S<:System,L<:Label} end
struct IndependentSimulation{S<:System,L<:Label} <: AbstractNoise{S,L} end # No noise input into the simulation: runs independently
convert(::Type{IndependentSimulation{S,L}}, x::IndependentSimulation{S}) where S where L = IndependentSimulation{S,L}()



struct ModelOutput{S<:System,L<:Label}
    y::AbstractSummaryStat{S,L}
    u::AbstractNoise{S,L}
end
