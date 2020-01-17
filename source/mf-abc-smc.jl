using Distributions
#import Distributions.rand
#import Distributions.pdf
#import Base.length

module MFABCSMC
include("types.jl")
include("parameters.jl")
include("abc.jl")
include("mf.jl")
include("clouds.jl")
include("eta.jl")
include("smc.jl")
include("density.jl")
end

using .MFABCSMC

include("examples\kuramoto.jl")
