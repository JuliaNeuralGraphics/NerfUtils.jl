module NerfUtils

export GridEncoding, IG
export Dense, Chain, Adam
export relu, softplus, sigmoid

using Adapt
using ChainRulesCore
using KernelAbstractions
using KernelAbstractions: @atomic
using Random
using StaticArrays

const IG = Val{:IG}()

import KernelAbstractions as KA

include("encoding/grid_encoding.jl")
include("nn/nn.jl")
include("nn/adam.jl")

end
