module NerfUtils

export GridEncoding, IG

using Adapt
using ChainRulesCore
using KernelAbstractions
using KernelAbstractions: @atomic
using StaticArrays

const IG = Val{:IG}()

import KernelAbstractions as KA

include("encoding/grid_encoding.jl")

end
