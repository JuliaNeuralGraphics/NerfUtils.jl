module NerfUtils

export GridEncoding, IG
export Dense, Chain, Adam
export relu, softplus, sigmoid
export CameraIntrinsics, Camera

using Adapt
using ChainRulesCore
using KernelAbstractions
using KernelAbstractions: @atomic
using Random
using StaticArrays

import KernelAbstractions as KA

const IG = Val{:IG}()
const Maybe{T} = Union{T, Nothing}

include("colmap.jl")
include("encoding/grid_encoding.jl")
include("nn/nn.jl")
include("nn/adam.jl")

include("render/intrinsics.jl")
include("render/camera.jl")

end
