include("grid_utils.jl")
include("grid_kernels.jl")

struct GridEncoding{O, R, L}
    offset_table::O
    resolutions::R
    level_ids::L
    n_dims::UInt32
    n_features::UInt32
    n_levels::UInt32
    n_params::UInt32
    base_resolution::UInt32
    scale::Float32
    align_corners::Bool
end

function GridEncoding(
    kab; n_dims::Int = 3, n_levels::Int = 16, scale::Float32 = 1.5f0,
    base_resolution::Int = 16, n_features::Int = 2, hashmap_size::Int = 19,
    align_corners::Bool = true, store_level_ids::Bool = false,
)
    n_levels < 34 || throw(ArgumentError(
        "`n_levels` must be < `34`, instead: `$n_levels`."))
    (1 ≤ n_dims ≤ 3) || throw(ArgumentError(
        "`n_dims` must be in `[1, 3]` range, instead: `$n_dims`."))

    max_params = typemax(UInt32) ÷ 0x2
    hashmap_params = one(UInt32) << hashmap_size

    offset_table = Vector{UInt32}(undef, n_levels + 1)
    resolutions = Vector{Int32}(undef, n_levels)

    offset = zero(UInt32)
    log_scale = log2(scale)
    for level in 1:n_levels
        level_scale::Float32 = compute_scale(
            UInt32(level), log_scale, UInt32(base_resolution))
        resolution = ceil(level_scale) + (align_corners ? 0 : 1)
        resolutions[level] = resolution

        level_params::UInt32 = min(resolution^n_dims, max_params)
        level_params = next_multiple(level_params, 0x8) # align to 8
        level_params = min(level_params, hashmap_params) # maybe clamp

        offset_table[level] = offset
        offset += level_params
    end
    offset_table[end] = offset

    if store_level_ids
        level_ids = Vector{Int8}(undef, offset)
        for l in 1:(n_levels - 1)
            level_ids[(1 + offset_table[l]):offset_table[l + 1]] .= l
        end
    else
        level_ids = nothing
    end

    n_params = offset * n_features
    GridEncoding(
        adapt(kab, offset_table), adapt(kab, resolutions), adapt(kab, level_ids),
        UInt32(n_dims), UInt32(n_features),
        UInt32(n_levels), UInt32(n_params),
        UInt32(base_resolution), scale, align_corners)
end

KernelAbstractions.get_backend(ge::GridEncoding) = get_backend(ge.offset_table)

function _kernel_params(ge::GridEncoding)
    NPD = Val{Int64(ge.n_dims)}()
    NFPL = Val{Int64(ge.n_features)}()
    ALG = Val(ge.align_corners)
    NPD, NFPL, ALG
end

function init(ge::GridEncoding)
    shape = Int64.((ge.n_features, ge.n_params ÷ ge.n_features))
    adapt(get_backend(ge), rand(Float32, shape) .* 2f-4 .- 1f-4)
end

function reset!(::GridEncoding, θ)
    copy!(θ, rand(Float32, size(θ)) .* 2f-4 .- 1f-4)
end

function output_size(ge::GridEncoding)
    Int.((ge.n_features, ge.n_levels))
end

function (ge::GridEncoding)(x::T, θ) where T <: AbstractMatrix
    n_dims, n = size(x)
    n_dims != ge.n_dims && throw(ArgumentError(
        "`x` must be of `($(Int(ge.n_dims)), N)` size, instead: `$size(x)`."))

    kab = get_backend(ge)
    y = allocate(kab, Float32, (output_size(ge)..., n))
    grid_kernel!(kab)(
        y, nothing, x, θ, ge.offset_table, _kernel_params(ge)...,
        ge.base_resolution, log2(ge.scale); ndrange=(n, ge.n_levels))
    reshape(y, :, n)
end

function (ge::GridEncoding)(x::T, θ, ::Val{:IG}) where T <: AbstractMatrix
    n_dims, n = size(x)
    n_dims != ge.n_dims && throw(ArgumentError(
        "`x` must be of `($(Int(ge.n_dims)), N)` size, instead: `$size(x)`."))

    kab = get_backend(ge)
    y = allocate(kab, Float32, (output_size(ge)..., n))
    ∂y∂x_shape = Int64.((ge.n_dims, output_size(ge)..., n))
    ∂y∂x = KA.zeros(kab, Float32, ∂y∂x_shape)

    grid_kernel!(kab)(
        y, ∂y∂x, x, θ, ge.offset_table,
        _kernel_params(ge)..., ge.base_resolution,
        log2(ge.scale); ndrange=(n, ge.n_levels))
    reshape(y, :, n), ∂y∂x
end

function ∇(ge::GridEncoding, ∂f∂y, x::T, θ) where T <: AbstractMatrix
    kab = get_backend(ge)
    n = size(x, 2)
    ∂grid = KA.zeros(kab, Float32, size(θ))
    ∇grid_kernel!(kab)(
        ∂grid, ∂f∂y, x, ge.offset_table, _kernel_params(ge)...,
        ge.base_resolution, log2(ge.scale); ndrange=(n, ge.n_levels))
    ∂grid
end

function ∇grid_input(ge::GridEncoding, ∂L∂y, ∂y∂x)
    kab = get_backend(ge)
    n, L = size(∂y∂x, 4), Val{ge.n_levels}()
    ∂L∂x = allocate(kab, Float32, Int64.((ge.n_dims, n)))
    ∇grid_kernel_input!(kab)(
        ∂L∂x, ∂L∂y, ∂y∂x, _kernel_params(ge)[1:2]..., L; ndrange=n)
    ∂L∂x
end

function ChainRulesCore.rrule(ge::GridEncoding, x, θ)
    n = size(x, 2)
    function encode_pullback(Δ)
        Δ2 = reshape(unthunk(Δ), (output_size(ge)..., n))
        Tangent{GridEncoding}(), NoTangent(), ∇(ge, Δ2, x, θ)
    end
    ge(x, θ), encode_pullback
end

function ChainRulesCore.rrule(ge::GridEncoding, x, θ, ::Val{:IG})
    n = size(x, 2)
    y, ∂y∂x = ge(x, θ, Val{:IG}())
    function encode_pullback(Δ)
        Δ2 = reshape(unthunk(Δ), (output_size(ge)..., n))
        (
            Tangent{GridEncoding}(), @thunk(∇grid_input(ge, Δ2, ∂y∂x)),
            @thunk(∇(ge, Δ2, x, θ)), NoTangent())
    end
    y, encode_pullback
end
