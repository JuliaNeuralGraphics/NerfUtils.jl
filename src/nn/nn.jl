struct Dense{T, F}
    activation::F
    in_channels::Int64
    out_channels::Int64

    function Dense{T}(
        mapping::Pair{Int64, Int64}, activation::F = identity,
    ) where {T <: Union{Float16, Float32}, F}
        new{T, F}(activation, first(mapping), last(mapping))
    end

    function Dense(
        mapping::Pair{Int64, Int64}, activation::F = identity,
    ) where F
        new{Float32, F}(activation, first(mapping), last(mapping))
    end
end

function (d::Dense{T, typeof(identity)})(x, θ) where T
    θ * x
end

function (d::Dense{T, F})(x, θ) where {T, F}
    d.activation.(θ * x)
end

precision(::Dense{T, F}) where {T, F} = T

function init(d::Dense, kab)
    glorot_uniform(kab, precision(d), (d.out_channels, d.in_channels))
end

reset!(::Dense, θ) = glorot_uniform!(θ)

struct Chain{L}
    layers::L
    Chain(layers...) = new{typeof(layers)}(layers)
end

function init(c::Chain, kab)
    recursive_init((), first(c.layers), Base.tail(c.layers), kab)
end

function recursive_init(θ, l, c::Tuple, kab)
    recursive_init((θ..., init(l, kab)), first(c), Base.tail(c), kab)
end
function recursive_init(θ, l, ::Tuple{}, kab)
    (θ..., init(l, kab))
end

function reset!(c::Chain, θ)
    foreach(l -> reset!(l[1], l[2]), zip(c.layers, θ))
end

function (c::Chain)(x, θ)
    recursive_apply(
        x, first(c.layers), Base.tail(c.layers), first(θ), Base.tail(θ))
end

function recursive_apply(x, l, c::Tuple, θₗ, θ)
    recursive_apply(l(x, θₗ), first(c), Base.tail(c), first(θ), Base.tail(θ))
end
function recursive_apply(x, l, ::Tuple{}, θₗ, ::Tuple{})
    l(x, θₗ)
end

function glorot_uniform(kab, ::Type{T}, dims; gain::T = one(T)) where T
    x = KA.allocate(kab, T, dims)
    glorot_uniform!(x; gain)
end

function glorot_uniform!(x::AbstractArray{T}; gain::T = one(T)) where T
    scale::T = gain * √(24f0 / sum(size(x)))
    rand!(x)
    x .-= T(0.5f0)
    x .*= scale
    return x
end

function relu(x::T) where T
    ifelse(x < zero(T), zero(T), x)
end

function softplus(x::T) where T
    log1p(exp(-abs(x))) + relu(x)
end

function sigmoid(x::T) where T
    t = exp(-abs(x))
    ifelse(x ≥ zero(T), inv(one(T) + t), t / (one(T) + t))
end
