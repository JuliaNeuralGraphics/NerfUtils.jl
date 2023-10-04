"""
    Adam(kab, θ; kwargs...)

Adam optimizer.

`θ` must be either plain array or a tuple (or named tuple).
"""
Base.@kwdef mutable struct Adam{T}
    μ::Vector{T}
    ν::Vector{T}
    current_step::UInt32 = UInt32(0)

    # Hyperparameters.
    lr::Float32 = 1f-2
    β1::Float32 = 0.9f0
    β2::Float32 = 0.999f0
    ϵ::Float32 = 1f-8
end

KernelAbstractions.get_backend(opt::Adam) = get_backend(first(opt.μ))

function Adam(kab, θ; kwargs...)
    μ, ν = [], [] # FIXME unstable
    _add_moments!(μ, ν, θ, kab)
    Adam(; μ, ν, kwargs...)
end

function _add_moments!(μ, ν, θ::T, kab) where T <: Union{Tuple, NamedTuple}
    foreach(θᵢ -> _add_moments!(μ, ν, θᵢ, kab), θ)
end

function _add_moments!(μ, ν, θ, kab)
    push!(μ, KA.zeros(kab, Float32, length(θ)))
    push!(ν, KA.zeros(kab, Float32, length(θ)))
end

"""
    reset!(opt::Adam)

Reset the optimizer state.
"""
function reset!(opt::Adam)
    fill!.(opt.μ, 0f0)
    fill!.(opt.ν, 0f0)
    opt.current_step = 0x0
end

"""
    step!(opt::Adam, θ, ∇; dispose::Bool)

Apply update rule to parameters `θ` with gradients `∇`.

# Arguments:

- `dispose::Bool`: Free memory taken by gradients `∇` after update.
"""
function step!(opt::Adam, θ, ∇; dispose::Bool)
    length(θ) == length(∇) || throw(ArgumentError(
        "Number of parameters must be the same as the number of gradients, " *
        "instead: `$(length(θ))` vs `$(length(∇))`."))

    opt.current_step += 0x1
    _step!(opt, θ, ∇, 1; dispose)
    return
end

function _step!(opt::Adam, θ::T, ∇::G, i; dispose::Bool) where {
    T <: Union{Tuple, NamedTuple}, G <: Union{Tuple, NamedTuple},
}
    for (θᵢ, ∇ᵢ) in zip(θ, ∇)
        i = _step!(opt, θᵢ, ∇ᵢ, i; dispose)
    end
    return i
end

function _step!(opt::Adam, θ::T, ∇::T, i; dispose::Bool) where T <: AbstractArray
    # TODO add check flag
    # @assert !any(isnan.(θ)) "NaN parameters of size $(size(θ))"
    # @assert !any(isnan.(∇)) "NaN parameters of size $(size(∇))"

    size(θ) == size(∇) || throw(ArgumentError(
        "Shape of parameters and gradients must be the same, " *
        "instead: `$(size(θ))` vs `$(size(∇))`."))

    # Debiasing.
    lr = opt.lr * √(1f0 - opt.β2^opt.current_step) / (1f0 - opt.β1^opt.current_step)
    adam_step_kernel!(get_backend(opt))(
        opt.μ[i], opt.ν[i], θ, ∇,
        opt.lr, opt.β1, opt.β2, opt.ϵ; ndrange=length(θ))

    dispose && KA.unsafe_free!(∇)

    return i + 1
end

@kernel function adam_step_kernel!(
    μ, ν, Θ, @Const(∇), lr::Float32,
    β1::Float32, β2::Float32, ϵ::Float32,
)
    i = @index(Global)
    @inbounds ∇ᵢ = ∇[i]
    @inbounds ωᵢ = Θ[i]

    ∇ᵢ² = ∇ᵢ * ∇ᵢ
    @inbounds μᵢ = μ[i] = β1 * μ[i] + (1f0 - β1) * ∇ᵢ
    @inbounds νᵢ = ν[i] = β2 * ν[i] + (1f0 - β2) * ∇ᵢ²

    @inbounds Θ[i] = ωᵢ - (lr * μᵢ) / (√νᵢ + ϵ)
end

