# MLP components

## Layers

[`Dense`](@ref) layer can be used to construct MLP.
Its parameters are handled explicitly, meaning after instantiation
`d = Dense(1=>2)`, parameters are obtained with `θ = NerfUtils.init(d, Backend)`.
Where you pass one of the supported [Backends](@ref) to initialize the model
directly on the device.

You can then pass inputs to the layer `y = d(x, θ)`.

[`Chain`](@ref) can be used to stack multiple dense layers.
Its parameters are handled in the same manner:
```julia
c = Chain(Dense(1=>2), Dense(2=>1))
θ = NerfUtils.init(c, Backend)

x = ...
y = c(x, θ)
```

```@docs
Dense
NerfUtils.init(::Dense, Backend)
NerfUtils.reset!(::Dense, θ)
Chain
NerfUtils.init(::Chain, Backend)
NerfUtils.reset!(::Chain, θ)
```

## Optimizer

There's only one optimizer available - [`Adam`](@ref).
It handles parameters that are either tuples (or named tuples) or plain arrays.

```julia
c = Chain(Dense(1 => 2), Dense(2 => 1))
θ = NerfUtils.init(c, Backend)
opt = Adam(Backend, θ; lr=1f-3)
```

After computing gradients `∇`, applying the update rule can be done with
`NerfUtils.step!` function.

```julia
NerfUtils.step!(opt, θ, ∇; dispose=true) # Free immediately gradient memory afterwards.
NerfUtils.step!(opt, θ, ∇; dispose=false) # Do not free.
```

```@docs
Adam
NerfUtils.reset!(::Adam)
NerfUtils.step!(::Adam, θ, ∇; dispose::Bool)
```
