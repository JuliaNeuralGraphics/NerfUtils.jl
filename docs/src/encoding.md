# Encodings

## Multiresolution Hash Encoding

[`GridEncoding`](@ref) implements Multiresolution Hash Encoding
which can be used to encode 3D points prior to MLP.

Instantiate `ge = GridEncoding(Backend)` with a supported backend.
`GridEncoding` handles its parameters explicitly, meaning it is up to the
user to initialize and maintain them with `θ = NerfUtils.init(ge)`.

To encode input coordinates `x` pass them along with the parameters `θ`:
```julia
# Random 3D points on the same `Backend`.
x = adapt(Backend, rand(Float32, 3, N))
y = ge(x, θ)
```

**Note**, that inputs must be in `[0, 1]` range, otherwise the kernel might break.
`GridEncoding` does not check input values.

### Computing gradients

`GridEncoding` defines respective chain rules for its kernels using
[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) package,
meaning it supports any AD that works with them.
Following examples use [Zygote.jl](https://github.com/FluxML/Zygote.jl) for that.

Compute gradients w.r.t. `θ`:
```julia
∇ = Zygote.gradient(θ) do θ # Passing θ explicitly to the `gradient` function.
    sum(ge(x, θ))
end
```

To compute gradients w.r.t. `θ` and input `x` you have to pass additional
input argument `IG`:
```julia
∇ = Zygote.gradient(x, θ) do x, θ
    sum(ge(x, θ, IG))
end
∇[1] # Gradient w.r.t. x.
∇[2] # Gradient w.r.t. θ.
```

It is done this way to dispatch to a different kernel that in the forward
pass precomputes necessary values for the backward pass to speed things up.

See [`GridEncoding`](@ref) docs for the description of each argument
during instantiation as you might want to configure number of levels,
their resolution and feature dimensions.

```@docs
GridEncoding
```
