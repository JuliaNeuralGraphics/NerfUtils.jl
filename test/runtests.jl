using Adapt
using Pkg
using Test
using NerfUtils
using Zygote

import KernelAbstractions as KA

function parse_flags!(args, flag; default = nothing, typ = typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing && typ <: AbstractString)
                val = parse(typ, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

_, BACKEND_NAME = parse_flags!(ARGS, "--backend"; default="AMDGPU", typ=String)
@info "Backend name: `$BACKEND_NAME`."

@static if BACKEND_NAME == "AMDGPU"
    Pkg.add("AMDGPU")
    using AMDGPU
    const BACKEND = ROCBackend()
    const AT = ROCArray
elseif BACKEND_NAME == "CUDA"
    Pkg.add("CUDA")
    using CUDA
    const BACKEND = CUDABackend()
    const AT = CuArray
else
    error("Invalid backend name: `$BACKEND_NAME`.")
end
@info "Running tests on backend: `$BACKEND`."

@testset "Grid Encoding" begin
    n = 3
    @testset "Input dim: $n_dims, N features: $n_features, N levels: $n_levels" for n_dims in 1:3, n_features in 1:4, n_levels in (1, 4, 16)
        ge = GridEncoding(BACKEND; n_dims, n_features, n_levels)
        θ = NerfUtils.init(ge)

        x = adapt(BACKEND, rand(Float32, (n_dims, n)))
        y = ge(x, θ)
        @test y isa AT

        target_size = (prod(NerfUtils.output_size(ge)), n)
        @test size(y) == target_size 

        ∇ = Zygote.gradient(θ) do θ
            sum(ge(x, θ))
        end
        @test length(∇) == 1
        @test size(∇[1]) == size(θ)

        ∇ = Zygote.gradient(x) do x
            sum(ge(x, θ, IG))
        end
        @test length(∇) == 1
        @test size(∇[1]) == (n_dims, n)

        ∇ = Zygote.gradient(θ, x) do θ, x
            sum(ge(x, θ, IG))
        end
        @test length(∇) == 2
        @test size(∇[1]) == size(θ)
        @test size(∇[2]) == (n_dims, n)
    end

    @testset "Invalid input arguments" begin
        @test_throws ArgumentError GridEncoding(BACKEND; n_dims=4)
        @test_throws ArgumentError GridEncoding(BACKEND; n_levels=34)

        ge = GridEncoding(BACKEND; n_dims=1)
        θ = NerfUtils.init(ge)
        x = adapt(BACKEND, rand(Float32, (2, n)))
        @test_throws ArgumentError ge(x, θ)
    end
end
