# NerfUtils.jl

Reusable NeRF components

## Test

To test on a specific backend, pass `--backend` argument:
```julia
julia> using Pkg; Pkg.test("NerfUtils"; test_args=["--backend=AMDGPU"])
```
