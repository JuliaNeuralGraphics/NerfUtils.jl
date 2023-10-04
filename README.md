# NerfUtils.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianeuralgraphics.github.io/NerfUtils.jl/dev)

Reusable NeRF components

## Test

To test on a specific backend, pass `--backend` argument:
```julia
julia> using Pkg; Pkg.test("NerfUtils"; test_args=["--backend=AMDGPU"])
```

## Projects that use NerfUtils.jl

- [Nerf.jl](https://github.com/JuliaNeuralGraphics/Nerf.jl)
- [NerfGUI.jl](https://github.com/JuliaNeuralGraphics/NerfGUI.jl)
