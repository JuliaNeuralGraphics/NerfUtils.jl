# NerfUtils.jl

Reusable NeRF components.

## Requirements

- Julia 1.9 or higher.
- AMD or Nvidia GPU (supported by AMDGPU.jl or CUDA.jl respectively).

## Backends

When using components from this library you have to pass:
- `ROCBackend()` from AMDGPU.jl package or
- `CUDABackend()` from CUDA.jl package
to use respective GPU.
