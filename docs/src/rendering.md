# Rendering

## Camera

[`Camera`](@ref) is used to cast rays according to its intrinsic
and extrinsic parameters.
When resizing its resolution it preserves original aspect ratio
of focal length since that's the value the model was trained on.

```@docs
Camera
NerfUtils.set_resolution!
NerfUtils.shift!
NerfUtils.rotate!
```

## Camera intrinsics

[`CameraIntrinsics`](@ref) is used to project from pixel space to camera space.

```@docs
CameraIntrinsics
```

## CameraKeyframe

To smoothly transform camera from one position to another,
use [`NerfUtils.CameraKeyframe`](@ref).
Used in Video Mode in [NerfGUI.jl](https://github.com/JuliaNeuralGraphics/NerfGUI.jl).

```@docs
NerfUtils.CameraKeyframe
NerfUtils.spline
```
