"""
Camera intrinsics for projecting from pixel to camera space.

Projection is done as follows: `((x, y) .- principal) ./ focal`.
Followed by undistortion if any.

# Arguments:

- `distortion::Maybe{SVector{4, Float32}}`: If no distortion, then `nothing`.
- `focal::SVector{2, Float32}`: Focal length in `(fx, fy)` format.
- `principal::SVector{2, Float32}`: Principal point in `(cx, cy)` format.
    Can be either normalized by the `resolution` (in `[0, 1]` range)
    or not (in `[0, width/height]` range).
- `resolution::SVector{2, UInt32}`: Resolution in `(width, height)` format.
"""
struct CameraIntrinsics{D <: Maybe{SVector{4, Float32}}}
    distortion::D
    focal::SVector{2, Float32}
    principal::SVector{2, Float32}
    resolution::SVector{2, UInt32}
end

function CameraIntrinsics(distortion, focal, principal, resolution)
    has_distortion = !iszero(distortion)
    CameraIntrinsics(
        has_distortion ? distortion : nothing,
        focal, principal, resolution)
end

# Copy constructor.
function CameraIntrinsics(
    c::CameraIntrinsics; distortion=c.distortion, focal=c.focal,
    principal=c.principal, resolution=c.resolution,
)
    CameraIntrinsics(distortion, focal, principal, resolution)
end

function fov2focal(resolution, degrees::Float32)
    0.5f0 * resolution / tan(0.5f0 * deg2rad(degrees))
end

function focal2fov(resolution, focal_length)
    2f0 * rad2deg(atan(resolution / (focal_length * 2f0)))
end
