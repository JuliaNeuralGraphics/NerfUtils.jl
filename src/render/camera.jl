"""
    Camera(projection, intrinsics::CameraIntrinsics)

Camera maintains original focal length,
since the model was trained on a specific value of it,
thus we want to preserve the original aspect.

# Arguments:

- `projection::MMatrix{3, 4, Float32}`: Camera-to-world projection.
- `intrinsics::CameraIntrinsics`: Camera intrinsics.
- `original_focal::SVector{2, Float32}`: Original focal length.
    Used during camera resizing to preserve original focal length aspect ratio.
- `original_resolution::SVector{2, UInt32}`: Original resolution.
    Used during camera resizing to compute scale value
    to multiply with `original_focal`.
"""
mutable struct Camera
    projection::MMatrix{3, 4, Float32, 12}
    intrinsics::CameraIntrinsics

    original_focal::SVector{2, Float32}
    original_resolution::SVector{2, UInt32}
end

function Camera(projection::MMatrix{3, 4, Float32, 12}, intrinsics::CameraIntrinsics)
    Camera(projection, intrinsics, intrinsics.focal, intrinsics.resolution)
end

Camera(projection, intrinsics) =
    Camera(convert(MMatrix{3, 4, Float32, 12}, projection), intrinsics)

function set_projection!(
    c::Camera, rotation::SMatrix{3, 3, Float32, 9},
    translation::SVector{3, Float32},
)
    c.projection[1:3, 1:3] .= rotation
    c.projection[1:3, 4] .= translation
end

"""
    shift!(c::Camera, relative)

Shift camera position by a `relative` value.
"""
function shift!(c::Camera, relative)
    c.projection[1:3, 4] .+= @view(c.projection[1:3, 1:3]) * relative
end

"""
    rotate!(c::Camera, rotation)

Apply rotation to the camera.
"""
function rotate!(c::Camera, rotation)
    c.projection[1:3, 1:3] .= rotation * @view(c.projection[1:3, 1:3])
end

view_side(c::Camera) = SVector{3, Float32}(@view(c.projection[1:3, 1])...)

view_up(c::Camera) = SVector{3, Float32}(@view(c.projection[1:3, 2])...)

view_dir(c::Camera) = SVector{3, Float32}(@view(c.projection[1:3, 3])...)

view_pos(c::Camera) = SVector{3, Float32}(@view(c.projection[1:3, 4])...)

look_at(c::Camera) = view_pos(c) .+ view_dir(c)

"""
    set_resolution!(c::Camera; width::Int, height::Int)

Change resolution of the camera.
Preserves original focal length aspect, scaling it instead.
"""
function set_resolution!(c::Camera; width::Int, height::Int)
    resolution = SVector{2, UInt32}(width, height)
    scale::Float32 = resolution[2] / c.original_resolution[2]
    focal = c.original_focal .* scale
    c.intrinsics = CameraIntrinsics(c.intrinsics; resolution, focal)
    return
end

get_resolution(c::Camera) = tuple(Int.(c.intrinsics.resolution)...)

function split_pose(c::Camera)
    rotation = SMatrix{3, 3, Float32, 9}(c.projection[1:3, 1:3])
    translation = SVector{3, Float32}(c.projection[1:3, 4])
    rotation, translation
end

function set_intrinsics!(c::Camera, intrinsics::CameraIntrinsics)
    c.intrinsics = intrinsics
    c.original_focal = intrinsics.focal
    c.original_resolution = intrinsics.resolution
    return
end
