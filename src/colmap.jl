module COLMAP

using OrderedCollections

struct Camera
    cam_type::Int32
    resolution::NTuple{2, Int}     # w, h
    intrinsics::NTuple{4, Float64} # fx, fy, cx, cy
    distortions::NTuple{4, Float64}
end

function Camera(
    cam_type::Integer, width::Integer, height::Integer, params::Vector{Float64},
)
    @assert 0 ≤ cam_type ≤ 1
    intrinsics, distortions = if cam_type == 0
        fx, cx, cy = params
        (fx, fx, cx, cy), (0, 0, 0, 0)
    elseif cam_type == 1
        (params...,), (0, 0, 0, 0)
    elseif cam_type == 2 # TODO
    elseif cam_type == 3
    elseif cam_type == 4
    end
    Camera(cam_type, (width, height), intrinsics, distortions)
end

struct Image
    name::String
    cam_id::Int32
    q::Vector{Float64}
    t::Vector{Float64}
    points_2d::Matrix{Float64}
    points_3d_ids::Vector{UInt64}
end

function cam_type_name(cam_type::Integer)
    cam_type == 0 && return "SIMPLE_PINHOLE"
    cam_type == 1 && return "PINHOLE"
    cam_type == 2 && return "SIMPLE_RADIAL"
    cam_type == 3 && return "RADIAL"
    cam_type == 4 && return "OPENCV"
    error("Unsupported camera type: `$cam_type`.")
end

function cam_type_params(cam_type::Integer)
    cam_type == 0 && return 3
    cam_type == 1 && return 4
    cam_type == 2 && return 4
    cam_type == 3 && return 5
    cam_type == 4 && return 8
    error("Unsupported camera type: `$cam_type`.")
end

# TODO check file extension, only .bin for now

function load_cameras_data(camfile::String)
    cams = Dict{Int, Camera}()
    open(camfile) do io
        n_cams = read(io, UInt64)
        for _ in 1:n_cams
            cam_id, cam_type, w, h = read.(io, (Int32, Int32, Int64, Int64))
            n_params = cam_type_params(cam_type)
            params = [read(io, Float64) for _ in 1:n_params]
            cams[cam_id] = Camera(cam_type, w, h, params)
        end
    end
    return cams
end

function load_images_data(imfile::String)
    function _read_name(io)
        bytes = UInt8[]
        while true
            byte = read(io, UInt8)
            byte == 0x0 && break
            push!(bytes, byte)
        end
        String(bytes)
    end

    images = OrderedDict{Int32, Image}()
    open(imfile) do io
        n_images = read(io, UInt64)
        for _ in 1:n_images
            image_id = read(io, Int32)
            q = [read(io, Float64) for _ in 1:4]
            t = [read(io, Float64) for _ in 1:3]
            cam_id = read(io, Int32)
            name = _read_name(io)

            n_points = read(io, UInt64)
            points_2d = Matrix{Float64}(undef, 2, n_points)
            points_3d_ids = Vector{UInt64}(undef, n_points)
            for i in 1:n_points
                points_2d[1, i] = read(io, Float64)
                points_2d[2, i] = read(io, Float64)
                points_3d_ids[i] = read(io, UInt64)
            end

            images[image_id] = Image(name, cam_id, q, t, points_2d, points_3d_ids)
        end
    end
    return images
end

function load_points_data(points_file::String)
    open(points_file) do io
        n_points = read(io, UInt64)

        points_3d = Matrix{Float32}(undef, 3, n_points)
        points_ids = Vector{UInt64}(undef, n_points) # point-to-image-id
        points_colors = Matrix{UInt8}(undef, 3, n_points)
        points_errors = Vector{Float64}(undef, n_points)

        point_id_to_idx = Dict{UInt64, UInt64}()
        # Point ID -> [(image ID, point 2d ID), ...]
        point_id_to_images = Dict{UInt64, Matrix{UInt32}}()

        for i in 1:n_points
            points_ids[i] = read(io, UInt64)
            points_3d[:, i] .= [read(io, Float64) for _ in 1:3]
            points_colors[:, i] .= [read(io, UInt8) for _ in 1:3]
            points_errors[i] = read(io, Float64)

            # Load (image ID, point 2d ID) pairs.
            n_pairs = read(io, UInt64)
            pairs = reshape([read(io, UInt32) for _ in 1:(2 * n_pairs)], 2, n_pairs)

            # Point ID -> to its id in other arrays (poitns_3d, etc.).
            point_id_to_idx[points_ids[i]] = i
            point_id_to_images[points_ids[i]] = pairs
        end

        return (;
            points_3d, points_ids, points_colors, points_errors,
            point_id_to_idx, point_id_to_images)
    end
end

end
