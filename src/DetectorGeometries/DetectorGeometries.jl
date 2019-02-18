"""
    abstract type SolidStateDetector{T} <: AbstractConfig{T}

Supertype of all detector structs.
"""
abstract type SolidStateDetector{T <: AbstractFloat} <: AbstractConfig{T} end

get_precision_type(d::SolidStateDetector{T}) where {T} = T

include("Geometries/Geometries.jl")
include("Contacts.jl")


include("BEGe.jl")
include("Coax.jl")
include("Inverted_Coax.jl")
# include("DetectorGeometries_V2.jl")

# Cartesian:
include("CGD.jl")

"""
    SolidStateDetector{T}(filename::AbstractString)::SolidStateDetector{T} where {T <: AbstractFloat}

Reads in a config-JSON file and returns an Detector struct which holds all information specified in the config file.
"""
function SolidStateDetector{T}(filename::AbstractString)::SolidStateDetector{T} where {T <: AbstractFloat}
    dicttext = read(filename, String)
    parsed_json_file = JSON.parse(dicttext)
    detector_class = parsed_json_file["class"]
    if detector_class == "Coax"
        return Coax{T}(filename)
    elseif detector_class == "BEGe"
        return BEGe{T}(filename)
    elseif detector_class == "InvertedCoax"
        return InvertedCoax{T}(filename)
    elseif detector_class == "CGD"
        return CGD{T}(filename)
    else
        error("Config File does not suit any of the predefined detector geometries. You may want to implement your own 'class'")
    end
end
function SolidStateDetector(T::Type{<:AbstractFloat} = Float32, filename::AbstractString = SSD_examples[:InvertedCoax])::SolidStateDetector{T}
    SolidStateDetector{T}(filename)
end
function SolidStateDetector(filename::AbstractString)::SolidStateDetector{Float32}
    SolidStateDetector{Float32}(filename)
end
function get_important_r_points_from_geometry(c::Coax)
    important_r_points_from_geometry::Vector = []
    push!(important_r_points_from_geometry,c.crystal_radius)
    push!(important_r_points_from_geometry,c.borehole_radius)
    push!(important_r_points_from_geometry,c.taper_inner_bot_rOuter)
    push!(important_r_points_from_geometry,c.taper_inner_top_rOuter)
    important_r_points_from_geometry
end

function get_important_r_points_from_geometry(b::BEGe)
    important_r_points_from_geometry::Vector = []
    push!(important_r_points_from_geometry,b.crystal_radius)
    push!(important_r_points_from_geometry,b.taper_bot_rInner)
    push!(important_r_points_from_geometry,b.taper_top_rInner)
    push!(important_r_points_from_geometry,b.groove_rInner)
    push!(important_r_points_from_geometry,b.groove_rInner+b.groove_width)
    important_r_points_from_geometry
end
# function get_important_r_points_from_geometry(ivc::InvertedCoax)
#     important_r_points_from_geometry::Vector = []
#     for v in ivc.volumes
#             push!(important_r_points_from_geometry,v.rStart)
#             push!(important_r_points_from_geometry,v.rStop)
#     end
#     important_r_points_from_geometry
# end
function get_important_r_points_from_geometry(ivc::InvertedCoax{T})::Vector{T} where {T <: AbstractFloat}
    important_r_points_from_geometry::Vector{T} = []
    for v in ivc.volumes
            push!(important_r_points_from_geometry, v.rStart)
            push!(important_r_points_from_geometry, v.rStop)
    end
    important_r_points_from_geometry
end


function get_important_r_points(d::SolidStateDetector{T})::Vector{T} where {T <: AbstractFloat}
    important_r_points::Vector{T} = []
    ## Outer Shape
    push!(important_r_points, get_important_r_points_from_geometry(d)...)
    ## From Segmentation
    for tuple in d.segmentation_r_ranges
        !in(tuple[1],important_r_points) ? push!(important_r_points, tuple[1]) : nothing
        !in(tuple[2],important_r_points) ? push!(important_r_points, tuple[2]) : nothing
    end
    return important_r_points
end

function get_important_φ_points(d::SolidStateDetector{T})::Vector{T} where {T <: AbstractFloat}
    important_φ_points::Vector{T} = []
    for tuple in d.segmentation_phi_ranges
        !in(tuple[1],important_φ_points) ? push!(important_φ_points,tuple[1]) : nothing
        !in(tuple[2],important_φ_points) ? push!(important_φ_points,tuple[2]) : nothing
    end
    for boundary_midpoint in d.segmentation_boundaryMidpoints_horizontal
        !in(boundary_midpoint,important_φ_points) ? push!(important_φ_points, boundary_midpoint) : nothing
    end
    return important_φ_points
end

function get_important_z_points_from_geometry(c::Coax{T})::Vector{T} where {T <: AbstractFloat}
    important_z_points_from_geometry::Vector{T} = []
    push!(important_z_points_from_geometry,0.0)
    push!(important_z_points_from_geometry,c.crystal_length)
    push!(important_z_points_from_geometry,c.taper_inner_bot_length)
    push!(important_z_points_from_geometry,c.crystal_length-c.taper_inner_top_length)
    important_z_points_from_geometry
end

function get_important_z_points_from_geometry(b::BEGe{T})::Vector{T} where {T <: AbstractFloat}
    important_z_points_from_geometry::Vector{T} = []
    push!(important_z_points_from_geometry,0.0)
    push!(important_z_points_from_geometry,b.crystal_length)
    push!(important_z_points_from_geometry,b.taper_bot_length)
    push!(important_z_points_from_geometry,b.crystal_length-b.taper_top_length)
    b.groove_endplate == "top" ? push!(important_z_points_from_geometry,b.crystal_length-b.groove_depth) : push!(important_z_points_from_geometry,b.groove_depth)
    important_z_points_from_geometry
end
function get_important_z_points_from_geometry(ivc::InvertedCoax)
    important_z_points_from_geometry::Vector = []
    for v in ivc.volumes
            push!(important_z_points_from_geometry,v.zStart)
            push!(important_z_points_from_geometry,v.zStop)
    end
    important_z_points_from_geometry
end
function get_important_z_points(d::SolidStateDetector)
    T=get_precision_type(d)
    important_z_points::Vector{T} = []
    ## Outer Shape
    push!(important_z_points,get_important_z_points_from_geometry(d)...)
    ## From Segmentation
    for tuple in d.segmentation_z_ranges
        !in(tuple[1],important_z_points) ? push!(important_z_points,tuple[1]) : nothing
        !in(tuple[2],important_z_points) ? push!(important_z_points,tuple[2]) : nothing
    end
    for boundary_midpoint in d.segmentation_boundaryMidpoints_vertical
        !in(boundary_midpoint,important_z_points) ? push!(important_z_points,boundary_midpoint) : nothing
    end
    important_z_points
end

function construct_segmentation_arrays_from_repetitive_segment(d::SolidStateDetector{T}, config_file::Dict)::Nothing where {T <: AbstractFloat}
    n_total_segments::Int = d.n_total_contacts
    f = d.geometry_unit_factor
    segmentation_r_ranges::Array{Tuple{T,T},1}= []
    segmentation_phi_ranges::Array{Tuple{T,T},1} = []
    segmentation_z_ranges::Array{Tuple{T,T},1} = []
    segment_bias_voltages::Array{T,1} = []
    segmentation_boundaryWidths_horizontal::Array{Tuple{T,T},1} = []
    segmentation_boundaryWidths_vertical::Array{Tuple{T,T},1} = []
    segmentation_boundaryMidpoints_vertical::Array{T,1} = []
    segmentation_boundaryMidpoints_horizontal::Array{T,1} = []

    #Core
    push!(segmentation_r_ranges,(geom_round(config_file["segmentation"]["core"]["rStart"]*f),geom_round(config_file["segmentation"]["core"]["rStop"]*f)))
    push!(segmentation_phi_ranges,(geom_round(deg2rad(config_file["segmentation"]["core"]["phiStart"])),geom_round(deg2rad(config_file["segmentation"]["core"]["phiStop"]))))
    push!(segmentation_z_ranges,(geom_round(config_file["segmentation"]["core"]["zStart"]*f),geom_round(config_file["segmentation"]["core"]["zStop"]*f)))
    push!(segment_bias_voltages,geom_round(config_file["segmentation"]["core"]["potential"]))

    #repetitive Segments
    # if d.n_repetitive_segments > 0
    rStart=geom_round(config_file["segmentation"]["repetitive_segment"]["rStart"]*f)
    rStop=geom_round(config_file["segmentation"]["repetitive_segment"]["rStop"]*f)
    boundaryWidth_horizontal = geom_round(deg2rad(config_file["segmentation"]["repetitive_segment"]["boundaryWidth"]["horizontal"]*f*180/(π*d.crystal_radius)))
    phiStart=geom_round(deg2rad(config_file["segmentation"]["repetitive_segment"]["phiStart"])+boundaryWidth_horizontal/2)
    phiStop=geom_round(deg2rad(config_file["segmentation"]["repetitive_segment"]["phiStop"]) - boundaryWidth_horizontal/2)
    zStart=geom_round(config_file["segmentation"]["repetitive_segment"]["zStart"]*f)
    boundaryWidth_vertical=geom_round(config_file["segmentation"]["repetitive_segment"]["boundaryWidth"]["vertical"]*f)
    zStop=geom_round(config_file["segmentation"]["repetitive_segment"]["zStop"]*f - boundaryWidth_vertical)
    potential=geom_round(config_file["segmentation"]["repetitive_segment"]["potential"])
    n_vertical_repetitions = config_file["segmentation"]["repetitive_segment"]["repetitions"]["vertical"]
    n_horizontal_repetitions = config_file["segmentation"]["repetitive_segment"]["repetitions"]["horizontal"]
    for v_rseg in 0 : n_vertical_repetitions-1
            for h_rseg in 0 : n_horizontal_repetitions-1
                    push!(segmentation_r_ranges, (rStart,rStop))
                    h_offset::T = h_rseg*(phiStop-phiStart + boundaryWidth_horizontal)
                    push!(segmentation_phi_ranges,(phiStart+h_offset,phiStop+h_offset))
                    push!(segmentation_boundaryMidpoints_horizontal,phiStart+h_offset-boundaryWidth_horizontal/2)
                    v_offset::T=v_rseg*(zStop-zStart + boundaryWidth_vertical)
                    # v_rseg == n_vertical_repetitions -1 ? push!(segmentation_z_ranges,((zStart+v_offset),(zStop+v_offset))) : push!(segmentation_z_ranges,((zStart+v_offset),(zStop+v_offset)))
                    v_rseg == n_vertical_repetitions -1 ? push!(segmentation_z_ranges,((zStart+v_offset),(zStop+v_offset +boundaryWidth_vertical))) : push!(segmentation_z_ranges,((zStart+v_offset),(zStop+v_offset)))
                    push!(segmentation_boundaryMidpoints_vertical,zStart+v_offset + boundaryWidth_vertical/2)
                    push!(segment_bias_voltages,potential)
            end
    end
    d.segmentation_r_ranges   = segmentation_r_ranges
    d.segmentation_phi_ranges = segmentation_phi_ranges
    d.segmentation_z_ranges   = segmentation_z_ranges
    d.segment_bias_voltages = segment_bias_voltages
    d.segmentation_boundaryMidpoints_horizontal = segmentation_boundaryMidpoints_horizontal
    d.segmentation_boundaryMidpoints_vertical = segmentation_boundaryMidpoints_vertical
    nothing
end

function construct_segmentation_arrays_for_individual_segments(d::SolidStateDetector{T}, config_file::Dict)::Nothing where {T <: AbstractFloat}
    n_individual_segments::Int = d.n_individual_segments
    f = T(d.geometry_unit_factor)

    segmentation_r_ranges::Array{Tuple{T,T},1}= []
    segmentation_phi_ranges::Array{Tuple{T,T},1} = []
    segmentation_z_ranges::Array{Tuple{T,T},1} = []
    segment_bias_voltages::Array{T,1} = []
    segmentation_types::Array{String,1} = []
    segmentation_boundaryWidths_horizontal::Array{Tuple{T,T},1} = []
    segmentation_boundaryWidths_vertical::Array{Tuple{T,T},1} = []
    segmentation_boundaryMidpoints_radial::Array{T,1} = []
    segmentation_boundaryMidpoints_vertical::Array{T,1} = []
    segmentation_boundaryMidpoints_horizontal::Array{T,1} = []

    push!(segmentation_r_ranges,(geom_round(config_file["segmentation"]["core"]["rStart"]*f),geom_round(config_file["segmentation"]["core"]["rStop"]*f)))
    push!(segmentation_phi_ranges,(geom_round(deg2rad(config_file["segmentation"]["core"]["phiStart"])),geom_round(deg2rad(config_file["segmentation"]["core"]["phiStop"]))))
    push!(segmentation_z_ranges,(geom_round(config_file["segmentation"]["core"]["zStart"]*f),geom_round(config_file["segmentation"]["core"]["zStop"]*f)))
    push!(segmentation_types,config_file["segmentation"]["core"]["type"])
    push!(segment_bias_voltages,geom_round(config_file["segmentation"]["core"]["potential"]))
    for i_idv_seg in 1:n_individual_segments
            ID = "S$i_idv_seg"
            if config_file["segmentation"][ID]["type"] == "Tubs"
            seg_type = "Tubs"
            elseif config_file["segmentation"][ID]["type"] == "Taper"
            seg_type = config_file["segmentation"][ID]["orientation"]
            end
            boundaryWidth_radial::T = geom_round(T(config_file["segmentation"][ID]["boundaryWidth"]["radial"]*f))
            rStart::T=geom_round(T(config_file["segmentation"][ID]["rStart"]*f+boundaryWidth_radial))
            rStop::T=geom_round(T(config_file["segmentation"][ID]["rStop"]*f))
            boundaryWidth_horizontal::T = geom_round(T(deg2rad(config_file["segmentation"][ID]["boundaryWidth"]["horizontal"]*f*180/(π*d.crystal_radius))))
            phiStart::T=geom_round(T(deg2rad(config_file["segmentation"][ID]["phiStart"]) + boundaryWidth_horizontal/2))
            phiStop::T=geom_round(T(deg2rad(config_file["segmentation"][ID]["phiStop"]) - boundaryWidth_horizontal/2))
            boundaryWidth_vertical::T=geom_round(T(config_file["segmentation"][ID]["boundaryWidth"]["vertical"]*f))
            zStart::T=geom_round(T(config_file["segmentation"][ID]["zStart"]*f + boundaryWidth_vertical/2))
            zStop::T=geom_round(T(config_file["segmentation"][ID]["zStop"]*f - boundaryWidth_vertical/2))
            potential::T=geom_round(T(config_file["segmentation"][ID]["potential"]))
            if config_file["segmentation"][ID]["repetitive"]==true
                n_radial_repetitions = config_file["segmentation"][ID]["repetitions"]["radial"]
                n_vertical_repetitions = config_file["segmentation"][ID]["repetitions"]["vertical"]
                n_horizontal_repetitions = config_file["segmentation"][ID]["repetitions"]["horizontal"]
                for ir in 0:n_radial_repetitions
                    for iv in 0:n_vertical_repetitions
                        for ih in 0:n_horizontal_repetitions
                            r_offset::T = ir*(rStop-rStart + boundaryWidth_radial)
                            h_offset::T = ih*(phiStop-phiStart + boundaryWidth_horizontal)
                            v_offset::T = iv*(zStop-zStart + boundaryWidth_vertical)
                            if iv == 0 && n_vertical_repetitions > 0
                                push!(segmentation_z_ranges,(T(0.0) , zStop + v_offset))

                            elseif iv == n_vertical_repetitions && n_vertical_repetitions > 0
                                push!(segmentation_z_ranges,(zStart + v_offset, d.crystal_length))

                            else
                                push!(segmentation_z_ranges,(zStart + v_offset, zStop + v_offset))
                            end
                            push!(segmentation_r_ranges,(rStart - r_offset, rStop - r_offset))
                            push!(segmentation_phi_ranges,(phiStart + h_offset, phiStop + h_offset))

                            push!(segmentation_types,seg_type)
                            push!(segment_bias_voltages,potential)

                            push!(segmentation_boundaryMidpoints_radial,rStart-r_offset-boundaryWidth_radial/2)
                            push!(segmentation_boundaryMidpoints_horizontal,phiStop+h_offset+boundaryWidth_horizontal/2)

                            iv < n_vertical_repetitions ? push!(segmentation_boundaryMidpoints_vertical,zStop+v_offset+boundaryWidth_vertical/2) : nothing

                        end
                    end
                end
            else
                push!(segmentation_r_ranges, (rStart,rStop))
                push!(segmentation_phi_ranges,(phiStart,phiStop))
                push!(segmentation_z_ranges,(zStart,zStop))
                push!(segmentation_types,seg_type)
                push!(segment_bias_voltages,potential)

                push!(segmentation_boundaryMidpoints_radial,rStart-boundaryWidth_radial/2)
                push!(segmentation_boundaryMidpoints_horizontal,phiStop+boundaryWidth_horizontal/2)
                push!(segmentation_boundaryMidpoints_vertical,zStop+boundaryWidth_vertical/2)
            end

    end
    d.segmentation_r_ranges   = segmentation_r_ranges
    d.segmentation_phi_ranges = segmentation_phi_ranges
    d.segmentation_z_ranges   = segmentation_z_ranges
    d.segmentation_types      = segmentation_types
    d.segment_bias_voltages = segment_bias_voltages
    d.segmentation_boundaryMidpoints_radial = segmentation_boundaryMidpoints_radial
    d.segmentation_boundaryMidpoints_horizontal = segmentation_boundaryMidpoints_horizontal
    d.segmentation_boundaryMidpoints_vertical = segmentation_boundaryMidpoints_vertical
    nothing
end

function construct_floating_boundary_arrays(d::SolidStateDetector{T}) where {T <: AbstractFloat}
    ## Groove
    floating_boundary_r_ranges=[]
    floating_boundary_phi_ranges=[]
    floating_boundary_z_ranges=[]
    floating_boundary_types=[]
    push!(floating_boundary_r_ranges,(d.groove_rInner,d.groove_rInner))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    if d.groove_endplate=="bot"
            push!(floating_boundary_z_ranges,(0.0,d.groove_depth))
    else
            push!(floating_boundary_z_ranges,(d.crystal_length-d.groove_depth,d.crystal_length))
    end
    push!(floating_boundary_types,"Tubs")

    push!(floating_boundary_r_ranges,(d.groove_rInner,d.groove_rInner+d.groove_width))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    push!(floating_boundary_z_ranges,(d.groove_depth,d.groove_depth))
    push!(floating_boundary_types,"Tubs")

    push!(floating_boundary_r_ranges,(d.groove_rInner+d.groove_width,d.groove_rInner+d.groove_width))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    if d.groove_endplate=="bot"
            push!(floating_boundary_z_ranges,(0.0,d.groove_depth))
    else
            push!(floating_boundary_z_ranges,(d.crystal_length-d.groove_depth,d.crystal_length))
    end
    push!(floating_boundary_types,"Tubs")


    ## outer_crystal
    push!(floating_boundary_r_ranges,(0.0,d.crystal_radius))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    push!(floating_boundary_z_ranges,(0.0,0.0))
    push!(floating_boundary_types,"Tubs")

    push!(floating_boundary_r_ranges,(d.crystal_radius,d.crystal_radius))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    push!(floating_boundary_z_ranges,(0.0,d.crystal_length))
    push!(floating_boundary_types,"Tubs")

    push!(floating_boundary_r_ranges,(0.0,d.crystal_radius))
    push!(floating_boundary_phi_ranges,(0.0,2π))
    push!(floating_boundary_z_ranges,(d.crystal_length,d.crystal_length))
    push!(floating_boundary_types,"Tubs")

    ## detector specific outer boundaries
    add_detector_specific_outer_boundaries(d,floating_boundary_r_ranges,floating_boundary_phi_ranges,floating_boundary_z_ranges,floating_boundary_types)

    d.floating_boundary_r_ranges=floating_boundary_r_ranges
    d.floating_boundary_phi_ranges=floating_boundary_phi_ranges
    d.floating_boundary_z_ranges=floating_boundary_z_ranges
    d.floating_boundary_types=floating_boundary_types
end

function add_detector_specific_outer_boundaries(c::Coax, rs,phis, zs, mytypes)
    ## taper_top
    if !iszero(c.taper_outer_top_length)
            push!(rs,(c.taper_outer_top_rInner,c.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(c.crystal_length-c.taper_outer_top_length,c.crystal_length))
            push!(mytypes,"c//")
    end
    if !iszero(c.taper_inner_top_length)
            push!(rs,(c.taper_inner_top_rOuter,c.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(c.crystal_length-c.taper_inner_top_length,c.crystal_length))
            push!(mytypes,"/c")
    end
    ## taper_bot
    if !iszero(c.taper_outer_bot_length)
            push!(rs,(c.taper_outer_bot_rInner,c.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(c.taper_outer_bot_length,c.crystal_length))
            push!(mytypes,"c/")
    end
    if !iszero(c.taper_inner_bot_length)
            push!(rs,(c.taper_inner_bot_rOuter,c.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(c.taper_inner_bot_length,c.crystal_length))
            push!(mytypes,"//c")
    end
end

function add_detector_specific_outer_boundaries(b::BEGe,rs,phis,zs,mytypes)
    ## taper_top
    if !iszero(b.taper_top_length)
            push!(rs,(b.taper_top_rInner,b.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(b.crystal_length-b.taper_top_length,b.crystal_length))
            push!(mytypes,"c//")
    end
    ## taper_bot
    if !iszero(b.taper_bot_length)
            push!(rs,(b.taper_bot_rInner,b.crystal_radius))
            push!(phis,(0.,2π))
            push!(zs,(b.taper_bot_length,b.crystal_length))
            push!(mytypes,"c/")
    end
end
function add_detector_specific_outer_boundaries(ivc::InvertedCoax,rs,phis,zs,mytypes)
        nothing
end


@inline in(point::CylindricalPoint, detector::SolidStateDetector) =
    contains(detector, point)

@inline in(point::StaticArray{Tuple{3}, <:Real}, detector::SolidStateDetector) =
    convert(CylindricalPoint, point) in detector

@inline in(point::StaticArray{Tuple{3}, <:Quantity}, detector::SolidStateDetector) =
    to_internal_units(u"m", point) in detector

# @inline in(point::CoordinateTransformations.Cylindrical, detector::SolidStateDetector) =
#     convert(CylindricalPoint, point) in detector




# TODO: Deprecate function contains in favour of Base.in (see above):

# false -> outside
function contains(c::Coax, r::Real, φ::Real, z::Real)::Bool
    rv::Bool = true
    if !check_outer_limits(c, r, φ, z) rv = false end
    if !check_borehole(c, r, φ, z) rv = false end
    if !check_tapers(c, r, φ, z) rv = false end
    return rv
end
function contains(c::Coax, p::CylindricalPoint)::Bool
    rv::Bool = true
    if !check_outer_limits(c, p.r,p.φ,p.z) rv = false end
    if !check_borehole(c, p.r,p.φ,p.z) rv = false end
    if !check_tapers(c, p.r,p.φ,p.z) rv = false end
    return rv
end
function contains(b::BEGe, r::Real, φ::Real, z::Real)::Bool
    rv::Bool = true
    check_outer_limits(b,r,φ,z) ? nothing : rv = false
    check_tapers(b,r,φ,z) ? nothing : rv = false
    check_grooves(b,r,φ,z) ? nothing : rv = false
    rv
end
function contains(b::BEGe, p::CylindricalPoint)::Bool
    rv::Bool = true
    check_outer_limits(b,p.r,p.φ,p.z) ? nothing : rv = false
    check_tapers(b,p.r,p.φ,p.z) ? nothing : rv = false
    check_grooves(b,p.r,p.φ,p.z) ? nothing : rv = false
    rv
end
# function contains(b::BEGe, p::Cylindrical)::Bool
#     rv::Bool = true
#     check_outer_limits(b,p.r,p.φ,p.z) ? nothing : rv = false
#     check_tapers(b,p.r,p.φ,p.z) ? nothing : rv = false
#     check_grooves(b,p.r,p.φ,p.z) ? nothing : rv = false
#     rv
# end

function contains(ivc::InvertedCoax,r::Real, φ::Real, z::Real)::Bool
    rv::Bool = true
    check_outer_limits(ivc,r,φ,z) ? nothing : rv = false
    check_tapers(ivc,r,φ,z) ? nothing : rv = false
    if ivc.borehole_modulation == true
        check_borehole(ivc,r,φ,z,ivc.borehole_ModulationFunction) ? nothing : rv = false
    else
        check_borehole(ivc,r,φ,z) ? nothing : rv = false
    end
    check_grooves(ivc,r,φ,z) ? nothing : rv = false
    rv
end

function contains(ivc::InvertedCoax,p::CylindricalPoint)::Bool
    rv::Bool = true
    check_outer_limits(ivc,p.r,p.φ,p.z) ? nothing : rv = false
    check_tapers(ivc,p.r,p.φ,p.z) ? nothing : rv = false
    if ivc.borehole_modulation == true
        check_borehole(ivc,p.r,p.φ,p.z,ivc.borehole_ModulationFunction) ? nothing : rv = false
    else
        check_borehole(ivc,p.r,p.φ,p.z) ? nothing : rv = false
    end
    check_grooves(ivc,p.r,p.φ,p.z) ? nothing : rv = false
    rv
end
# function contains(ivc::InvertedCoax, p::Cylindrical)::Bool
#     rv::Bool = true
#     check_outer_limits(ivc,p.r,p.φ,p.z) ? nothing : rv = false
#     check_tapers(ivc,p.r,p.φ,p.z) ? nothing : rv = false
#     if ivc.borehole_modulation == true
#         check_borehole(ivc,p.r,p.φ,p.z,ivc.borehole_ModulationFunction) ? nothing : rv = false
#     else
#         check_borehole(ivc,p.r,p.φ,p.z) ? nothing : rv = false
#     end
#     check_grooves(ivc,p.r,p.φ,p.z) ? nothing : rv = false
#     rv
# end
# function contains(ivc::InvertedCoax,r::Real, φ::Real, z::Real, accepted_ϵ=[16.0])::Bool
#     rv::Bool = true
#     crystal_basic_shape = ivc.volumes[1]
#     if check_volume(crystal_basic_shape,r,φ,z)
#         for v in ivc.volumes
#             if v.ϵ in accepted_ϵ
#                 nothing
#             else
#                 if check_volume(v,r,φ,z) rv=false end
#             end
#         end
#         check_tapers(ivc,r,φ,z) ? nothing : rv = false
#     else
#         rv = false
#     end
#     rv
# end

function contains(b::BEGe, p::Tuple)::UInt8
    check_outer_limits(b,p) ? nothing : return 0
    check_tapers(b,p) ? nothing : return 0
    check_grooves(b,p) ? nothing : return 0
    return 1
end

function contains(c::Coax, p::Tuple)::UInt8
    check_outer_limits(c,p) ? nothing : return 0
    check_borehole(c,p) ? nothing : return 0
    check_tapers(c,p) ? nothing : return 0
    return 1
end

function check_outer_limits(d::SolidStateDetector, r::Real, φ::Real, z::Real)::Bool
    rv::Bool = true
    if (r > d.crystal_radius) || (z < 0) || (z > d.crystal_length)
        return false
    end
    return rv
end

function check_outer_limits(b::SolidStateDetector, p::Tuple)::Bool
    rv::Bool = true
    T::Type = get_precision_type(b)
    @fastmath begin
        r::T = geom_round(sqrt(p[1]^2 + p[2]^2))
        if r > b.crystal_radius || p[3] < 0 || p[3] > b.crystal_length
                rv = false
        end
    end
    return rv
end

function check_borehole(c::Coax, r::Real, φ::Real, z::Real)::Bool
    rv = true
    if r < c.borehole_radius
        rv = false
    end
    return rv
end
function check_borehole(c::Coax,p::Tuple)::Bool
    T::Type = get_precision_type(c)
    rv::Bool = true
    @fastmath begin
        r::T = geom_round(sqrt(p[1]^2 + p[2]^2))
        if r < c.borehole_radius
                rv = false
        end
    end
    return rv
end

function check_borehole(ivc::InvertedCoax, r::Real, φ::Real, z::Real)::Bool#returns true if point is not inside borehole
    rv = true
    if r < geom_round(ivc.borehole_radius) && z >geom_round(ivc.crystal_length-ivc.borehole_length)
        rv = false
    end
    rv
end

function check_borehole(ivc::InvertedCoax, r::T, φ::T, z::T, modulation_function::Function)::Bool where T<:Real#returns true if point is not inside borehole
    rv = true
    epsilon::T=0.000
    # if r < round(ivc.borehole_radius+modulation_function(φ)-epsilon,digits=6) && z >round(ivc.crystal_length-ivc.borehole_length,digits=6)
    if r < geom_round(ivc.borehole_radius+modulation_function(φ)) && z >geom_round(ivc.crystal_length-ivc.borehole_length)
        rv = false
    end
    rv
end

function check_tapers(b::BEGe, p::Tuple)
    T = get_precision_type(b)
    # p[1]=T(p[1])
    # p[2]=T(p[2])
    # p[3]=T(p[3])
    z::T = p[3]
    if z > (b.crystal_length-b.taper_top_length) && z <= b.crystal_length ##Check top taper
        angle_taper_top::T = atan((b.crystal_radius-b.taper_top_rInner)/b.taper_top_length)
        r_taper_top::T = tan(angle_taper_top)*(b.crystal_length-p[3])+b.taper_top_rInner
        r_point::T = sqrt(p[1]^2 + p[2]^2)
        if r_point > r_taper_top
            return false
        else
            nothing
        end
    elseif p[3] < (b.taper_bot_length) && p[3] >= 0.0 ## Check bot taper
        angle_taper_bot = atan((b.crystal_radius-b.taper_bot_rInner)/b.taper_bot_length)
        r_taper_bot = tan(angle_taper_bot) * p[3] + b.taper_bot_rInner
        r_point = rfromxy(p[1],p[2])
        if r_point > r_taper_bot
            # println("bot taper")
            return false
        else
            nothing
        end
    end
    return true
end

function check_tapers(b::BEGe,r::Real,φ::Real,z::Real)::Bool
    rv::Bool = true
    if z > (b.crystal_length-b.taper_top_length) && z <= b.crystal_length ##Check top taper
        angle_taper_top = atan((b.crystal_radius-b.taper_top_rInner)/b.taper_top_length)
        r_taper_top = tan(angle_taper_top)*(b.crystal_length-z)+b.taper_top_rInner
        if r > r_taper_top
            rv = false
        else
            nothing
        end
    elseif z < (b.taper_bot_length) && z >= 0.0 ## Check bot taper
        angle_taper_bot = atan((b.crystal_radius-b.taper_bot_rInner)/b.taper_bot_length)
        r_taper_bot = tan(angle_taper_bot) * z + b.taper_bot_rInner
        if r > r_taper_bot
            rv = false
        else
            nothing
        end
    end
    rv
end

function check_tapers(ivc::InvertedCoax, r::T, φ::T, z::T)::Bool where T<:Real
    rv::Bool = true
    if !iszero(ivc.taper_outer_length) && z > geom_round(ivc.crystal_length-ivc.taper_outer_length)
        if r > geom_round(ivc.crystal_radius-get_r_from_z_for_taper(ivc.taper_outer_angle, z-(ivc.crystal_length-ivc.taper_outer_length)))
                rv=false
        end
    end
    if !iszero(ivc.taper_inner_length) && z > geom_round(ivc.crystal_length-ivc.taper_inner_length)
        if r < ivc.borehole_radius + get_r_from_z_for_taper(ivc.taper_inner_angle, z-(ivc.crystal_length-ivc.taper_inner_length))
                rv=false
        end
    end
    rv
end

function get_r_from_z_for_taper(angle::T,z::T)::T where T<:Real
    return geom_round(z*tan(angle))
end

function check_tapers(c::Coax, r::Real, φ::Real, z::Real)::Bool
    if z > (c.crystal_length-c.taper_outer_top_length) && z <= c.crystal_length ##Check top taper
        angle_taper_outer_top = atan((c.crystal_radius-c.taper_outer_top_rInner)/c.taper_outer_top_length)
        r_taper_outer_top = tan(angle_taper_outer_top)*(c.crystal_length-z)+c.taper_outer_top_rInner
        # if r>c.type_precision(r_taper_outer_top)
        if r > r_taper_outer_top
            # println("top outer taper")
            return false
        end
        # elseif z < (c.taper_outer_bot_length) && z >= c.type_precision(0.0) ## Check bot taper
    elseif z < (c.taper_outer_bot_length) && z >= 0.0 ## Check bot taper
        angle_taper_outer_bot = atan((c.crystal_radius-c.taper_outer_bot_rInner)/c.taper_outer_bot_length)
        r_taper_outer_bot = tan(angle_taper_outer_bot) * z + c.taper_outer_bot_rInner
        # if r > c.type_precision(r_taper_outer_bot)
        if r > r_taper_outer_bot
            # println("bot outer taper")
            return false
        end
    end

    if z > (c.crystal_length-c.taper_inner_top_length) && z <= c.crystal_length ##Check top taper
        angle_taper_inner_top = atan((c.taper_inner_top_rOuter-c.borehole_radius)/c.taper_inner_top_length)
        r_taper_inner_top = c.taper_inner_top_rOuter - tan(angle_taper_inner_top) * (c.crystal_length - z)
        # if r < signif(c.type_precision(r_taper_inner_top), 5)
        if r < geom_round(r_taper_inner_top)
            # println("top inner taper")
            return false
        end
        # elseif z < (c.taper_inner_bot_length) && z >= c.type_precision(0.0) ## Check bot taper
    elseif z < (c.taper_inner_bot_length) && z >= 0 ## Check bot taper
        angle_taper_inner_bot = atan((c.taper_inner_bot_rOuter-c.borehole_radius)/c.taper_inner_bot_length)
        r_taper_inner_bot = c.taper_inner_bot_rOuter - tan(angle_taper_inner_bot) * z
        # if r < signif(c.type_precision(r_taper_inner_bot), 5)
        if r < geom_round(r_taper_inner_bot)
            # println("bot inner taper")
            return false
        end
    end
    return true
end

function check_tapers(c::Coax, p::Tuple)::Bool
        T::Type = get_precision_type(c)
        rv::Bool = true
        @fastmath begin
        if p[3] > (c.crystal_length - c.taper_outer_top_length) && p[3] <= c.crystal_length ##Check top taper
            angle_taper_outer_top::T = atan((c.crystal_radius - c.taper_outer_top_rInner) / c.taper_outer_top_length)
            r_taper_outer_top::T = tan(angle_taper_outer_top) * (c.crystal_length - p[3]) + c.taper_outer_top_rInner
            r_point::T = geom_round(sqrt(p[1]^2 + p[2]^2))
            if r_point > r_taper_outer_top
                # println("top outer taper")
                rv = false
            end
        elseif p[3] < c.taper_outer_bot_length && p[3] >= 0 ## Check bot taper
            angle_taper_outer_bot = atan((c.crystal_radius - c.taper_outer_bot_rInner) / c.taper_outer_bot_length)
            r_taper_outer_bot = tan(angle_taper_outer_bot) * p[3] + c.taper_outer_bot_rInner
            r_point = geom_round(sqrt(p[1]^2 + p[2]^2))
            if r_point > r_taper_outer_bot
                # println("bot outer taper")
                rv = false
            end
        end

        if p[3] > (c.crystal_length - c.taper_inner_top_length) && p[3] <= c.crystal_length ##Check top taper
            angle_taper_inner_top = atan((c.taper_inner_top_rOuter - c.borehole_radius) / c.taper_inner_top_length)
            r_taper_inner_top = c.taper_inner_top_rOuter - tan(angle_taper_inner_top) * (c.crystal_length - p[3])
            r_point = geom_round(sqrt(p[1]^2 + p[2]^2))
            if r_point < r_taper_inner_top
                # println("top inner taper")
                rv = false
            end
        elseif p[3] < (c.taper_inner_bot_length) && p[3] >= T(0) ## Check bot taper
            angle_taper_inner_bot = atan((c.taper_inner_bot_rOuter - c.borehole_radius) / c.taper_inner_bot_length)
            r_taper_inner_bot = c.taper_inner_bot_rOuter - tan(angle_taper_inner_bot) * p[3]
            r_point = geom_round(sqrt(p[1]^2 + p[2]^2))
            if r_point < r_taper_inner_bot
                # println("bot inner taper")
                rv = false
            end
        end
end
return rv
end

function check_grooves(b::BEGe,r::Real,φ::Real,z::Real)::Bool
    # T = get_precision_type(b)
    rv::Bool=true
    if b.groove_endplate == "bot"
        if z >= 0 && z < b.groove_depth
            if r > b.groove_rInner && r < (b.groove_rInner+b.groove_width)
                rv = false
            end
        end
    elseif b.groove_endplate == "top"
        if z < b.crystal_length && z > (b.crystal_length-b.groove_depth)
            if r > b.groove_rInner && r < (b.groove_rInner+b.groove_width)
                rv = false
            end
        end
    end
    rv
end

function check_grooves(ivc::InvertedCoax,r::Real,φ::Real,z::Real)::Bool
    T=typeof(z)
    rv::Bool=true
    if z >= T(0.0) && z < ivc.groove_depth
        if r>ivc.groove_rInner && r < (ivc.groove_rInner+ivc.groove_width)
            rv = false
        else
            nothing
        end
    end
    rv
end

function check_volume(v::Volume,r::Real,φ::Real,z::Real)::Bool #return true if in volume
    rv=true
    T=typeof(v.rStart)
    if typeof(v) == Tubs{T}
        if r< v.rStart || r > v.rStop
            rv = false
        elseif φ < v.φStart || φ > v.φStop
            rv = false
        elseif z < v.zStart || z > v.zStop
            rv = false
        end
    end
    rv
end

function check_grooves(b::BEGe, p::Tuple)
    T = get_precision_type(b)
    if p[3] > T(0) && p[3]<b.groove_depth
        r_point::T = rfromxy(p[1],p[2])
        if r_point>b.groove_rInner && r_point< (b.groove_rInner+b.groove_width)
            # println("groove")
            return false
        else
            nothing
        end
    end
    return true
end

function rfromxy(x::Real, y::Real)
    return sqrt(x^2+y^2)
end

# is_boundary
# function is_boundary_point(d::SolidStateDetector, r::Real, φ::Real, z::Real)::Bool
#     rv::Bool = false
#     for iseg in 1:size(d.segment_bias_voltages,1)
#         if (d.segmentation_r_ranges[iseg][1] <= r <= d.segmentation_r_ranges[iseg][2])
#             if (d.segmentation_phi_ranges[iseg][1] <= φ <= d.segmentation_phi_ranges[iseg][2])
#                 if (d.segmentation_z_ranges[iseg][1] <= z <= d.segmentation_z_ranges[iseg][2])
#                     rv = true
#                 end
#             end
#         end
#     end
#     rv
# end
# function is_boundary_point(d::Coax, r::Real, φ::Real, z::Real, rs, φs, zs)
#     is_boundary_point(d,r,φ,z)
# end
function is_boundary_point(d::SolidStateDetector, r::T, φ::T, z::T, rs::Vector{T}, φs::Vector{T}, zs::Vector{T}) where T <:AbstractFloat
    rv::Bool = false
    if φ < 0
        φ += d.cyclic
    end
    digits::Int=6
    if  d.borehole_modulation == true
        idx_r_closest_gridpoint_to_borehole = searchsortednearest(rs, d.borehole_radius + d.borehole_ModulationFunction(φ))
        idx_r = findfirst(x->x==r,rs)
        bore_hole_r = rs[idx_r_closest_gridpoint_to_borehole]
    else
        nothing
    end

    for iseg in 1:size(d.segment_bias_voltages,1)
        if d.borehole_modulation == true && iseg == d.borehole_segment_idx
             # x = (round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)-ϵ,digits=digits) <= r <= round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)+ϵ,digits=digits))
             x = (idx_r_closest_gridpoint_to_borehole == idx_r)
        elseif d.borehole_modulation == true && iseg == d.borehole_bot_segment_idx
             x = (d.segmentation_r_ranges[iseg][1] <= r <= geom_round(bore_hole_r))
             # x = idx_r_closest_gridpoint_to_borehole == idx_r
        elseif d.borehole_modulation == true && iseg == d.borehole_top_segment_idx
            x = (geom_round(bore_hole_r) <= r <= d.segmentation_r_ranges[iseg][2])
            # x = idx_r_closest_gridpoint_to_borehole == idx_r
        else
            x = (d.segmentation_r_ranges[iseg][1] <= r <= d.segmentation_r_ranges[iseg][2])
        end
        if x
            if (d.segmentation_phi_ranges[iseg][1] <= φ <= d.segmentation_phi_ranges[iseg][2])
                if (d.segmentation_z_ranges[iseg][1] <= z <= d.segmentation_z_ranges[iseg][2])
                    if d.segmentation_types[iseg]=="Tubs"
                        rv = true
                    else
                        if isapprox(rs[searchsortednearest(rs,analytical_taper_r_from_φz(φ,z,
                            d.segmentation_types[iseg],
                            d.segmentation_r_ranges[iseg],
                            d.segmentation_phi_ranges[iseg],
                            d.segmentation_z_ranges[iseg]
                            ))],r)
                            rv = true
                        end
                    end
                end
            end
        end
    end
    rv
end

function point_type(d::SolidStateDetector, p::CylindricalPoint)
    point_type(d, p.r, p.φ, p.z)
end
function point_type(d::SolidStateDetector{T}, r::T, φ::T, z::T) where { T<: AbstractFloat}
    # T==Float32 ? atol = 0.000001 : atol = 0.000000000000001
    rv::Symbol = :bulk
    !contains(d, CylindricalPoint{T}(r,φ,z)) ? rv = :outside : nothing
    i=0
    digits::Int=5
    r = geom_round(r)
    φ = geom_round(φ)
    z = geom_round(z)

    while φ < 0 φ += (2π) end
    while φ >= T(2π) φ -= (2π) end
    ############################# Electrode Definitions
    for iseg in 1:size(d.segment_bias_voltages,1)
        if d.borehole_modulation == true && iseg == d.borehole_segment_idx
             x = (geom_round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)) <= r <= geom_round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)))
        elseif d.borehole_modulation == true && iseg == d.borehole_bot_segment_idx
             x = (d.segmentation_r_ranges[iseg][1] <= r <= geom_round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)))
        elseif d.borehole_modulation == true && iseg == d.borehole_top_segment_idx
            x = (geom_round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)) <= r <= d.segmentation_r_ranges[iseg][2])
        else
            x = (d.segmentation_r_ranges[iseg][1] <= r <= d.segmentation_r_ranges[iseg][2])
        end
        if x
            if (d.segmentation_phi_ranges[iseg][1] <= φ <= d.segmentation_phi_ranges[iseg][2])
                if (d.segmentation_z_ranges[iseg][1] <= z <= d.segmentation_z_ranges[iseg][2])
                    if d.segmentation_types[iseg]=="Tubs"
                        rv = :electrode
                        i=iseg
                    else
                        analytic_r::T = geom_round(analytical_taper_r_from_φz(φ,z,
                            d.segmentation_types[iseg],
                            d.segmentation_r_ranges[iseg],
                            d.segmentation_phi_ranges[iseg],
                            d.segmentation_z_ranges[iseg]
                        ))
                        if isapprox(analytic_r, r )
                            rv = :electrode
                            i=iseg
                        end
                    end
                end
            end
        end
    end

    if i == 0
        ############################ Floating Boundary Definitions
        for iseg in 1:size(d.floating_boundary_r_ranges,1)
            if (d.floating_boundary_r_ranges[iseg][1] <= r <= d.floating_boundary_r_ranges[iseg][2])
                if (d.floating_boundary_phi_ranges[iseg][1] <= φ <= d.floating_boundary_phi_ranges[iseg][2])
                    if (d.floating_boundary_z_ranges[iseg][1] <= z <= d.floating_boundary_z_ranges[iseg][2])
                        if d.floating_boundary_types[iseg]=="Tubs"
                            rv = :floating_boundary
                            i=iseg
                        else
                            analytic_r = geom_round(analytical_taper_r_from_φz(φ,z,
                                d.floating_boundary_types[iseg],
                                d.floating_boundary_r_ranges[iseg],
                                d.floating_boundary_phi_ranges[iseg],
                                d.floating_boundary_z_ranges[iseg]
                            ))
                            if isapprox(analytic_r, r)
                                rv = :floating_boundary
                                i=iseg
                            end
                        end
                    end
                end
            end
        end
    end
    return rv, i
end

function analytical_taper_r_from_φz(φ,z,orientation,r_bounds,φ_bounds,z_bounds)
    r=0
    angle = atan(abs(r_bounds[2]-r_bounds[1])/abs(z_bounds[2]-z_bounds[1]))
    if orientation == "c//"
        r = r_bounds[2] - tan(angle)*(z-minimum(z_bounds))
    elseif orientation == "/c"
        r = r_bounds[1] + tan(angle)*(z-minimum(z_bounds))
    end
    r
end



function get_segment_idx(d::SolidStateDetector,r::Real,φ::Real,z::Real)
    digits=6
    for iseg in 1:size(d.segment_bias_voltages,1)
        if d.borehole_modulation == true && iseg == d.borehole_segment_idx
             x = (geom_round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)) <= r <= geom_round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)))
        elseif d.borehole_modulation == true && iseg == d.borehole_bot_segment_idx
             x = (d.segmentation_r_ranges[iseg][1] <= r <= geom_round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)))
        elseif d.borehole_modulation == true && iseg == d.borehole_top_segment_idx
            x = (geom_round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)) <= r <= d.segmentation_r_ranges[iseg][2])
        else
            x = (d.segmentation_r_ranges[iseg][1] <= r <= d.segmentation_r_ranges[iseg][2])
        end
        if x && (d.segmentation_phi_ranges[iseg][1] <= φ <= d.segmentation_phi_ranges[iseg][2]) && (d.segmentation_z_ranges[iseg][1] <= z <= d.segmentation_z_ranges[iseg][2])
            return iseg
        else
            nothing
        end
    end
    return -1
end
function get_segment_idx(d::SolidStateDetector,r::Real,φ::Real,z::Real,rs::Vector{<:Real})
    digits=6
    idx_r_closest_gridpoint_to_borehole = searchsortednearest(rs, d.borehole_radius+d.borehole_ModulationFunction(φ))
    idx_r = findfirst(x->x==r,rs)
    bore_hole_r = rs[idx_r_closest_gridpoint_to_borehole]
    for iseg in 1:size(d.segment_bias_voltages,1)
        if d.borehole_modulation == true && iseg == d.borehole_segment_idx
             # x = (round(d.segmentation_r_ranges[iseg][1]+d.borehole_ModulationFunction(φ)-ϵ,digits=digits) <= r <= round(d.segmentation_r_ranges[iseg][2]+d.borehole_ModulationFunction(φ)+ϵ,digits=digits))
             x = (idx_r_closest_gridpoint_to_borehole == idx_r)
        elseif d.borehole_modulation == true && iseg == d.borehole_bot_segment_idx
             x = (d.segmentation_r_ranges[iseg][1] <= r <= geom_round(bore_hole_r))
             # x = idx_r_closest_gridpoint_to_borehole == idx_r
        elseif d.borehole_modulation == true && iseg == d.borehole_top_segment_idx
            x = (geom_round(bore_hole_r) <= r <= d.segmentation_r_ranges[iseg][2])
            # x = idx_r_closest_gridpoint_to_borehole == idx_r
        else
            x = (d.segmentation_r_ranges[iseg][1] <= r <= d.segmentation_r_ranges[iseg][2])
        end
        if x && (d.segmentation_phi_ranges[iseg][1] <= φ <= d.segmentation_phi_ranges[iseg][2]) && (d.segmentation_z_ranges[iseg][1] <= z <= d.segmentation_z_ranges[iseg][2])
            return iseg
        else
            nothing
        end
    end
    return -1
end

# get_potential
function get_boundary_value(d::SolidStateDetector{T}, r::Real, φ::Real, z::Real, rs::Vector{<:Real})::T where {T <: AbstractFloat}
    if d.borehole_modulation==true
        res::get_precision_type(d)=0.0
        try
            res = d.segment_bias_voltages[ get_segment_idx(d, r, φ, z, rs) ]
        catch
            println("kek")
            @show r, φ, z
            res = d.segment_bias_voltages[end]
        end
        return res
    else
        return d.segment_bias_voltages[ get_segment_idx(d, r, φ, z) ]
    end
end


function get_charge_density(detector::SolidStateDetector{T}, r::Real, φ::Real, z::Real)::T where {T <: AbstractFloat}
    top_net_charge_carrier_density::T = detector.charge_carrier_density_top * 1e10 * 1e6  #  1/cm^3 -> 1/m^3
    bot_net_charge_carrier_density::T = detector.charge_carrier_density_bot * 1e10 * 1e6  #  1/cm^3 -> 1/m^3
    slope::T = (top_net_charge_carrier_density - bot_net_charge_carrier_density) / detector.crystal_length
    ρ::T = bot_net_charge_carrier_density + z * slope
    return ρ
end


function get_ρ_and_ϵ(pt::CylindricalPoint{T}, ssd::SolidStateDetector{T})::Tuple{T, T} where {T <: AbstractFloat}
    if in(pt, ssd)
        ρ::T = get_charge_density(ssd, pt.r, pt.φ, pt.z) * elementary_charge
        ϵ::T = ssd.material_detector.ϵ_r
        return ρ, ϵ
    else
        ρ = 0
        ϵ = ssd.material_environment.ϵ_r
        return ρ, ϵ
    end
end
function set_pointtypes_and_fixed_potentials!(pointtypes::Array{PointType, N}, potential::Array{T, N},
        grid::Grid{T, N, :Cylindrical}, ssd::SolidStateDetector{T}; weighting_potential_contact_id::Union{Missing, Int} = missing)::Nothing where {T <: AbstractFloat, N}

    channels::Array{Int, 1} = if !ismissing(weighting_potential_contact_id)
        ssd.grouped_channels[weighting_potential_contact_id]
    else
        Int[]
    end

    axr::Vector{T} = grid[:r].ticks
    axφ::Vector{T} = grid[:φ].ticks
    axz::Vector{T} = grid[:z].ticks
    for iz in axes(potential, 3)
        z::T = axz[iz]
        for iφ in axes(potential, 2)
            φ::T = axφ[iφ]
            for ir in axes(potential, 1)
                r::T = axr[ir]
                pt::CylindricalPoint{T} = CylindricalPoint{T}( r, φ, z )

                if is_boundary_point(ssd, r, φ, z, axr, axφ, axz)
                    pot::T = if ismissing(weighting_potential_contact_id)
                        get_boundary_value( ssd, r, φ, z, axr)
                    else
                        in(ssd.borehole_modulation ? get_segment_idx(ssd, r, φ, z, axr) : get_segment_idx(ssd, r, φ, z), channels) ? 1 : 0
                    end
                    potential[ ir, iφ, iz ] = pot
                    pointtypes[ ir, iφ, iz ] = zero(PointType)
                elseif in(pt, ssd)
                    pointtypes[ ir, iφ, iz ] += pn_junction_bit
                end

            end
        end
    end
    nothing
end




function json_to_dict(inputfile::String)::Dict
    parsed_json_file = Dict()
    open(inputfile,"r") do f
        global parsed_json_file
        dicttext = readstring(f)
        parsed_json_file = JSON.parse(dicttext)
    end
    return parsed_json_file
end
function bounding_box(d::SolidStateDetector)
    T = get_precision_type(d)
    (
    r_range = ClosedInterval{T}(0.0,d.crystal_radius),
    φ_range = ClosedInterval{T}(0.0,2π),
    z_range = ClosedInterval{T}(0.0,d.crystal_length)
    )
end
include("plot_recipes.jl")
