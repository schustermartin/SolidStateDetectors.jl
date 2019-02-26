mutable struct SolidStateDetector{T<:AbstractFloat} <: AbstractConfig{T}
    name::String
    class::Symbol
    material_detector::NamedTuple
    material_environment::NamedTuple
    cyclic::T
    mirror_symmetry_φ::Bool

    geometry_unit_factor::Real
    geometry_unit::Unitful.Units

    bulk_type::Symbol
    charge_carrier_density_unit_factor::T
    charge_carrier_density_top::T
    charge_carrier_density_bot::T

    world::Vector{AbstractGeometry{T}}

    crystal_geometry::AbstractGeometry{T}

    crystal_length::T
    crystal_radius::T

    geometry_external_positive::Vector{AbstractGeometry{T}}
    geometry_external_negative::Vector{AbstractGeometry{T}}

    contacts_geometry_unit::Unitful.Units

    n_total_contacts::Int
    contacts::Vector{AbstractContact{T}}

    rs::Vector{T}
    φs::Vector{T}
    zs::Vector{T}

    SolidStateDetector{T}() where {T<:AbstractFloat} = new{T}()
end

function SolidStateDetector{T}(config_file::Dict)::SolidStateDetector{T} where T <: AbstractFloat
    c = SolidStateDetector{T}()
    c.class = Symbol(config_file["class"])
    c.name = config_file["name"]
    c.cyclic = T(deg2rad(config_file["cyclic"]))
    c.mirror_symmetry_φ = config_file["mirror_symmetry_phi"] == "true"

    c.material_environment = material_properties[materials[config_file["geometry"]["world"]["material"]]]
    c.material_detector = material_properties[materials[config_file["geometry"]["crystal"]["material"]]]

    c.geometry_unit = unit_conversion[config_file["geometry"]["unit"]]
    c.world = Geometry(T, config_file["geometry"]["world"]["geometry"], c.geometry_unit)

    geometry_positive = Geometry(T, config_file["geometry"]["crystal"]["geometry"]["positive"], c.geometry_unit)
    haskey(config_file["geometry"]["crystal"]["geometry"],"negative") ? geometry_negative = Geometry(T, config_file["geometry"]["crystal"]["geometry"]["negative"], c.geometry_unit) : geometry_negative = []
    c.crystal_geometry = geometry_positive[1]
    for g in sort!(vcat(geometry_positive,geometry_negative))
        g in geometry_negative ? c.crystal_geometry -= g : nothing
        g in geometry_positive ? c.crystal_geometry += g : nothing
    end

    c.crystal_length = geom_round(T(width(geometry_positive[1].z_interval)))
    c.crystal_radius = geom_round(T(width(geometry_positive[1].r_interval)))

    c.bulk_type = bulk_types[config_file["bulk_type"]]
    c.charge_carrier_density_top = config_file["charge_carrier_density"]["top"]
    c.charge_carrier_density_bot = config_file["charge_carrier_density"]["bot"]

    c.contacts_geometry_unit = unit_conversion[config_file["contacts"]["unit"]]
    haskey(config_file["contacts"], "p") ? p_contacts = Contact{T, :P}[ Contact{T, :P}( contact_dict, c.geometry_unit ) for contact_dict in config_file["contacts"]["p"] ] : nothing
    haskey(config_file["contacts"], "n") ? n_contacts = Contact{T, :N}[ Contact{T, :N}( contact_dict, c.geometry_unit ) for contact_dict in config_file["contacts"]["n"] ] : nothing
    c.contacts = vcat(p_contacts,n_contacts)
    c.n_total_contacts = size(c.contacts,1)

    return c
end

function SolidStateDetector(parsed_dict::Dict)
    SolidStateDetector{Float32}(parsed_dict)
end

function Base.sort!(v::AbstractVector{<:AbstractGeometry})
    hierarchies::Vector{Int} = map(x->x.hierarchy,v)
    v_result::typeof(v) = []
    for idx in sort!(unique!(hierarchies))
        push!(v_result,filter(x->x.hierarchy == hierarchies[idx],v)...)
    end
    return v_result
end

function SolidStateDetector{T}(parsed_dict::Dict) where T
    SolidStateDetector{T}(parsed_dict)
end

function contains(c::SolidStateDetector, point::AbstractCoordinatePoint{T,3})::Bool where T
    for contact in c.contacts
        if point in contact
            return true
        end
    end
    if point in c.crystal_geometry
        return true
    else
        return false
    end
end
# function contains(c::SolidStateDetector, point::AbstractCoordinatePoint{T,3})::Bool where T
#     for v in c.geometry_positive
#         if !(point in v)
#             return false
#         end
#     end
#     for contact in vcat(c.n_contacts, c.p_contacts)
#         if point in contact
#             return true
#         end
#     end
#     if !isempty(c.geometry_negative)
#         for v in c.geometry_negative
#             if point in v
#                 return false
#             end
#         end
#     end
#     return true
# end

function println(io::IO, d::SolidStateDetector{T}) where {T <: AbstractFloat}
    println("________"*d.name*"________\n")
    println("Class: ",d.class)
    println("---General Properties---")
    println("Detector Material: \t $(d.material_detector.name)")
    println("Environment Material: \t $(d.material_environment.name)")
    println("Bulk type: \t\t $(d.bulk_type)")
    # println("Core Bias Voltage: \t $(d.segment_bias_voltages[1]) V")
    # println("Mantle Bias Voltage: \t $(d.segment_bias_voltages[2]) V\n")
    println("---Geometry---")
    println("Outer Crystal Dimensions: ")
    println("Crystal length: \t $(round(d.crystal_length * 1000, sigdigits=6)) mm")
    println("Crystal diameter: \t $(round(2 * d.crystal_radius * 1000, sigdigits=6)) mm")
end

function show(io::IO, d::SolidStateDetector{T}) where {T <: AbstractFloat} println(d) end
function print(io::IO, d::SolidStateDetector{T}) where {T <: AbstractFloat} println(d) end
function display(io::IO, d::SolidStateDetector{T} ) where {T <: AbstractFloat} println(d) end
function show(io::IO,::MIME"text/plain", d::SolidStateDetector) where {T <: AbstractFloat}
    show(io, d)
end
