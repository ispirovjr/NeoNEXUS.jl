# Connected Components for Morphological Features
# Identifies and stores contiguous regions of non-zero signature

"""
A connected component representing a contiguous region of non-zero signature voxels.

# Fields
- `id::Int` — Unique identifier for this component
- `voxels::Vector{CartesianIndex{3}}` — Indices of voxels in this component
- `totalSignature::Float32` — Sum of signature values in this component
"""
struct ConnectedComponent
    id::Int
    voxels::Vector{CartesianIndex{3}}
    totalSignature::Float32
end

# Convenience accessors
Base.length(cc::ConnectedComponent) = length(cc.voxels)
volume(cc::ConnectedComponent) = length(cc.voxels)


"""
Find all connected components (contiguous regions of non-zero signature) in a 3D signature field.
Uses 6-connectivity (face-adjacent voxels only, not diagonal).

# Arguments
- `signatureMap::AbstractArray{<:Real,3}` — 3D signature field

# Returns
- `Vector{ConnectedComponent}` — List of connected components, sorted in descending order by total signature 
"""
function findConnectedComponents(signatureMap::AbstractArray{<:Real,3})
    dims = size(signatureMap)
    visited = falses(dims)
    components = ConnectedComponent[]

    # 6-connectivity offsets (face-adjacent only)
    offsets = [
        CartesianIndex(1, 0, 0), CartesianIndex(-1, 0, 0),
        CartesianIndex(0, 1, 0), CartesianIndex(0, -1, 0),
        CartesianIndex(0, 0, 1), CartesianIndex(0, 0, -1)
    ]

    componentId = 0

    @inbounds for I in CartesianIndices(signatureMap)
        # Skip if already visited or zero signature
        if visited[I] || signatureMap[I] <= 0
            continue
        end

        # Start new component with BFS
        componentId += 1
        voxels = CartesianIndex{3}[]
        totalSig = 0f0

        queue = [I]
        visited[I] = true

        while !isempty(queue)
            current = popfirst!(queue)
            push!(voxels, current)
            totalSig += Float32(signatureMap[current])

            # Check all 6-connected neighbors
            for offset in offsets
                neighbor = current + offset

                # Bounds check
                if checkbounds(Bool, signatureMap, neighbor) &&
                   !visited[neighbor] &&
                   signatureMap[neighbor] > 0
                    visited[neighbor] = true
                    push!(queue, neighbor)
                end
            end
        end

        push!(components, ConnectedComponent(componentId, voxels, totalSig))
    end

    # Sort by total signature descending (most significant first)
    sort!(components, by=cc -> cc.totalSignature, rev=true)

    return components
end


"""
Create a label map where each voxel is assigned its component ID (0 for background).

# Returns
- `labelMap::Array{Int32,3}` — 3D array of component IDs
- `components::Vector{ConnectedComponent}` — List of components
"""
function labelConnectedComponents(signatureMap::AbstractArray{<:Real,3})
    components = findConnectedComponents(signatureMap)
    labelMap = zeros(Int32, size(signatureMap))

    for cc in components
        for voxel in cc.voxels
            labelMap[voxel] = Int32(cc.id)
        end
    end

    return labelMap, components
end


"""
    componentAverageDensity(component, densityField)

Compute the average density of a connected component.
Average density = total mass / volume = sum(density at voxels) / number of voxels.
"""
function componentAverageDensity(cc::ConnectedComponent, densityField::AbstractArray{<:Real,3})
    totalMass = 0f0
    @inbounds for voxel in cc.voxels
        totalMass += Float32(densityField[voxel])
    end
    return totalMass / length(cc)
end


"""
Threshold based on connected component average density.

This function:
1. Finds connected components in the signature field
2. Computes average density per component (total mass / volume)
3. Keeps all components where average density ≥ densityCutoff
4. Marks voxels of qualifying components in thresholdMap

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `densityField::AbstractArray{<:Real,3}` — Density field for computing component mass
- `densityCutoff::Real` — Minimum average density for a component to qualify

# Returns
- Tuple of (number of qualifying components, total components)
"""
function componentDensityThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    densityCutoff::Real
)
    sigMap = feature.significanceMap
    thresMap = feature.thresholdMap

    @assert size(sigMap) == size(densityField) "Signature and density fields must have same size"

    # Find connected components
    components = findConnectedComponents(sigMap)

    if isempty(components)
        fill!(thresMap, 0f0)
        return (0, 0)
    end

    # Mark voxels of components that meet the density cutoff
    fill!(thresMap, 0f0)
    nQualifying = 0

    for cc in components
        avgDensity = componentAverageDensity(cc, densityField)
        if avgDensity >= densityCutoff
            nQualifying += 1
            for voxel in cc.voxels
                thresMap[voxel] = 1f0
            end
        end
    end

    return (nQualifying, length(components))
end

