# Connected Components for Morphological Features
# Identifies and stores contiguous regions of non-zero signature

"""
A connected component representing a contiguous region of non-zero signature voxels.

# Fields
- `id::Int` — Unique identifier for this component
- `voxels::Vector{CartesianIndex{3}}` — Indices of voxels in this component
- `maxSignature::Float32` — Maximum signature value in this component
"""
struct ConnectedComponent
    id::Int
    voxels::Vector{CartesianIndex{3}}
    maxSignature::Float32
end

# Convenience accessors
Base.length(cc::ConnectedComponent) = length(cc.voxels)
volume(cc::ConnectedComponent) = length(cc.voxels)


"""
Find all connected components (contiguous regions of non-zero signature) in a 3D signature field.
Uses 6-connectivity (face-adjacent voxels only, not diagonal).

# Arguments
- `signatureMap`: 3D signature field

# Returns
- `Vector{ConnectedComponent}`: List of connected components, sorted in descending order by maximum signature 
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
        maxSig = 0f0

        queue = [I]
        visited[I] = true

        while !isempty(queue)
            current = popfirst!(queue)
            push!(voxels, current)
            sig = Float32(signatureMap[current])
            if sig > maxSig
                maxSig = sig
            end

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

        push!(components, ConnectedComponent(componentId, voxels, maxSig))
    end

    # Sort by maximum signature descending (most significant first)
    sort!(components, by=cc -> cc.maxSignature, rev=true)

    return components
end


"""
Create a label map where each voxel is assigned its component ID (0 for background).

# Returns
- `labelMap::Array{Int32,3}`: 3D array of component IDs
- `components::Vector{ConnectedComponent}`: List of components
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
Remove small connected components from a signature map by setting their values to zero.

# Returns
- Tuple of (nKept, nPruned, nVoxelsPruned)
"""
function pruneSmallComponents!(signatureMap::AbstractArray{<:Real,3}, minVoxels::Int)
    components = findConnectedComponents(signatureMap)

    nKept = 0
    nPruned = 0
    nVoxelsPruned = 0

    for cc in components
        if length(cc.voxels) < minVoxels
            # Set all voxels in this component to zero
            for voxel in cc.voxels
                signatureMap[voxel] = 0
            end
            nPruned += 1
            nVoxelsPruned += length(cc.voxels)
        else
            nKept += 1
        end
    end

    return (nKept, nPruned, nVoxelsPruned)
end


"""
Remove small connected components from a feature's significance map.
Convenience method that operates on a feature's significanceMap.

# Returns
- Tuple of (nKept, nPruned, nVoxelsPruned)
"""

"""
Remove connected components from a signature map that have a total mass less than `minMass`.
Mass is calculated using the provided `densityField`.

# Arguments
- `signatureMap::AbstractArray{<:Real,3}`: 3D signature field (modified in-place)
- `densityField::AbstractArray{<:Real,3}`: 3D density field for mass calculation
- `minMass::Real`: Minimum total mass for a component to be kept

# Returns
- Tuple of (nKept, nPruned, nVoxelsPruned, massPruned)
"""
function pruneSmallMassComponents!(
    signatureMap::AbstractArray{<:Real,3},
    densityField::AbstractArray{<:Real,3},
    minMass::Real
)
    components = findConnectedComponents(signatureMap)

    nKept = 0
    nPruned = 0
    nVoxelsPruned = 0
    massPruned = 0.0

    @inbounds for cc in components
        currentMass = 0.0
        for idx in cc.voxels
            currentMass += densityField[idx]
        end

        if currentMass < minMass
            signatureMap[cc.voxels] .= 0
            nPruned += 1
            nVoxelsPruned += length(cc.voxels)
            massPruned += currentMass
        else
            nKept += 1
        end
    end

    return (nKept, nPruned, nVoxelsPruned, massPruned)
end

function pruneSmallMassComponents!(feature::AbstractMorphologicalFeature, densityField::AbstractArray{<:Real,3}, minMass::Real)
    return pruneSmallMassComponents!(feature.significanceMap, densityField, minMass)
end


