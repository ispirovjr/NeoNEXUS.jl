# Connected Components for Morphological Features
# Identifies and stores contiguous regions of non-zero signature

"""
    ConnectedComponent

A contiguous region of non-zero signature voxels identified by BFS.

# Fields
- `id::Int` — unique identifier
- `voxels::Vector{CartesianIndex{3}}` — voxel indices in this component
- `maxSignature::Float32` — peak signature value in this component
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
    findConnectedComponents(signatureMap) -> Vector{ConnectedComponent}

Find all connected components (6-connectivity) of non-zero voxels in a 3D field.
Returns components sorted by maximum signature (descending).
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
    labelConnectedComponents(signatureMap) -> (labelMap, components)

Create a label map assigning each voxel its component ID (0 for background).
Returns the `Int32` label array and the list of [`ConnectedComponent`](@ref)s.
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
    componentAverageDensity(component, densityField) -> Float32

Average density of a connected component: `sum(density) / volume`.
"""
function componentAverageDensity(cc::ConnectedComponent, densityField::AbstractArray{<:Real,3})
    totalMass = 0f0
    @inbounds for voxel in cc.voxels
        totalMass += Float32(densityField[voxel])
    end
    return totalMass / length(cc)
end


"""
    pruneSmallComponents!(signatureMap, minVoxels) -> (nKept, nPruned, nVoxelsPruned)

Zero out connected components with fewer than `minVoxels` voxels.
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
    pruneSmallMassComponents!(signatureMap, densityField, minMass) -> (nKept, nPruned, nVoxelsPruned, massPruned)

Zero out connected components whose total mass (sum of `densityField` at component
voxels) is less than `minMass`.
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

"""
    pruneSmallMassComponents!(feature, densityField, minMass)

Convenience method operating on `feature.significanceMap`.
"""
function pruneSmallMassComponents!(feature::AbstractMorphologicalFeature, densityField::AbstractArray{<:Real,3}, minMass::Real)
    return pruneSmallMassComponents!(feature.significanceMap, densityField, minMass)
end
