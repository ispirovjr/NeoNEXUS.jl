# Thresholding functions for morphological feature classification
# Each function operates on feature.significanceMap and populates feature.thresholdMap


"""
    flatThreshold!(feature, threshold)

Apply a flat (hard) threshold to the feature's significance map.
Voxels with signature ≥ threshold are marked as 1.0 in thresholdMap, otherwise 0.0.

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `threshold::Real` — Cutoff value; voxels ≥ this value pass

# Returns
- The computed threshold value (same as input)
"""
function flatThreshold!(
    feature::AbstractMorphologicalFeature,
    threshold::Real
)
    sigMap = feature.significanceMap
    thresMap = feature.thresholdMap

    @inbounds for I in eachindex(sigMap)
        thresMap[I] = sigMap[I] >= threshold ? 1f0 : 0f0
    end

    return Float32(threshold)
end


"""
    volumeThreshold!(feature, fraction=0.5)

Find threshold such that `fraction` of non-zero voxels pass.
Given N voxels with signature > 0, we find threshold T such that
the number of voxels with signature ≥ T is approximately `fraction * N`.

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `fraction::Real` — Fraction of non-zero voxels to retain (default 0.5 = 50%)

# Returns
- The computed threshold value
"""
function volumeThreshold!(
    feature::AbstractMorphologicalFeature,
    fraction::Real=0.5
)
    sigMap = feature.significanceMap
    thresMap = feature.thresholdMap

    # Collect non-zero values
    nonzeroVals = filter(x -> x > 0, vec(sigMap))

    if isempty(nonzeroVals)
        fill!(thresMap, 0f0)
        return 0f0
    end

    # Sort descending to find the threshold at the desired percentile
    sort!(nonzeroVals, rev=true)

    # Number of voxels to keep
    nKeep = max(1, round(Int, fraction * length(nonzeroVals)))

    # Threshold is the value of the nKeep-th element (last one to pass)
    threshold = nonzeroVals[nKeep]

    # Apply threshold
    @inbounds for I in eachindex(sigMap)
        thresMap[I] = sigMap[I] >= threshold ? 1f0 : 0f0
    end

    return Float32(threshold)
end


"""
    massThreshold!(feature, densityField, fraction=0.5)

Find threshold such that `fraction` of total mass (density-weighted) passes.
Mass per voxel = signature × density. We find threshold T such that
voxels with signature ≥ T contain `fraction` of total mass.

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `densityField::AbstractArray{<:Real,3}` — Density field for mass weighting
- `fraction::Real` — Fraction of total mass to retain (default 0.5 = 50%)

# Returns
- The computed threshold value
"""
function massThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    fraction::Real=0.5
)
    sigMap = feature.significanceMap
    thresMap = feature.thresholdMap

    @assert size(sigMap) == size(densityField) "Signature and density fields must have same size"

    # Collect (signature, mass) pairs for non-zero signatures
    # Mass = signature × density (treating negative density as zero contribution)
    pairs = Vector{Tuple{Float32,Float32}}()
    totalMass = 0f0

    @inbounds for I in eachindex(sigMap)
        sig = sigMap[I]
        if sig > 0
            ρ = max(0f0, Float32(densityField[I]))
            mass = sig * ρ
            push!(pairs, (sig, mass))
            totalMass += mass
        end
    end

    if isempty(pairs) || totalMass ≤ 0
        fill!(thresMap, 0f0)
        return 0f0
    end

    # Sort by signature descending (highest signature first)
    sort!(pairs, by=first, rev=true)

    # Accumulate mass until we reach the target fraction
    targetMass = fraction * totalMass
    accumulatedMass = 0f0
    threshold = 0f0

    for (sig, mass) in pairs
        accumulatedMass += mass
        threshold = sig
        if accumulatedMass >= targetMass
            break
        end
    end

    # Apply threshold
    @inbounds for I in eachindex(sigMap)
        thresMap[I] = sigMap[I] >= threshold ? 1f0 : 0f0
    end

    return Float32(threshold)
end
