# Thresholding functions for morphological feature classification
# Each function operates on feature.significanceMap and populates feature.thresholdMap


"""
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

    # Count non-zero voxels first to pre-allocate
    nNonzero = count(x -> x > 0, sigMap)

    if nNonzero == 0
        fill!(thresMap, 0f0)
        return 0f0
    end

    # Pre-allocate and fill non-zero values
    nonzeroVals = Vector{Float32}(undef, nNonzero)
    idx = 1
    @inbounds for I in eachindex(sigMap)
        if sigMap[I] > 0
            nonzeroVals[idx] = sigMap[I]
            idx += 1
        end
    end

    # Number of voxels to keep
    nKeep = max(1, round(Int, fraction * nNonzero))

    # Use partialsort! to find threshold without full sort (O(n) vs O(n log n))
    partialsort!(nonzeroVals, nKeep, rev=true)
    threshold = nonzeroVals[nKeep]

    # Apply threshold
    @inbounds for I in eachindex(sigMap)
        thresMap[I] = sigMap[I] >= threshold ? 1f0 : 0f0
    end

    return Float32(threshold)
end


"""
Find threshold such that voxels passing the threshold contain `fraction` of total mass.
The signature determines which voxels pass (sig ≥ threshold), and mass = density.
We find threshold T such that: sum(thresholdMap .* density) / sum(density) ≈ fraction

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `densityField::AbstractArray{<:Real,3}` — Density field (mass per voxel)
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

    # Count non-zero signatures and compute total mass in one pass
    nNonzero = 0
    totalMass = 0f0
    @inbounds for I in eachindex(sigMap)
        if sigMap[I] > 0
            nNonzero += 1
            totalMass += max(0f0, Float32(densityField[I]))
        end
    end

    if nNonzero == 0 || totalMass <= 0
        fill!(thresMap, 0f0)
        return 0f0
    end

    # Pre-allocate tuple array (sig, density)
    pairs = Vector{Tuple{Float32,Float32}}(undef, nNonzero)

    idx = 1
    @inbounds for I in eachindex(sigMap)
        if sigMap[I] > 0
            pairs[idx] = (sigMap[I], max(0f0, Float32(densityField[I])))
            idx += 1
        end
    end

    # Sort by signature descending (in-place)
    sort!(pairs, by=first, rev=true)

    # Accumulate mass (density) until we reach the target fraction
    targetMass = fraction * totalMass
    accumulatedMass = 0f0
    threshold = 0f0

    @inbounds for (sig, ρ) in pairs
        accumulatedMass += ρ
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


"""
Apply a density-based threshold: voxels pass if they have non-zero signature 
AND density ≥ densityCutoff.


# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with significanceMap to threshold
- `densityField::AbstractArray{<:Real,3}` — Density field
- `densityCutoff::Real` — Minimum density required for voxel to pass

"""
function massCutoffThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    densityCutoff::Real
)
    sigMap = feature.significanceMap
    thresMap = feature.thresholdMap

    @assert size(sigMap) == size(densityField) "Signature and density fields must have same size"

    @inbounds for I in eachindex(sigMap)
        # Pass if signature > 0 AND density >= cutoff
        thresMap[I] = (sigMap[I] > 0 && densityField[I] >= densityCutoff) ? 1f0 : 0f0
    end

    return Float32(densityCutoff)
end


"""
Compute the average density of voxels that are marked in the threshold map.
Average = sum(density where thresholdMap == 1) / count(thresholdMap == 1)

Returns 0 if no voxels are thresholded.
"""
function thresholdedAverageDensity!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3}
)
    thresMap = feature.thresholdMap

    totalMass = 0f0
    count = 0

    @inbounds for I in eachindex(thresMap)
        if thresMap[I] > 0
            totalMass += Float32(densityField[I])
            count += 1
        end
    end

    return count > 0 ? totalMass / count : 0f0
end


"""
Validate that the already-thresholded voxels have an average density ≥ minAverageDensity.

This function does NOT modify the thresholdMap. It only checks if the current thresholded
region meets the minimum average density requirement.

Use this after applying another threshold (e.g., volumeThreshold!, massThreshold!) to 
verify the resulting selection has sufficient average density.

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature with already-populated thresholdMap
- `densityField::AbstractArray{<:Real,3}` — Density field
- `minAverageDensity::Real` — Minimum required average density

# Returns
- Tuple of (averageDensity, meetsRequirement::Bool)
"""
function averageDensityThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    minAverageDensity::Real
)
    avgDensity = thresholdedAverageDensity!(feature, densityField)
    meetsRequirement = avgDensity >= minAverageDensity

    # If requirement not met, clear the threshold map
    if !meetsRequirement
        fill!(feature.thresholdMap, 0f0)
    end

    return (avgDensity, meetsRequirement)
end

"""
Calculate the change in squared mass with respect to logarithmic signature:
ΔM² = |d(M²) / d(log S)|

# Algorithm
1. Bins the signature range logarithmically.
2. For each bin edge S_i, calculates the total mass M_i of all voxels where signature > S_i.
3. Computes the discrete derivative |(M_{i+1}² - M_i²) / (log(S_{i+1}) - log(S_i))|.

# Arguments
- `feature::AbstractMorphologicalFeature`: Feature with `significanceMap`.
- `densityField::AbstractArray`: Density field for mass calculation.
- `nBins::Int`: Number of logarithmic bins (default: 100).

# Returns
- `logSCenters`: Log10 signature values at bin centers.
- `ΔM²`: Array of calculated derivative values.
"""
function calculateΔM²(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    nBins::Int=100
)
    sigMap = feature.significanceMap

    # Filter valid positive signatures
    validMask = sigMap .> 0
    if count(validMask) == 0
        return Float64[], Float64[]
    end

    fsig = vec(sigMap[validMask])

    # Range for log bins
    minS = minimum(fsig)
    maxS = maximum(fsig)

    if minS ≈ maxS
        return Float64[], Float64[]
    end

    # Log-spaced bin edges
    logEdges = range(log10(minS), log10(maxS), length=nBins + 1)
    edges = 10 .^ logEdges

    # Calculate cumulative mass for each threshold
    # M(> S_i)
    # Optimized approach:
    # 1. Flatten signature and density
    # 2. Sort by signature
    # 3. Compute reverse cumulative sum of density (mass)
    # 4. Interpolate mass at bin edges

    # Flatten and sort
    flatSig = vec(sigMap)
    flatDen = vec(densityField)

    # Filter out <= 0 signatures if we only care about log space
    posIndices = findall(x -> x > 0, flatSig)
    sortedIdx = sortperm(flatSig[posIndices])

    sortedSig = flatSig[posIndices][sortedIdx]
    sortedDen = flatDen[posIndices][sortedIdx]

    # Reverse cumulative sum (Total mass > S)
    # cumsum from end to start
    cumMass = reverse(cumsum(reverse(sortedDen)))

    # We need M(> edge)
    # For each edge, find index in sortedSig where sig > edge
    massAtEdges = zeros(Float64, nBins + 1)

    for i in 1:nBins+1
        threshold = edges[i]
        # Find first index where sortedSig >= threshold
        idx = searchsortedfirst(sortedSig, threshold)

        if idx > length(cumMass)
            massAtEdges[i] = 0.0
        else
            massAtEdges[i] = cumMass[idx]
        end
    end

    # Compute derivative
    # d(M^2) / d(log S)

    logSCenters = zeros(Float64, nBins)
    ΔM² = zeros(Float64, nBins)

    for i in 1:nBins
        logS1 = logEdges[i]
        logS2 = logEdges[i+1]

        M1 = massAtEdges[i]
        M2 = massAtEdges[i+1]

        dM2 = M2^2 - M1^2
        dLogS = logS2 - logS1 # This is constant = step size

        derivative = dM2 / dLogS

        logSCenters[i] = (logS1 + logS2) / 2
        ΔM²[i] = abs(derivative)
    end

    return logSCenters, ΔM²
end

"""
Apply a flat threshold at the signature value that maximizes ΔM² = |d(M²) / d(log S)|.

# Arguments
- `feature::AbstractMorphologicalFeature`
- `densityField::AbstractArray`
- `nBins::Int`: Number of bins for peak detection (default: 100)

# Returns
- `thresholdVal`: The signature threshold value used.
"""
function deltaMSquaredThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    nBins::Int=100
)
    logSCenters, ΔM² = calculateΔM²(feature, densityField, nBins)

    if isempty(ΔM²)
        return 0.0f0
    end

    _, maxIdx = findmax(ΔM²)
    optimalLogS = logSCenters[maxIdx]
    optimalS = 10^optimalLogS

    flatThreshold!(feature, Float32(optimalS))

    return Float32(optimalS)
end


"""
Mask `feature.significanceMap` where `mask.thresholdMap > 0`.
Sets masked voxels' signature to 0 (before thresholding).

Use this to implement hierarchical feature masking:
- First threshold the higher-priority feature (e.g., nodes)
- Then mask lower-priority features by the thresholded regions
- Finally threshold the masked feature

# Arguments
- `feature::AbstractMorphologicalFeature` — Feature whose significanceMap will be masked
- `mask::AbstractMorphologicalFeature` — Feature whose thresholdMap defines the mask

# Returns
- Number of voxels masked
"""
function maskSignatureMap!(feature::AbstractMorphologicalFeature, mask::AbstractMorphologicalFeature)
    nMasked = 0
    @inbounds for I in eachindex(feature.significanceMap)
        if mask.thresholdMap[I] > 0
            feature.significanceMap[I] = 0f0
            nMasked += 1
        end
    end
    return nMasked
end

