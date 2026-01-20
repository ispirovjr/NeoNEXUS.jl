# Thresholding functions for morphological feature classification
# Each function operates on feature.significanceMap and populates feature.thresholdMap
# All functions return the threshold value (if a universal one is found)


"""
Apply a flat (hard) threshold to the feature's significance map.
Voxels with signature ≥ threshold are marked as 1.0 in thresholdMap, otherwise 0.0.
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
function thresholdedAverageDensity(
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
Find the signature threshold that achieves a target average density.

Searches for the lowest signature threshold such that the average density
of thresholded voxels is ≥ targetDensity.
"""
function averageDensityThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    targetDensity::Real,
    nBins::Int=100
)
    sigMap = feature.significanceMap

    # Get valid positive signatures
    validMask = sigMap .> 0
    nValid = count(validMask)

    if nValid == 0
        fill!(feature.thresholdMap, 0f0)
        return 0f0
    end

    # Extract signature and density pairs
    flatSig = vec(sigMap)
    flatDen = vec(densityField)

    posIndices = findall(x -> x > 0, flatSig)

    # Sort by signature descending
    sortedIdx = sortperm(flatSig[posIndices], rev=true)
    sortedSig = flatSig[posIndices][sortedIdx]
    sortedDen = flatDen[posIndices][sortedIdx]

    cumMass = cumsum(sortedDen)

    # Start search from beginning (highest signatures)
    optimalIdx = 1
    for i in 1:length(sortedSig)
        avgDensity = cumMass[i] / i
        if avgDensity >= targetDensity
            optimalIdx = i
        else
            # Average density dropped below target, use previous
            break
        end
    end

    # Threshold is the signature value at optimalIdx
    threshold = sortedSig[optimalIdx]

    # Apply the threshold
    flatThreshold!(feature, threshold)

    return Float32(threshold)
end

"""
Calculate the change in squared mass with respect to logarithmic signature:
ΔM² = |d(M²) / d(log S)|

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

    # Flatten and sort
    flatSig = vec(sigMap)
    flatDen = vec(densityField)

    # Filter out <= 0 signatures (somewhat redundant but safe, remove if performance is critical)
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

For more details on procedure see calculateΔM²(feature, densityField, nBins)
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

# Arguments
- feature::AbstractMorphologicalFeature: Feature whose significanceMap will be masked
- mask::AbstractMorphologicalFeature: Feature whose thresholdMap defines the mask

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


"""
Find the signature threshold such that a specific percentage of connected components
meet a density requirement. Generates a threshold map where voxels belonging to
components that meet the density requirement are set to the threshold value. 

excludeEmpty: Components with no remaining voxels at a given 
threshold are excluded from the total count. If false, they count as "failing".
"""
function findComponentPercentageThreshold!(
    feature::AbstractMorphologicalFeature,
    densityField::AbstractArray{<:Real,3},
    densityRequirement::Real,
    targetPercentage::Real;
    nBins::Int=100,
    excludeEmpty::Bool=true
)
    sigMap = feature.significanceMap

    # 1. Identify connected components on the base mask (sig > 0)
    components = findConnectedComponents(sigMap)
    nComponents = length(components)

    if nComponents == 0
        fill!(feature.thresholdMap, 0f0)
        return 0f0
    end

    # 2. Define candidate thresholds
    validSigs = filter(x -> x > 0, sigMap)
    if isempty(validSigs)
        fill!(feature.thresholdMap, 0f0)
        return 0f0
    end

    minSig, maxSig = extrema(validSigs)

    # If min approx max, just return min
    if minSig ≈ maxSig
        flatThreshold!(feature, minSig)
        return minSig
    end

    # Create nBins candidates
    thresholds = range(minSig, maxSig, length=nBins)

    bestThreshold = 0f0
    minDiff = Inf

    # Pre-calculate component voxel indices and densities for speed
    # componentData[i] is a vector of (signature, density) for component i
    componentData = Vector{Vector{Tuple{Float32,Float32}}}(undef, nComponents)

    @inbounds for (i, cc) in enumerate(components)
        data = Vector{Tuple{Float32,Float32}}(undef, length(cc.voxels))
        for (j, voxel) in enumerate(cc.voxels)
            data[j] = (Float32(sigMap[voxel]), Float32(densityField[voxel]))
        end
        componentData[i] = data
    end

    # 3. Iterate thresholds
    prevDiff = Inf
    for thresh in thresholds
        nPassing = 0
        nActive = 0  # Components with at least one voxel at this threshold

        for data in componentData
            # Calculate average density for this component at this threshold
            currentMass = 0f0
            currentCount = 0

            for (sig, rho) in data
                if sig >= thresh
                    currentMass += rho
                    currentCount += 1
                end
            end

            # Handle component based on excludeEmpty setting
            if currentCount > 0
                nActive += 1
                avgDen = currentMass / currentCount
                if avgDen >= densityRequirement
                    nPassing += 1
                end
            elseif !excludeEmpty
                # Count empty components as failing
                nActive += 1
            end
        end

        # Skip if no active components at this threshold
        if nActive == 0
            continue
        end

        fraction = nPassing / nActive
        diff = abs(fraction - targetPercentage)

        if diff < minDiff
            minDiff = diff
            bestThreshold = thresh
        elseif diff > prevDiff
            # Diff started increasing, no point in continuing
            break
        end

        prevDiff = diff
    end

    # 4. Apply best threshold
    flatThreshold!(feature, bestThreshold)

    return Float32(bestThreshold)
end


"""
Threshold based on connected component average density.

Finds connected components and average density per component (total mass / volume).
Keeps all components where average density ≥ densityCutoff.
Marks voxels of qualifying components in thresholdMap.

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


"""
Analyze component erosion by checking how many components survive at each threshold.

# Returns
- logThresholds: Log10 values of bin thresholds
- survivalFractions: Fraction of components surviving at each threshold
"""
function calculateComponentSurvival(
    feature::AbstractMorphologicalFeature,
    nBins::Int=100
)
    sigMap = feature.significanceMap
    components = findConnectedComponents(sigMap)
    nTotal = length(components)

    if nTotal == 0
        return Float64[], Float64[]
    end

    # Use maxSignature field from ConnectedComponent
    componentMaxSigs = Float32[cc.maxSignature for cc in components]

    # Range
    minSig = max(minimum(componentMaxSigs), 1f-6)
    maxSig = maximum(componentMaxSigs)

    if minSig ≈ maxSig
        return [log10(minSig)], [1.0]
    end

    logThresholds = range(log10(minSig), log10(maxSig), length=nBins)
    thresholds = 10 .^ logThresholds

    survivalFractions = zeros(Float64, nBins)
    sortedMaxSigs = sort(componentMaxSigs)

    for i in 1:nBins
        t = thresholds[i]
        # Count components with maxSig >= t
        idx = searchsortedfirst(sortedMaxSigs, t)
        nSurviving = nTotal - idx + 1
        survivalFractions[i] = nSurviving / nTotal
    end

    return logThresholds, survivalFractions
end


"""
Applies a flat threshold where the fraction of surviving components equals `targetPercent`.
Default is 0.5 (50% survival point).
"""
function componentErosionPercentileThreshold!(
    feature::AbstractMorphologicalFeature,
    targetPercent::Real=0.5;
    nBins::Int=100
)
    logThresholds, survivalFractions = calculateComponentSurvival(feature, nBins)

    if isempty(survivalFractions)
        return 0.0f0
    end

    # Find first index where fraction <= target
    idx = findfirst(x -> x <= targetPercent, survivalFractions)

    thresholdLog = if isnothing(idx)
        # Fallback if never reaches target (e.g., if even at max, survival > target)
        logThresholds[end]
    elseif idx == 1
        logThresholds[1]
    else
        # Interpolate between idx-1 and idx
        f1 = survivalFractions[idx-1]
        f2 = survivalFractions[idx]
        l1 = logThresholds[idx-1]
        l2 = logThresholds[idx]

        # Linear interpolation in log space
        if f1 == f2
            l1
        else
            t = (targetPercent - f1) / (f2 - f1)
            l1 + t * (l2 - l1)
        end
    end

    thresholdVal = 10^thresholdLog
    flatThreshold!(feature, Float32(thresholdVal))
    return Float32(thresholdVal)
end


"""
Applies a threshold at the plateau where component elimination rate stabilizes.
This is detected by finding where the rate of change (1st derivative magnitude) 
drops below `rateThreshold` fraction of the maximum rate.
"""
function componentErosionPlateauThreshold!(
    feature::AbstractMorphologicalFeature;
    nBins::Int=500,
    rateThreshold::Real=0.05
)
    logThresholds, survivalFractions = calculateComponentSurvival(feature, nBins)

    if length(survivalFractions) < 5
        return 0.0f0
    end

    # Calculate 1st derivative (rate of change)
    d1 = zeros(Float64, length(survivalFractions))
    for i in 2:length(survivalFractions)-1
        d1[i] = (survivalFractions[i+1] - survivalFractions[i-1]) / 2
    end

    # Find max magnitude of rate (most negative = steepest decline)
    maxRate = maximum(abs.(d1))

    if maxRate ≈ 0
        return 0.0f0
    end

    # Plateau threshold: where |rate| drops below rateThreshold * maxRate
    targetRate = rateThreshold * maxRate

    # Search from the point of maximum decline forward
    _, steepestIdx = findmin(d1)  # Most negative = steepest decline

    bestIdx = length(logThresholds)  # Default to end

    for i in steepestIdx:length(d1)-1
        if abs(d1[i]) < targetRate
            bestIdx = i
            break
        end
    end

    thresholdVal = 10^logThresholds[bestIdx]
    flatThreshold!(feature, Float32(thresholdVal))
    return Float32(thresholdVal)
end
