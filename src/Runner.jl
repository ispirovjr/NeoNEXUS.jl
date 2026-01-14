# Orchestrates multi-scale feature detection pipeline
# MMFClassic: Simplest runner - iterate scales, compute features, aggregate via max-pooling

"""
Classic Multi-scale Morphology Filter runner.
Iterates over scales, applies filtering, computes Hessian once per scale,
and evaluates all feature signatures with cache reuse.

# Fields
- `filter::AbstractScaleFilter` — Scale-space filter (e.g., GaussianFourierFilter)
- `features::Vector{AbstractFeature}` — Feature detectors (Sheet, Line, Node)
- `scales::Vector{Float64}` — Smoothing scales to iterate over
"""
struct MMFClassic
    filter::AbstractScaleFilter
    features::Vector{AbstractFeature}
    scales::Vector{Float64}
end


"""
Execute the MMF pipeline on a density field.

# Algorithm
1. For each scale in `runner.scales`:
   - Apply filter at current scale
   - Compute Hessian eigenvalues (cached)
   - Evaluate all feature signatures (reusing cached Hessian)
   - Max-pool signatures into each feature's `significanceMap`

2. After scale loop: apply thresholding (TODO)

# Arguments
- `runner::MMFClassic` — Configured runner with filter, features, and scales
- `densityField::AbstractArray{<:Real,3}` — Input density field
- `method::SignatureMethod` — Signature computation method (Default or NexusPlus)

# Returns
- `nothing` — Results are stored in each feature's `significanceMap`
"""
function run(
    runner::MMFClassic,
    densityField::AbstractArray{<:Real,3},
    method::SignatureMethod=Default
)
    # Get grid size from first feature
    gridSize = size(runner.features[1].significanceMap)

    # Create shared cache for Hessian eigenvalues
    cache = HessianEigenCache(gridSize...)

    # Scale loop
    for scale in runner.scales
        # Apply filter at this scale
        filteredField = runner.filter(densityField, scale)

        # Feature loop with cache reuse
        for (i, feature) in enumerate(runner.features)
            if i == 1
                # First feature: compute and cache Hessian
                feature(filteredField, cache, Write)
            else
                # Subsequent features: reuse cached Hessian
                feature(filteredField, cache, Read)
            end
        end
    end

    # TODO: Apply thresholding to each feature after scale loop
    # Options: flatThreshold!, volumeThreshold!, massThreshold!, etc.
    # May require densityField for mass-based thresholds

    return nothing
end


"""
NEXUS+ Multi-scale Morphology Filter runner.

Uses feature-specific filtering strategies:
- Nodes: Linear Gaussian filter (no cache)
- Filaments: Log-Gaussian filter (writes cache)
- Walls: Log-Gaussian filter (reads filament cache)

Includes hierarchical thresholding:
- Nodes: Average density threshold (370 × normalized mean)
- Filaments: Masked by nodes → ΔM² threshold
- Walls: Masked by nodes+filaments → ΔM² threshold

# Fields
- `filter::AbstractScaleFilter` — Scale-space filter
- `node::NodeFeature` — Node detector
- `filament::LineFeature` — Filament detector  
- `wall::SheetFeature` — Wall detector
- `scales::Vector{Float64}` — Smoothing scales
"""
struct NEXUSPlus
    filter::AbstractScaleFilter
    node::NodeFeature
    filament::LineFeature
    wall::SheetFeature
    scales::Vector{Float64}
end


"""
Execute the NEXUS+ pipeline on a density field.

# Algorithm
1. Normalize density by mean
2. For each scale:
   - Nodes: linear filter → compute signature (no cache)
   - Filaments: log filter → compute + write cache
   - Walls: log filter → read filament cache
3. Thresholding (order matters!):
   - Nodes: averageDensityThreshold!(node, normDensity, 370.0)
   - Filaments: mask by node thresholds → deltaMSquaredThreshold!
   - Walls: mask by node+filament thresholds → deltaMSquaredThreshold!

# Arguments
- `runner::NEXUSPlus` — Configured runner
- `densityField::AbstractArray{<:Real,3}` — Input density field (will be normalized)

# Returns
- Named tuple with threshold values: (nodeThres, filamentThres, wallThres)
"""
function run(runner::NEXUSPlus, densityField::AbstractArray{<:Real,3})
    # Normalize density by mean
    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    # Get grid size
    gridSize = size(runner.node.significanceMap)

    # Create cache for filament→wall reuse
    cache = HessianEigenCache(gridSize...)

    # === Scale Loop ===
    for scale in runner.scales
        # Nodes: linear filter, no cache
        nodeFiltered = runner.filter(normDensity, scale, runner.node)
        runner.node(nodeFiltered, nothing, None)

        # Filaments: log filter, write cache
        filamentFiltered = runner.filter(normDensity, scale, runner.filament)
        runner.filament(filamentFiltered, cache, Write)

        # Walls: log filter, read filament cache
        wallFiltered = runner.filter(normDensity, scale, runner.wall)
        runner.wall(wallFiltered, cache, Read)
    end

    # === Thresholding (hierarchical with signature masking) ===

    # 1. Nodes: average density threshold
    avgDensity, _ = averageDensityThreshold!(runner.node, normDensity, 200.0)
    nodeThres = 200.0f0

    # 2. Filaments: mask by nodes, then ΔM² threshold
    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, normDensity)

    # 3. Walls: mask by nodes and filaments, then ΔM² threshold
    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, normDensity)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end
