"""
Attempted recreation of the Classic Multi-scale Morphology Filter from Aragón-Calvo et al. (2007).
Iterates over scales, applies filtering, computes Hessian once per scale,
and evaluates all feature signatures with cache reuse.

# Fields
- `filter: Scale-space filter
- `features: Feature detectors (Sheet, Line, Node)
- `scales: Smoothing scales to iterate over
"""
struct MMFClassic
    filter::AbstractScaleFilter
    features::Vector{AbstractFeature}
    scales::Vector{Float64}
end


"""
Execute the MMF pipeline on a density field.
Loop over scales, compute signatures with linear filter, aggregate via max-pooling.
Threshold via number of connected components declining.
"""
function run(runner::MMFClassic, densityField::AbstractArray{<:Real,3})
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

    # === Hierarchical Thresholding ===
    # Process features in order, masking each by all previous thresholds
    thresholds = Float32[]

    for (i, feature) in enumerate(runner.features)
        # Mask by all previously thresholded features
        for j in 1:(i-1)
            maskSignatureMap!(feature, runner.features[j])
        end

        # Apply plateau threshold (where component elimination rate stabilizes)
        thresh = componentErosionPlateauThreshold!(feature)
        push!(thresholds, thresh)
    end

    return thresholds
end


"""
Implemenntation of NEXUS+ from Cautun et al. (2013).

# Fields
- `filter: Scale-space filter
- `node: Node feature detector
- `filament: Filament feature detector
- `wall: Wall feature detector
- `scales: Smoothing scales to iterate over
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

Loop over scales, compute node signature via linear filter, 
wall and filament signatures with log filter, aggregate signatures via max-pooling.
Uses cachine between the latter two

Thresholding is done as follows:
- Nodes: ensure collapse in 50% of components (density >= 370x mean)
- Filaments: ΔM² maximum
- Walls: ΔM² maximum

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
        linFiltered = runner.filter(normDensity, scale, runner.node)
        runner.node(linFiltered, nothing, None)

        # Filaments: log filter, write cache
        logFiltered = runner.filter(normDensity, scale, runner.filament)
        runner.filament(logFiltered, cache, Write)

        # Walls: log filter, read filament cache
        runner.wall(logFiltered, cache, Read)
    end

    # === Thresholding (hierarchical with signature masking) ===

    # 1. Nodes: find threshold where 50% of components (even empty ones) have >= 370x density
    nodeThres = findComponentPercentageThreshold!(
        runner.node,
        normDensity,
        370.0,
        0.50;
        excludeEmpty=false
    )

    # 2. Filaments: mask by nodes, then ΔM² threshold
    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, normDensity)

    # 3. Walls: mask by nodes and filaments, then ΔM² threshold
    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, normDensity)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end
