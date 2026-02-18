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
        # Apply filter at this scale, with R² signature scaling 
        filteredField = runner.filter(densityField, scale) .* scale^2

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

function NEXUSPlus(gridSize::Int, scales::Vector{Float64})
    kx = ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π
    sheet = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    line = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    node = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)

    return NEXUSPlus(GaussianFourierFilter((gridSize, gridSize, gridSize)), node, line, sheet, scales)
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
    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    gridSize = size(runner.node.significanceMap)
    cache = HessianEigenCache(gridSize...)

    # === Scale Loop ===
    # R² signature scaling applied after filtering (Cautun et al. 2013)
    for scale in runner.scales
        R² = scale^2

        linFiltered = runner.filter(normDensity, scale, runner.node) .* R²
        runner.node(linFiltered, nothing, None)

        logFiltered = runner.filter(normDensity, scale, runner.filament) .* R²
        runner.filament(logFiltered, cache, Write)

        runner.wall(logFiltered, cache, Read)
    end

    # === Thresholding (hierarchical with signature masking) ===
    nodeThres = findComponentPercentageThreshold!(
        runner.node,
        normDensity,
        370.0,
        0.50;
        excludeEmpty=false
    )

    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, normDensity)

    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, normDensity)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end


"""
Multithreaded variant of the NEXUS+ pipeline.

Parallelizes the scale loop using `Threads.@threads`. Each scale gets its own
`HessianEigenCache` and signature arrays, avoiding all race conditions.
The per-scale signatures are reduced via element-wise max into the feature's
`significanceMap` after the parallel loop.

Thresholding is identical to the sequential `run`.

!!! note
    Requires Julia to be started with multiple threads (`julia --threads=N`).
    FFTW internal threading is disabled to avoid thread-safety issues.
"""
function runMultithreaded(runner::NEXUSPlus, densityField::AbstractArray{<:Real,3})
    # Disable FFTW internal threading for thread safety
    FFTW.set_num_threads(1)

    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    gridSize = size(runner.node.significanceMap)
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

    # === Parallel scale loop ===
    Threads.@threads for idx in 1:nScales
        scale = runner.scales[idx]
        R² = scale^2


        linFiltered = runner.filter(normDensity, scale, runner.node) .* R²
        localCache = computeHessianEigenvalues(linFiltered, runner.node.kx, runner.node.ky, runner.node.kz)
        nodeSigs[idx] .= computeSignature(runner.node, localCache)

        logFiltered = runner.filter(normDensity, scale, runner.filament) .* R²
        computeHessianEigenvalues!(logFiltered, runner.filament.kx, runner.filament.ky, runner.filament.kz, localCache)
        filaSigs[idx] .= computeSignature(runner.filament, localCache)
        wallSigs[idx] .= computeSignature(runner.wall, localCache)
    end

    for idx in 1:nScales
        @. runner.node.significanceMap = max(runner.node.significanceMap, nodeSigs[idx])
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, filaSigs[idx])
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, wallSigs[idx])
    end


    # 1. Nodes: find threshold where 50% of components have >= 370x density
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
