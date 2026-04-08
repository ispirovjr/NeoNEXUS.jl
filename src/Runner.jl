"""
    MMFClassic

Classic Multi-scale Morphology Filter pipeline (Aragón-Calvo et al. 2007).

Iterates over scales with linear Gaussian filtering, computes Hessian once per
scale, and evaluates all feature signatures with cache reuse. Threshold via
component erosion plateau.

# Fields
- `filter::AbstractScaleFilter` — scale-space filter
- `features::Vector{AbstractFeature}` — feature detectors (Sheet, Line, Node)
- `scales::Vector{Float64}` — smoothing scales
"""
struct MMFClassic
    filter::AbstractScaleFilter
    features::Vector{AbstractFeature}
    scales::Vector{Float64}
end


"""
    run(runner::MMFClassic, densityField) -> Vector{Float32}

Execute the MMF pipeline: scale loop with cache reuse, then hierarchical
plateau-based thresholding. Returns per-feature threshold values.
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
    NEXUSPlus

NEXUS+ pipeline (Cautun et al. 2013).

Uses linear filtering for nodes and log-Gaussian filtering for filaments/walls.
Threshold: density-based for nodes, ΔM² peak for filaments and walls.

# Fields
- `filter::AbstractScaleFilter` — scale-space filter
- `node::NodeFeature` — node detector
- `filament::LineFeature` — filament detector
- `wall::SheetFeature` — wall detector
- `scales::Vector{Float64}` — smoothing scales
"""
struct NEXUSPlus
    filter::AbstractScaleFilter
    node::NodeFeature
    filament::LineFeature
    wall::SheetFeature
    scales::Vector{Float64}
end

"""
    NEXUSPlus(gridSize::Int, scales)

Convenience constructor for a cubic grid of side `gridSize`.
"""
function NEXUSPlus(gridSize::Int, scales::Vector{Float64})
    kx = FFTW.rfftfreq(gridSize) .* gridSize .* 2π
    ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π
    sheet = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    line = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    node = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)

    return NEXUSPlus(GaussianFourierFilter((gridSize, gridSize, gridSize)), node, line, sheet, scales)
end

"""
    (nexus::NEXUSPlus)(densityField; multithread = false)

Execute the NEXUS+ pipeline. 
Depending on multithread runs [`run(runner, densityField)`](@ref) or [`runMultithreaded(runner, densityField)`](@ref).

Returns `(nodeThres, filamentThres, wallThres)`.
"""

function (nexus::NEXUSPlus)(densityField::AbstractArray{<:Real,3}; multithread=false)
    if multithread
        return runMultithreaded(nexus, densityField)
    else
        return run(nexus, densityField)
    end
end


"""
    run(runner::NEXUSPlus, densityField) -> NamedTuple

Execute the NEXUS+ pipeline on a density field (normalised to mean=1 internally).

Thresholding:
- Nodes: 50 % of components must have density ≥ 370× mean.
- Filaments / Walls: ΔM² maximum.

Returns `(nodeThres, filamentThres, wallThres)`.
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
    runMultithreaded(runner::NEXUSPlus, densityField) -> NamedTuple

Multithreaded variant of [`run`](@ref) for [`NEXUSPlus`](@ref).

Parallelises the scale loop using `Threads.@threads`; each scale gets its own
cache and signature arrays. Thresholding is identical to the sequential version.

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
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
