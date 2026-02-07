#!/usr/bin/env julia
#=
NeoNEXUS Orchestration Demo
============================
Demonstrates NeoNEXUS orchestration methods
=#

using NeoNEXUS
using HDF5
using FFTW
using Statistics
using Plots

println("="^70)
println("NeoNEXUS Orchestration Demo")
println("="^70)

# Load Data
function loadDensity(filepath::String, datasetName::String="densityfield")
    h5open(filepath, "r") do file
        return read(file, datasetName)
    end
end

densityPath = joinpath(@__DIR__, "exampleDensity.h5")
println("\n1. Loading density from: $densityPath")

densityRaw = Float32.(loadDensity(densityPath))
densityRaw = max.(densityRaw, 1f-6)  # Ensure positive for log filtering

gridSize = size(densityRaw, 1)
meanDensity = mean(densityRaw)
totalMass = sum(densityRaw)
totalVoxels = gridSize^3

println("   Grid size: $(gridSize)³ = $totalVoxels voxels")
println("   Mean density: $(round(meanDensity, sigdigits=4))")
println("   Density range: $(round(minimum(densityRaw), sigdigits=3)) to $(round(maximum(densityRaw), sigdigits=3))")

# Setup k-space grid
kx = ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π

# Setup scales
scales = [sqrt(2)^n for n in 2:2:8]


# =============================================================================
# Method 1: MMFClassic
# =============================================================================
println("\n" * "="^70)
println("METHOD 1: MMFClassic")
println("="^70)
println("""
The classic MMF approach:
- Linear Gaussian filtering at multiple scales
- Hessian eigenvalues cached for efficiency across all features
- Plateau-based thresholding (component erosion stability)
""")

# Normalize density for MMFClassic
densityNormMmf = densityRaw ./ meanDensity

# Create features
sheetMmf = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
lineMmf = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
nodeMmf = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)


# Create runner
gaussianFilter = GaussianFourierFilter((gridSize, gridSize, gridSize))
runnerMmf = MMFClassic(gaussianFilter, AbstractFeature[nodeMmf, lineMmf, sheetMmf], scales)
println("Scales: $scales")
println("\nRunning MMFClassic...")

timeMmf = @elapsed thresholdsMmf = NeoNEXUS.run(runnerMmf, densityNormMmf)

println("\n┌─────────────────────────────────────────────────────────┐")
println("│ MMFClassic Performance                                  │")
println("├─────────────────────────────────────────────────────────┤")
println("│ Total runtime:        $(lpad(round(timeMmf, digits=2), 8)) seconds              │")
println("│ Time per scale:       $(lpad(round(timeMmf/length(scales), digits=2), 8)) seconds              │")
println("│ Throughput:           $(lpad(round(totalVoxels/timeMmf/1e6, digits=2), 8)) Mvoxels/sec          │")
println("└─────────────────────────────────────────────────────────┘")

println("\nThresholds:")
println("   Sheet:    $(round(thresholdsMmf[1], sigdigits=4))")
println("   Filament: $(round(thresholdsMmf[2], sigdigits=4))")
println("   Node:     $(round(thresholdsMmf[3], sigdigits=4))")

nSheetsMmf = count(x -> x > 0, sheetMmf.thresholdMap)
nLinesMmf = count(x -> x > 0, lineMmf.thresholdMap)
nNodesMmf = count(x -> x > 0, nodeMmf.thresholdMap)

println("\nClassified voxels:")
println("   Sheets:    $nSheetsMmf ($(round(100*nSheetsMmf/totalVoxels, digits=2))%)")
println("   Filaments: $nLinesMmf ($(round(100*nLinesMmf/totalVoxels, digits=2))%)")
println("   Nodes:     $nNodesMmf ($(round(100*nNodesMmf/totalVoxels, digits=2))%)")

# =============================================================================
# Method 2: NEXUSPlus
# =============================================================================
println("\n" * "="^70)
println("METHOD 2: NEXUSPlus")
println("="^70)
println("""
The NEXUS+ approach (Cautun et al. 2013):
- LINEAR filtering for Nodes (spherical overdensities)
- LOG filtering for Filaments and Walls (enhances low-density contrast)
- Different thresholding:
  * Nodes: 50% of components with density ≥ 370×mean
  * Filaments/Walls: ΔM² peak (maximum mass growth rate)
- Feature order: Nodes → Filaments → Walls
""")

# Create features
nodeNexus = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)
filamentNexus = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
wallNexus = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)

runnerNexus = NEXUSPlus(gaussianFilter, nodeNexus, filamentNexus, wallNexus, scales)

println("Scales: $(round.(scales, sigdigits=3))")
println("\nRunning NEXUSPlus...")

timeNexus = @elapsed thresholdsNexus = NeoNEXUS.run(runnerNexus, densityRaw)

println("\n┌─────────────────────────────────────────────────────────┐")
println("│ NEXUSPlus Performance                                   │")
println("├─────────────────────────────────────────────────────────┤")
println("│ Total runtime:        $(lpad(round(timeNexus, digits=2), 8)) seconds              │")
println("│ Time per scale:       $(lpad(round(timeNexus/length(scales), digits=2), 8)) seconds              │")
println("│ Throughput:           $(lpad(round(totalVoxels/timeNexus/1e6, digits=2), 8)) Mvoxels/sec          │")
println("└─────────────────────────────────────────────────────────┘")

println("\nThresholds:")
println("   Node:     $(round(thresholdsNexus.nodeThres, sigdigits=4))")
println("   Filament: $(round(thresholdsNexus.filamentThres, sigdigits=4))")
println("   Wall:     $(round(thresholdsNexus.wallThres, sigdigits=4))")

nNodesNexus = count(x -> x > 0, nodeNexus.thresholdMap)
nFilamentsNexus = count(x -> x > 0, filamentNexus.thresholdMap)
nWallsNexus = count(x -> x > 0, wallNexus.thresholdMap)

println("\nClassified voxels:")
println("   Nodes:     $nNodesNexus ($(round(100*nNodesNexus/totalVoxels, digits=2))%)")
println("   Filaments: $nFilamentsNexus ($(round(100*nFilamentsNexus/totalVoxels, digits=2))%)")
println("   Walls:     $nWallsNexus ($(round(100*nWallsNexus/totalVoxels, digits=2))%)")

# =============================================================================
# Visualization: Layered Contour Plots
# =============================================================================
println("\n" * "="^70)
println("VISUALIZATION")
println("="^70)
println("Generating layered contour plots (Walls → Filaments → Nodes)...")

sliceIdx = gridSize ÷ 2
gr(size=(1400, 600))

# Extract slices
sheetSliceMmf = sheetMmf.thresholdMap[:, :, sliceIdx]
lineSliceMmf = lineMmf.thresholdMap[:, :, sliceIdx]
nodeSliceMmf = nodeMmf.thresholdMap[:, :, sliceIdx]

wallSliceNexus = wallNexus.thresholdMap[:, :, sliceIdx]
filamentSliceNexus = filamentNexus.thresholdMap[:, :, sliceIdx]
nodeSliceNexus = nodeNexus.thresholdMap[:, :, sliceIdx]

# Create layered contour plots
# MMFClassic
pMmf = contour(1:gridSize, 1:gridSize, sheetSliceMmf',
    levels=[0.5], linecolor=:blue, linewidth=2,
    title="MMFClassic (z=$sliceIdx)",
    xlabel="x", ylabel="y",
    aspect_ratio=1, legend=:topright,
    label="Walls", xlims=(1, gridSize), ylims=(1, gridSize))

contour!(pMmf, 1:gridSize, 1:gridSize, lineSliceMmf',
    levels=[0.5], linecolor=:green, linewidth=2, label="Filaments")

contour!(pMmf, 1:gridSize, 1:gridSize, nodeSliceMmf',
    levels=[0.5], linecolor=:red, linewidth=2, label="Nodes")

# NEXUSPlus
pNexus = contour(1:gridSize, 1:gridSize, wallSliceNexus',
    levels=[0.5], linecolor=:blue, linewidth=2,
    title="NEXUSPlus (z=$sliceIdx)",
    xlabel="x", ylabel="y",
    aspect_ratio=1, legend=:topright,
    label="Walls", xlims=(1, gridSize), ylims=(1, gridSize))

contour!(pNexus, 1:gridSize, 1:gridSize, filamentSliceNexus',
    levels=[0.5], linecolor=:green, linewidth=2, label="Filaments")

contour!(pNexus, 1:gridSize, 1:gridSize, nodeSliceNexus',
    levels=[0.5], linecolor=:red, linewidth=2, label="Nodes")

pltComparison = plot(pMmf, pNexus,
    layout=(1, 2),
    size=(1400, 600),
    plot_title="Morphological Classification Comparison",
    margin=5Plots.mm)

outputPath = joinpath(@__DIR__, "orchestrationComparison.png")
savefig(pltComparison, outputPath)
println("Comparison plot saved: $outputPath")

# Create 4-slice view for each method
slices = [gridSize ÷ 4, gridSize ÷ 2, 3 * gridSize ÷ 4, gridSize - 1]

# MMFClassic 4-slice
plotsMmf = []
for (i, s) in enumerate(slices)
    sheetS = sheetMmf.thresholdMap[:, :, s]
    lineS = lineMmf.thresholdMap[:, :, s]
    nodeS = nodeMmf.thresholdMap[:, :, s]

    p = contour(1:gridSize, 1:gridSize, sheetS',
        levels=[0.5], linecolor=:blue, linewidth=1.5,
        title="z=$s", xlabel="x", ylabel="y",
        aspect_ratio=1, legend=false,
        xlims=(1, gridSize), ylims=(1, gridSize))
    contour!(p, 1:gridSize, 1:gridSize, lineS',
        levels=[0.5], linecolor=:green, linewidth=1.5)
    contour!(p, 1:gridSize, 1:gridSize, nodeS',
        levels=[0.5], linecolor=:red, linewidth=1.5)
    push!(plotsMmf, p)
end

pltMmfSlices = plot(plotsMmf...,
    layout=(2, 2),
    size=(1000, 1000),
    plot_title="MMFClassic: 4-Slice View (Blue=Walls, Green=Filaments, Red=Nodes)",
    margin=3Plots.mm)

outputPathMmf = joinpath(@__DIR__, "mmfClassicSlices.png")
savefig(pltMmfSlices, outputPathMmf)
println("MMFClassic slices saved: $outputPathMmf")

# NEXUSPlus 4-slice
plotsNexus = []
for (i, s) in enumerate(slices)
    wallS = wallNexus.thresholdMap[:, :, s]
    filamentS = filamentNexus.thresholdMap[:, :, s]
    nodeS = nodeNexus.thresholdMap[:, :, s]

    p = contour(1:gridSize, 1:gridSize, wallS',
        levels=[0.5], linecolor=:blue, linewidth=1.5,
        title="z=$s", xlabel="x", ylabel="y",
        aspect_ratio=1, legend=false,
        xlims=(1, gridSize), ylims=(1, gridSize))
    contour!(p, 1:gridSize, 1:gridSize, filamentS',
        levels=[0.5], linecolor=:green, linewidth=1.5)
    contour!(p, 1:gridSize, 1:gridSize, nodeS',
        levels=[0.5], linecolor=:red, linewidth=1.5)
    push!(plotsNexus, p)
end

pltNexusSlices = plot(plotsNexus...,
    layout=(2, 2),
    size=(1000, 1000),
    plot_title="NEXUSPlus: 4-Slice View (Blue=Walls, Green=Filaments, Red=Nodes)",
    margin=3Plots.mm)

outputPathNexus = joinpath(@__DIR__, "nexusPlusSlices.png")
savefig(pltNexusSlices, outputPathNexus)
println("NEXUSPlus slices saved: $outputPathNexus")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "="^70)
println("PERFORMANCE COMPARISON")
println("="^70)
println("""
┌─────────────────┬──────────────────┬──────────────────┐
│ Metric          │ MMFClassic       │ NEXUSPlus        │
├─────────────────┼──────────────────┼──────────────────┤
│ Runtime         │ $(lpad(round(timeMmf, digits=2), 10))s     │ $(lpad(round(timeNexus, digits=2), 10))s     │
│ Scales          │ $(lpad(length(scales), 10))       │ $(lpad(length(scales), 10))       │
│ Nodes           │ $(lpad(nNodesMmf, 10))       │ $(lpad(nNodesNexus, 10))       │
│ Filaments       │ $(lpad(nLinesMmf, 10))       │ $(lpad(nFilamentsNexus, 10))       │
│ Walls           │ $(lpad(nSheetsMmf, 10))       │ $(lpad(nWallsNexus, 10))       │
└─────────────────┴──────────────────┴──────────────────┘
""")

println("Demo complete!")
println("="^70)
