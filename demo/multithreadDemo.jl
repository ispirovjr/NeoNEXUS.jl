#!/usr/bin/env julia
#=
NeoNEXUS Quick Start Demo

use --threads=N to run with N threads. Performance caps at N = Nscales (4 in this example)

=#

using NeoNEXUS
using HDF5
using FFTW
using Statistics
using Plots

densityPath = joinpath(@__DIR__, "exampleDensity.h5")
densityRaw = Float32.(h5open(densityPath, "r") do file
    return read(file, "densityfield")
end)
densityRaw = max.(densityRaw, 1f-6)

gridSize = size(densityRaw, 1)
meanDensity = mean(densityRaw)
normDensity = densityRaw ./ meanDensity

scales = [sqrt(2)^n for n in 2:2:8]

runner = NEXUSPlus(gridSize, scales)

resultSingle = @timed NeoNEXUS.run(runner, normDensity)

resultParallel = @timed NeoNEXUS.runMultithreaded(runner, normDensity)

timeSingle = resultSingle[2]
timeParallel = resultParallel[2]

thresholdsSingle = resultSingle[1]
thresholdsParallel = resultParallel[1]


nodeDiff = abs(thresholdsSingle.nodeThres - thresholdsParallel.nodeThres)
filamentDiff = abs(thresholdsSingle.filamentThres - thresholdsParallel.filamentThres)
wallDiff = abs(thresholdsSingle.wallThres - thresholdsParallel.wallThres)

timeDiff = timeSingle - timeParallel
timeRatio = timeSingle / timeParallel

println("="^50)
println("Thresholds:")
println("Node: $(thresholdsSingle.nodeThres) vs $(thresholdsParallel.nodeThres) (diff: $nodeDiff)")
println("Filament: $(thresholdsSingle.filamentThres) vs $(thresholdsParallel.filamentThres) (diff: $filamentDiff)")
println("Wall: $(thresholdsSingle.wallThres) vs $(thresholdsParallel.wallThres) (diff: $wallDiff)")

println("="^50)
println("Time:")
println("Single: $timeSingle")
println("Parallel: $timeParallel")
println("Diff: $timeDiff")
println("Ratio: $timeRatio")
println("="^50)
