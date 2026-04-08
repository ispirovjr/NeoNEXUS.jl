#!/usr/bin/env julia
#=
NeoNEXUS multithread demo.

Use `--threads=N` to run with N threads. The speedup is bounded by the number of
scales because `runMultithreaded` parallelizes over the scale loop.
=#

using Pkg
Pkg.activate(@__DIR__)
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using JLD2
using NeoNEXUS

densityPath = joinpath(@__DIR__, "exampleDensity.jld2")
densityRaw = Float32.(load(densityPath)["grid"])

gridSize = size(densityRaw, 1)
scales = [sqrt(2)^n for n in 1:4]

runnerSingle = NEXUSPlus(gridSize, scales)
runnerParallel = NEXUSPlus(gridSize, scales)

resultSingle = @timed run(runnerSingle, densityRaw)
resultParallel = @timed runMultithreaded(runnerParallel, densityRaw)

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
