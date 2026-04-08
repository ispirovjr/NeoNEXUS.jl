#!/usr/bin/env julia
#=
NeoNEXUS Quick Start Demo
=#

using Pkg
Pkg.activate(@__DIR__)
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using NeoNEXUS
using JLD2
using Plots

densityPath = joinpath(@__DIR__, "exampleDensity.jld2")
densityRaw = Float32.(load(densityPath)["grid"])

gridSize = size(densityRaw, 1)
scales = [sqrt(2)^n for n in 1:4]

runner = NEXUSPlus(gridSize, scales)
thresholds = runner(densityRaw)

sliceIndex = gridSize ÷ 2
slice = log10.(densityRaw[:, :, sliceIndex])
slice = (slice .- minimum(slice)) ./ (maximum(slice) - minimum(slice))

heatmap(
    slice;
    aspect_ratio=:equal,
    color=:acton,
    title="Density Field (Slice z=$sliceIndex)",
    colorbar=false,
    axis=false,
    grid=false,
)

contour!(runner.wall.thresholdMap[:, :, sliceIndex], levels=1, color=:blue, linewidth=2, fill=false)
contour!(runner.filament.thresholdMap[:, :, sliceIndex], levels=1, color=:green, linewidth=2, fill=false)
contour!(runner.node.thresholdMap[:, :, sliceIndex], levels=5, color=:red, linewidth=2, fill=false)

println("Thresholds: ", thresholds)
savefig(joinpath(@__DIR__, "demoResult.png"))
