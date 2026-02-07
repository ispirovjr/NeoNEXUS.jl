#!/usr/bin/env julia
#=
NeoNEXUS Quick Start Demo
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
densityRaw = max.(densityRaw, 1f-6)  # Ensure positive for log filtering

gridSize = size(densityRaw, 1)
meanDensity = mean(densityRaw)
normDensity = densityRaw ./ meanDensity

scales = [sqrt(2)^n for n in 2:2:8]

runner = NEXUSPlus(gridSize, scales)

thresholds = NeoNEXUS.run(runner, normDensity)

slice = log10.(densityRaw[:, :, gridSize÷2])

cl = extrema(slice)
nx, ny = size(slice)
x = 1:nx
y = 1:ny

heatmap(x, y, slice, aspect_ratio=:equal, color=:acton, title="Density Field (Slice z=$(gridSize÷2))", clim=cl, colorbar=false, axis=false, grid=false)

contour!(x, y, runner.wall.thresholdMap[:, :, gridSize÷2] .* 100, levels=1, color=:blue, linewidth=2, fill=false) # *100 to make it fit in the colorbar
contour!(x, y, runner.filament.thresholdMap[:, :, gridSize÷2] .* 100, levels=1, color=:green, linewidth=2, fill=false)
contour!(x, y, runner.node.thresholdMap[:, :, gridSize÷2] .* 100, levels=1, color=:red, linewidth=2, fill=false)



