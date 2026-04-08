# NeoNEXUS.jl

**NeoNEXUS** is a Julia package for multi-scale, Hessian-based morphology analysis on 3D scalar fields. It exposes low-level tools for filters, Hessian eigenvalues, feature signatures, thresholding, and connected components, together with higher-level MMF and NEXUS+ runners.

## Installation

```julia
using Pkg
Pkg.add("NeoNEXUS")
```

## Quick Start

```julia
using NeoNEXUS

N = 64
density = abs.(randn(Float32, N, N, N)) .+ 1f0
scales = [sqrt(2.0)^n for n in 1:4]

runner = NEXUSPlus(N, scales)
thresholds = runner(density)

println(thresholds)
println(sum(runner.filament.thresholdMap))
```

## What the Package Provides

- **Features**: [`SheetFeature`](@ref), [`LineFeature`](@ref), and [`NodeFeature`](@ref) compute wall, filament, and node signatures.
- **Filters**: [`GaussianFourierFilter`](@ref) and [`TopHatFourierFilter`](@ref) smooth fields in Fourier space.
- **Thresholding**: flat, mass-based, density-based, `deltaMSquaredThreshold!`, and connected-component threshold helpers are available.
- **Connected components**: post-processing utilities identify, label, and prune 6-connected structures.
- **Runners**: [`MMFClassic`](@ref), [`NEXUSPlus`](@ref), and [`runMultithreaded`](@ref) support end-to-end workflows.

## Usage Notes

- The package currently operates on 3D arrays.
- `NEXUSPlus(gridSize, scales)` is a convenience constructor for cubic grids.
- `run(runner::NEXUSPlus, densityField)` normalizes the density field internally.
- Feature objects and runners are stateful: their `significanceMap` and `thresholdMap` arrays are stored on the structs and reused across calls.

## Next Steps

- See [Workflow](workflow.md) for the high-level processing stages and the difference between `MMFClassic` and `NEXUSPlus`.
- See [API Reference](api.md) for the exported types and functions.
