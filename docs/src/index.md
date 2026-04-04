# NeoNEXUS.jl

**NeoNEXUS** is a modular, high-performance implementation of the Multi-scale Morphology Filter (MMF / NEXUS) in Julia. It detects and classifies multi-scale morphological structures — nodes, filaments, and walls — in 3D scalar fields.

## Installation

```julia
using Pkg
Pkg.add("NeoNEXUS")
```

## Quick Start

```julia
using NeoNEXUS, FFTW

N = 64
kx = ky = kz = fftfreq(N) .* N .* 2π

sheet = SheetFeature((N, N, N), kx, ky, kz)
density = randn(Float32, N, N, N)

sig = sheet(density)
println("Max sheet signature: ", maximum(sig))
```

## Overview

The package provides:

- **Filters** ([`GaussianFourierFilter`](@ref), [`TopHatFourierFilter`](@ref)) for Fourier-space smoothing.
- **Features** ([`SheetFeature`](@ref), [`LineFeature`](@ref), [`NodeFeature`](@ref)) for Hessian-based morphological classification.
- **Thresholding** functions for noise suppression and structure selection.
- **Connected component** analysis for post-processing.
- **Pipelines** ([`MMFClassic`](@ref), [`NEXUSPlus`](@ref)) for end-to-end analysis.
