# NeoNEXUS

[![Julia](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Modernized and optimized implementation of MMF/NEXUS in Julia** for detecting cosmic web morphological structures in 3D density fields.

## Overview

NeoNEXUS implements the **Multi-scale Morphology Filter (MMF)** algorithm, also known as **NEXUS**, which identifies coherent structures in the cosmic web:

- **Sheets/Walls** — Planar overdensities where matter flows from voids
- **Filaments** — Cylindrical structures connecting clusters
- **Nodes/Clusters** — Spherical regions at intersection points

The method uses **Hessian matrix eigenvalue analysis** of the density field at multiple smoothing scales to classify local geometry.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/NeoNEXUS.jl")
```

Or for development:
```julia
Pkg.develop(path="/path/to/NeoNEXUS")
```

## Quick Start

```julia
using NeoNEXUS
using FFTW

# Grid setup
N = 64
kx = ky = kz = fftfreq(N) .* 2π

# Create a sheet (wall) detector
sheet = SheetFeature((N, N, N), kx, ky, kz)

# Load or generate your 3D density field
densityField = randn(Float32, N, N, N)

# Compute signature map
signatureMap = sheet(densityField)

# The feature's significanceMap accumulates max across scales
println("Max sheet significance: ", maximum(sheet.significanceMap))
```

## How It Works

### 1. Hessian Eigenvalue Computation

For a scalar field δ(x), compute the Hessian:

```
H_ij = ∂²δ / ∂x_i ∂x_j
```

Using Fourier derivatives: `H_ij(k) = -k_i k_j δ̂(k)`

The eigenvalues λ₁ ≤ λ₂ ≤ λ₃ encode local curvature.

### 2. Morphological Classification

| Feature | Eigenvalue Signature |
|---------|---------------------|
| Sheet   | λ₁ < 0, \|λ₂/λ₁\| < 1, \|λ₃/λ₁\| < 1 |
| Filament| λ₁ < 0, λ₂ < 0 |
| Node    | λ₁ < 0, λ₂ < 0, λ₃ < 0 |

### 3. Multi-scale Analysis

Apply Gaussian smoothing at multiple scales, compute signatures at each, and aggregate via voxel-wise maximum.

## API Reference

### Features

```julia
# Sheet detector (walls)
sheet = SheetFeature(gridSize::Tuple, kx, ky, kz)

# Filament detector (in development)
line = LineFeature(sigMap, respMap, kx, ky, kz)

# Node detector (in development)
node = NodeFeature(sigMap, respMap, kx, ky, kz)
```

### Hessian Computation

```julia
# Compute eigenvalues (allocating)
cache = computeHessianEigenvalues(field, kx, ky, kz)

# Compute eigenvalues (in-place)
computeHessianEigenvalues!(field, kx, ky, kz, cache)
```

### Cache Modes

```julia
# Direct computation (no caching)
result = feature(field)

# Read from pre-computed cache
result = feature(field, cache, Read)

# Compute and write to cache
result = feature(field, cache, Write)
```

## Project Structure

```
NeoNEXUS/
├── src/
│   ├── NeoNEXUS.jl    # Module entry point
│   ├── Types.jl       # Abstract types and enums
│   ├── Hessian.jl     # Eigenvalue computation
│   ├── Features.jl    # Morphological feature detectors
│   ├── Filters.jl     # Scale-space filters
│   └── Runner.jl      # Pipeline orchestrator
└── test/
    ├── runtests.jl    # Test entry point
    └── test_*.jl      # Modular test files
```

## Running Tests

```julia
using Pkg
Pkg.test("NeoNEXUS")
```
 