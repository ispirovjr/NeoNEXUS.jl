# NeoNEXUS

[![Julia](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/ispirovjr/NeoNEXUS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ispirovjr/NeoNEXUS.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/ispirovjr/NeoNEXUS.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/ispirovjr/NeoNEXUS.jl/actions/workflows/Documentation.yml)

**NeoNEXUS** is a modernized, modular, and high-performance implementation of the **Multi-scale Morphology Filter (MMF)** (later modernized as **NEXUS**) in Julia. It is designed to detect and classify multi-scale morphological structures in 3D scalar fields — including nodes, filaments, and walls.

## Introduction

In the era of "Big Data," scientific datasets are growing exponentially. Processing this data requires efficient, automated tools.

**NeoNEXUS** (Network Extraction via Unsupervised Scale-space) addresses this by providing a physically motivated, training-free morphological analysis framework. Originating from medical imaging, it identifies structures based on local geometry (Hessian eigenvalues) rather than simple density thresholds.

### API Documentation

For detailed API documentation, please refer to the [**API Documentation**](https://ispirovjr.github.io/NeoNEXUS.jl/).

### Key Philosophy
*   **Modularity**: Decoupled architecture allows users to easily swap filters, feature definitions, and thresholding logic.
*   **Performance**: Built in Julia, leveraging **multiple dispatch**, **functors**, and a **singleton pattern** for memory-efficient caching of heavy computations (FFTs, Eigenvalues).
*   **Domain Agnostic**: The core logic applies to any scalar field — cosmology, medical imaging, materials science, and more.

## Features

*   **Multiscale Analysis**: Detects structures at various smoothing scales to capture hierarchical geometry.
*   **Morphological Classifiers**:
    *   **Nodes**: Spherical collapse ($\lambda_1, \lambda_2, \lambda_3 < 0$).
    *   **Filaments**: Cylindrical collapse ($\lambda_1, \lambda_2 < 0$).
    *   **Sheets (Walls)**: Planar collapse ($\lambda_1 < 0$).
*   **Optimized Computation**:
    *   **HessianEigenCache**: Singleton structure to reuse eigenvalue computations across different features at the same scale/filter setting, preventing redundant FFTs.
    *   **Explicit Control**: `Read`/`Write` cache modes allow fine-grained memory management.

## Installation

```julia
using Pkg
Pkg.add("NeoNEXUS")
```

For development:
```julia
using Pkg
Pkg.develop(path="path/to/NeoNEXUS")
```

## Quick Start
Here is a functional example of detecting "Sheet" (Wall) structures in a random noise field.

```julia
using NeoNEXUS
using FFTW
using Statistics

# 1. Setup Grid & K-Space
N = 64
L = 1.0
dx = L / N
axis = range(-L/2 + dx/2, L/2 - dx/2, length=N)
kx = ky = kz = fftfreq(N) .* N .* 2π  # Standard circular frequency setup

# 2. Initialize Feature Detector
# We create a SheetFeature (Wall) detector
sheet = SheetFeature((N, N, N), kx, ky, kz)

# 3. Create/Load Data
# Using random noise for demonstration
density_field = randn(Float32, N, N, N)

# 4. Compute Significance Map
# properties(field) calls the functor which handles:
# - Hessian computation (using FFTs)
# - Eigenvalue decomposition
# - Signature calculation based on geometry
signature_map = sheet(density_field)

println("Max sheet signature: ", maximum(signature_map))

# 5. Access Results
# The feature object itself stores accumulated significance across scales if run in a loop
# (For a single scale, it matches the return value)
println("Stored significance max: ", maximum(sheet.significanceMap))
```

## Architecture & Methodology

NeoNEXUS reimagines the classic MMF/NEXUS pipeline with software engineering best practices.

### Workflow
The analysis follows a two-stage loop process:

1.  **Signature Computation Loop**:
    *   Iterate over **Scales** ($R_1, R_2, \dots$).
    *   Apply **Filter** (e.g., Gaussian) to the scalar field.
    *   Compute **Hessian Eigenvalues** ($\lambda_1 \le \lambda_2 \le \lambda_3$).
    *   Evaluate **Feature Signatures** (Response functions) for all requested features (Nodes, Filaments, Sheets).
    *   *Optimization*: The Hessian is computed once per scale and cached (`HessianEigenCache`) for all features.

2.  **Thresholding Loop**:
    *   After aggregating signatures across scales (max-pooling), a global noise threshold is applied.
    *   Features are masked hierarchically (e.g., Filaments mask Nodes) to ensure clean segmentation.

### Components
*   **Features (`AbstractMorphologicalFeature`)**: Defines the geometric signature (e.g., `SheetFeature`, `LineFeature`, `NodeFeature`). They act as functors `feature(field)` to compute maps.
*   **Filters (`AbstractScaleFilter`)**: Handles smoothing in Fourier space (e.g., `GaussianFourierFilter`).
*   **Hessian**: Core module for computing derivatives via FFTs. Uses explicit caching strategies (`Read`, `Write`, `None`) to manage memory.

## Use Cases

### Astrophysics
*   **Stellar Streams**: Detecting linear structures in galactic density fields for Galactic Archaeology.
*   **Phase Space Analysis**: Identifying structures in HR Diagrams or the Fundamental Plane of elliptical galaxies.

### Engineering & Materials
*   **Fracture Detection**: Inverting the density field allows detection of "negative" density features like cracks or voids in materials.
*   **Microstructure Analysis**: Identifying patterns in alloy compositions.

### Medicine
*   **Vascular Mapping**: Tracing blood vessels (tubular/filamentary structures).
*   **Tumor Detection**: Identifying nodular growths in 3D scans.

### Cosmology
For cosmology-specific extensions (tidal tensor classification, velocity divergence analysis), see **[CosmoNEXUS](https://github.com/ispirovjr/CosmoNEXUS.jl)**.

## Project Structure

```
NeoNEXUS/
├── src/
│   ├── NeoNEXUS.jl           # Module entry point & exports
│   ├── Types.jl              # Abstract types & Enums (CacheMode)
│   ├── Hessian.jl            # FFT-based Hessian & Eigenvalue computation
│   ├── Features.jl           # Sheet, Line, Node feature definitions
│   ├── Filters.jl            # Scale-space filters (Gaussian, Log-Gaussian, TopHat)
│   ├── Thresholds.jl         # Thresholding functions (volume, mass, ΔM², etc.)
│   ├── ConnectedComponents.jl # Connected component analysis & pruning
│   └── Runner.jl             # Pipeline orchestration (MMFClassic, NEXUSPlus)
├── test/
│   ├── runtests.jl           # Test suite entry point
│   ├── testHessians.jl       # Hessian computation tests
│   ├── testFeatureSignatureMap.jl  # Feature signature tests
│   ├── testFilters.jl        # Filter tests
│   ├── testThresholds.jl     # Thresholding function tests
│   ├── testConnectedComponents.jl  # Connected component tests
│   └── testOrchestration.jl  # Pipeline orchestration tests
└── demo/
    ├── orchestrationDemo.jl  # Demo comparing MMFClassic vs NEXUSPlus
    ├── quickStartDemo.jl     # Minimal quick-start example
    └── multithreadDemo.jl    # Multithreaded scaling demo
```

## Two Orchestration Methods

NeoNEXUS provides two complete pipelines:

### MMFClassic
The classic Multi-scale Morphology Filter (Aragón-Calvo et al. 2007):
- Linear Gaussian filtering at multiple scales
- Processes features: Sheets → Filaments → Nodes
- Plateau-based thresholding (component erosion stability)

### NEXUSPlus
The enhanced NEXUS+ method (Cautun et al. 2013):
- Linear filtering for Nodes, Log-Gaussian for Filaments/Walls
- Processes features: Nodes → Filaments → Walls
- Density-based thresholding for Nodes, ΔM² peak for others

Run the demo to compare both methods:
```bash
julia --project=. demo/orchestrationDemo.jl
```

## Acknowledgements

The author would like to thank the following people for their contributions and assistance with this project:

* **Rien van de Weygaert** - For Supervision of the project and guidance through the process.

* **Konstantin Spirov** - For Technical Guidance and Support, especially in the initial stages of the project.

* **Bram Alferink**, **Marius Cautun** and **Miguel Aragon-Calvo** - For Their previous implementations of the MMF and NEXUS+ algorithms, which served as a foundation for this project.



