using Test
using Statistics
using FFTW
using NeoNEXUS

#=
NeoNEXUS Test Suite
===================
Modularized test structure:
- Hessians: Physical correctness on analytic fields
- FeatureSignatureMap: Signature maps and cache mode validation
- Thresholds: Thresholding functions
- Filters: Filter functions
- TDD: for test driven development
- ConnectedComponents: Connected component analysis
- Pipeline: High-level orchestration
=#

@testset "NeoNEXUS" begin
    include("testHessians.jl")
    include("testFeatureSignatureMap.jl")
    include("testFilters.jl")
    include("testTdd.jl")
    include("testThresholds.jl")
    include("testConnectedComponents.jl")
end
