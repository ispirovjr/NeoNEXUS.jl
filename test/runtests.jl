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
- Orchestration: Pipeline orchestration (MMFClassic, NEXUSPlus)
- ConnectedComponents: Connected component analysis
=#

@testset "NeoNEXUS" begin
    include("testHessians.jl")
    include("testFeatureSignatureMap.jl")
    include("testFilters.jl")
    include("testOrchestration.jl")
    include("testThresholds.jl")
    include("testConnectedComponents.jl")
end

