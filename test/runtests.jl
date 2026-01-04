using Test
using Statistics
using FFTW
using NeoNEXUS

#=
NeoNEXUS Test Suite
===================
Modularized test structure:
- types: Hierarchy and instantiation 
- hessian: Basic computation and memory
- hessian_physics: Physical correctness on analytic fields
- features: Feature map generation and integration
- pipeline: High-level orchestration
=#

@testset "NeoNEXUS" begin
    include("test_types.jl")
    include("test_hessian.jl")
    include("test_hessian_physics.jl")
    include("test_features.jl")
    include("test_pipeline.jl")
end
