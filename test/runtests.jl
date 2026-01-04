using Test
using Statistics
using FFTW
using NeoNEXUS

#=
NeoNEXUS Test Suite
===================
Modularized test structure:
- hessian_physics: Physical correctness on analytic fields
- feature_signature_map: Signature maps and cache mode validation
- pipeline: High-level orchestration
=#

@testset "NeoNEXUS" begin
    include("test_hessian_physics.jl")
    include("test_feature_signature_map.jl")
end
