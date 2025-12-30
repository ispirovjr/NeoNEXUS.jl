using Test
using Statistics
using NeoNEXUS
using FFTW

function centeredGrid(N, L=1.0)
    # Centered symmetric grid for periodicity: x[i] + x[N-i+1] = 0
    dx = L/N
    x = range(-L/2 + dx/2, L/2 - dx/2; length=N)
    return x
end


const N = 32
x = centeredGrid(N)
y = centeredGrid(N)
z = centeredGrid(N)

X = reshape(x, :, 1, 1)
Y = reshape(y, 1, :, 1)
Z = reshape(z, 1, 1, :)

# Fourier wave numbers (consistent with FFT ordering)
L = 1

kr = fftfreq(N) .* N .* 2π / L

kx = kr  # 1D vector
ky = kr  # 1D vector
kz = kr  # 1D vector

@testset "Hessian Eigenvalue Tests" begin

    @testset "Constant Field" begin
        field = ones(Float32, N, N, N)
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Linear Field (Periodic)" begin
        # Use abs() to ensure periodicity - no jump at boundaries
        field = abs.(X)
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        
        # For |x|, derivative is ±1 (except at x=0), so second derivative ≈ 0
        # Near origin there are numerical artifacts, so use median
        @test isapprox(median(abs.(cache.λ1)), 0.0; atol=0.2)
        @test isapprox(median(abs.(cache.λ2)), 0.0; atol=0.2)
        @test isapprox(median(abs.(cache.λ3)), 0.0; atol=0.2)
    end

    @testset "Isotropic Quadratic (Node)" begin
        field = X.^2 .+ Y.^2 .+ Z.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        # Use median to avoid boundary artifacts
        @test isapprox(median(cache.λ1), 2.0; atol=0.15)
        @test isapprox(median(cache.λ2), 2.0; atol=0.15)
        @test isapprox(median(cache.λ3), 2.0; atol=0.15)
    end

    @testset "Cylindrical Filament" begin
        field = X.^2 .+ Y.^2 .+Z*0
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 2.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end

    @testset "Planar Wall" begin
        field = X.^2 .+ Y*0 .+ Z*0
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 0.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end
end

@testset "Feature Signature Map Tests" begin
    
    @testset "SheetFeature - Planar Wall" begin
        # Create planar wall field (varies only in x)
        field = X.^2 .+ Y*0 .+ Z*0
        
        # Initialize feature with 1D k-vectors
        wall = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        
        # Test with cache in Write mode
        testCache = NeoNEXUS.HessianEigenCache(N, N, N)
        wall(field, testCache, NeoNEXUS.Write)
        
        # Cache should be populated (non-zero eigenvalues)
        @test !all(testCache.λ1 .== 0)
        @test !all(testCache.λ2 .== 0) || !all(testCache.λ3 .== 0)
        
        # Significance map should be populated
        @test size(wall.significanceMap) == (N, N, N)
        @test any(wall.significanceMap .> 0)  # Some voxels should have non-zero significance
    end
    
    @testset "Cache Mode - None" begin
        field = X.^2 .+ Y*0 .+ Z*0  # Full 3D grid
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        
        # Test None mode (creates and uses own internal cache)
        sigMap = feature(field, nothing, NeoNEXUS.None)
        @test size(sigMap) == (N, N, N)
        @test size(feature.significanceMap) == (N, N, N)
    end
    
    @testset "Cache Mode - Write" begin
        field = X.^2 .+ Y*0 .+ Z*0  # Full 3D grid
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        cache = NeoNEXUS.HessianEigenCache(N, N, N)
        
        # Test Write mode (uses provided cache and fills it)
        sigMap = feature(field, cache, NeoNEXUS.Write)
        @test size(sigMap) == (N, N, N)
        @test !all(cache.λ1 .== 0)  # Cache should be populated
    end
    
    @testset "Cache Mode - Read" begin
        field = X.^2 .+ Y*0 .+ Z*0  # Full 3D grid
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        cache = NeoNEXUS.HessianEigenCache(N, N, N)
        
        # Pre-populate cache with Write mode
        _ = feature(field, cache, NeoNEXUS.Write)
        
        # Create new feature and test Read mode (reads from cache without recomputing)
        feature2 = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        sigMap = feature2(field, cache, NeoNEXUS.Read)
        @test size(sigMap) == (N, N, N)
    end
end
