# Tests for Hessian eigenvalue computation
@testset "Hessian" begin

    #==========================================================================
    Hessian eigenvalues reveal local geometry:
        λ1 ≤ λ2 ≤ λ3 (sorted)
        
        Sheet:    λ1 ≈ 0,  λ2 < 0,  λ3 < 0
        Filament: λ1 < 0,  λ2 < 0,  λ3 ≈ 0  
        Blob:     λ1 < 0,  λ2 < 0,  λ3 < 0
    ==========================================================================#

    # Helper: create wavenumber grids
    function make_kgrids(Nx, Ny, Nz)
        kx = [2π * i / Nx for i in 0:Nx-1, _ in 1:Ny, _ in 1:Nz]
        ky = [2π * j / Ny for _ in 1:Nx, j in 0:Ny-1, _ in 1:Nz]
        kz = [2π * k / Nz for _ in 1:Nx, _ in 1:Ny, k in 0:Nz-1]
        return kx, ky, kz
    end

    @testset "HessianEigenCache Construction" begin
        Nx, Ny, Nz = 8, 8, 8
        cache = NeoNEXUS.HessianEigenCache(Nx, Ny, Nz)
        
        @test size(cache.λ1) == (Nx, Ny, Nz)
        @test size(cache.λ2) == (Nx, Ny, Nz)
        @test size(cache.λ3) == (Nx, Ny, Nz)
        @test eltype(cache.λ1) == Float32
    end

    @testset "Allocating Computation" begin
        Nx, Ny, Nz = 8, 8, 8
        field = randn(Float32, Nx, Ny, Nz)
        kx, ky, kz = make_kgrids(Nx, Ny, Nz)
        
        cache = NeoNEXUS.computeHessianEigenvalues(field, kx, ky, kz)
        
        @test cache isa NeoNEXUS.HessianEigenCache
        @test size(cache.λ1) == size(field)
    end

    @testset "Eigenvalue Ordering" begin
        Nx, Ny, Nz = 4, 4, 4
        field = randn(Float32, Nx, Ny, Nz)
        kx, ky, kz = make_kgrids(Nx, Ny, Nz)
        
        cache = NeoNEXUS.computeHessianEigenvalues(field, kx, ky, kz)
        
        @test all(cache.λ1 .<= cache.λ2)
        @test all(cache.λ2 .<= cache.λ3)
    end

    @testset "In-place Computation" begin
        Nx, Ny, Nz = 4, 4, 4
        field1 = randn(Float32, Nx, Ny, Nz)
        field2 = randn(Float32, Nx, Ny, Nz)
        kx, ky, kz = make_kgrids(Nx, Ny, Nz)
        
        cache = NeoNEXUS.HessianEigenCache(Nx, Ny, Nz)
        
        NeoNEXUS.computeHessianEigenvalues!(field1, kx, ky, kz, cache)
        λ1_first = copy(cache.λ1)
        
        NeoNEXUS.computeHessianEigenvalues!(field2, kx, ky, kz, cache)
        
        @test λ1_first != cache.λ1
    end

end
