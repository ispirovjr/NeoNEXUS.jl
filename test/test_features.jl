@testset "Feature Computation" begin

    # Re-using grid helpers locally since tests run in filtered scope
    function centeredGrid(N, L=1.0)
        dx = L/N
        range(-L/2 + dx/2, L/2 - dx/2; length=N)
    end

    N = 32
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)

    L = 1
    kr = fftfreq(N) .* N .* 2π / L
    kx = kr; ky = kr; kz = kr

    @testset "SheetFeature Integration" begin
        # Planar wall: x² -> should trigger sheet feature
        field = X.^2 .+ Y.*0 .+ Z.*0
        
        # Internal vectors kx, ky, kz are stored in float
        wall = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        
        testCache = NeoNEXUS.HessianEigenCache(N, N, N)
        wall(field, testCache, NeoNEXUS.Write)
        
        # 1. Cache populated?
        @test !all(testCache.λ3 .== 0)
        
        # 2. Map populated?
        @test size(wall.significanceMap) == (N, N, N)
        # Should have response where eigenvalues match sheet signature
        @test any(wall.significanceMap .> 0)
    end

    @testset "Cache Modes" begin
        field = X.^2 .+ Y.*0 .+ Z.*0
        kx_ = kx; ky_ = ky; kz_ = kz
        
        # Mode: None (Allocates internal cache)
        feature = NeoNEXUS.SheetFeature(size(field), kx_, ky_, kz_)
        sigMap = feature(field, nothing, NeoNEXUS.None)
        @test size(sigMap) == (N, N, N)

        # Mode: Write (Fills provided cache)
        cache = NeoNEXUS.HessianEigenCache(N, N, N)
        fill!(cache.λ1, 0)
        feature(field, cache, NeoNEXUS.Write)
        @test !all(cache.λ3 .== 0)

        # Mode: Read (Uses provided cache)
        feature2 = NeoNEXUS.SheetFeature(size(field), kx_, ky_, kz_)
        # Zero out field to prove we're reading cache, not computing from field
        field_zero = zeros(Float32, size(field))
        sigMap2 = feature2(field_zero, cache, NeoNEXUS.Read)
        @test isequal(sigMap2, sigMap) # Should match result from populated cache (including NaNs)
    end

end
