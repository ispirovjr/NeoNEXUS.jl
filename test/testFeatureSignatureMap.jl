# Tests for feature signature maps and cache mode integration
@testset "Feature Signature Map Tests" begin

    # Grid setup helpers
    function centeredGrid(N, L=1.0)
        dx = L / N
        range(-L / 2 + dx / 2, L / 2 - dx / 2; length=N)
    end

    N = 32
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)

    L = 1
    kr = fftfreq(N) .* N .* 2π / L
    kx = kr
    ky = kr
    kz = kr

    @testset "SheetFeature - Planar Wall" begin
        # Create planar wall field (varies only in x)
        field = X .^ 2 .+ Y * 0 .+ Z * 0

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
        field = X .^ 2 .+ Y * 0 .+ Z * 0
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)

        # Test None mode (creates and uses own internal cache)
        sigMap = feature(field, nothing, NeoNEXUS.None)
        @test size(sigMap) == (N, N, N)
        @test size(feature.significanceMap) == (N, N, N)
    end

    @testset "Cache Mode - Write" begin
        field = X .^ 2 .+ Y * 0 .+ Z * 0
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        cache = NeoNEXUS.HessianEigenCache(N, N, N)

        # Test Write mode (uses provided cache and fills it)
        sigMap = feature(field, cache, NeoNEXUS.Write)
        @test size(sigMap) == (N, N, N)
        @test !all(cache.λ1 .== 0)  # Cache should be populated
    end

    @testset "Cache Mode - Read" begin
        field = X .^ 2 .+ Y * 0 .+ Z * 0
        feature = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        cache = NeoNEXUS.HessianEigenCache(N, N, N)

        # Pre-populate cache with Write mode
        _ = feature(field, cache, NeoNEXUS.Write)

        # Create new feature and test Read mode (reads from cache without recomputing)
        feature2 = NeoNEXUS.SheetFeature(size(field), kx, ky, kz)
        sigMap = feature2(field, cache, NeoNEXUS.Read)
        @test size(sigMap) == (N, N, N)
    end

    @testset "Filament Detection (LineFeature)" begin
        # Cylindrical overdensity along Z-axis: X² + Y² creates compression in X,Y
        # This gives λ1, λ2 < 0 (compression perpendicular to cylinder axis)
        field = X .^ 2 .+ Y .^ 2 .+ Z .* 0
        line = LineFeature((N, N, N), kx, ky, kz)

        sigMap = line(field)
        @test size(sigMap) == (N, N, N)
        @test any(sigMap .> 0)  # Should detect filament signature
    end

    @testset "Node Detection (NodeFeature)" begin
        # Isotropic overdensity: X² + Y² + Z² creates compression in all directions
        # This gives λ1, λ2, λ3 all < 0
        field = X .^ 2 .+ Y .^ 2 .+ Z .^ 2
        node = NodeFeature((N, N, N), kx, ky, kz)

        sigMap = node(field)
        @test size(sigMap) == (N, N, N)
        @test any(sigMap .> 0)  # Should detect node signature
    end


end
