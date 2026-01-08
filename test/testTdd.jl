# TDD Future Development Tests
# These tests represent expected functionality that is not yet implemented.
# They are marked as @test_broken and will serve as a guide for ongoing development.

@testset "TDD Future Features" begin

    # Shared grid setup for tests
    function centeredGrid(N, L=1.0)
        dx = L / N
        range(-L / 2 + dx / 2, L / 2 - dx / 2; length=N)
    end

    N = 16
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)
    kr = fftfreq(N) .* N .* 2π
    kx = kr
    ky = kr
    kz = kr

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

    @testset "Gaussian Fourier Filter" begin
        filter = GaussianFourierFilter((N, N, N))
        field = randn(Float32, N, N, N)

        # Filter is now implemented - test it works
        filtered = filter(field, 2.0)
        @test size(filtered) == (N, N, N)
    end

    @testset "Runner Pipeline Orchestration" begin
        filter = GaussianFourierFilter((N, N, N))
        features = AbstractFeature[SheetFeature((N, N, N), kx, ky, kz)]
        scales = [1.0, 2.0]
        runner = NeoNEXUSRunner(filter, features, scales)

        field = randn(Float32, N, N, N)

        # Current implementation of run() is empty/no-op
        @test_broken begin
            NeoNEXUS.run(runner, field, NeoNEXUS.Default)
            # Check if significance maps are actually populated
            any(features[1].significanceMap .> 0)
        end
    end

end
