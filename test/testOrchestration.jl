# Orchestration tests for NeoNEXUS pipelines
# Tests MMFClassic and NEXUSPlus runners

@testset "Orchestration" begin

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


    @testset "MMFClassic Pipeline" begin
        filter = GaussianFourierFilter((N, N, N))
        features = AbstractFeature[SheetFeature((N, N, N), kx, ky, kz)]
        scales = [1.0, 2.0]
        runner = MMFClassic(filter, features, scales)

        field = randn(Float32, N, N, N)

        # Run the pipeline
        NeoNEXUS.run(runner, field)

        # Check if significance maps are populated
        @test any(features[1].significanceMap .> 0)
    end

    @testset "NEXUSPlus Pipeline" begin
        filter = GaussianFourierFilter((N, N, N))
        node = NodeFeature((N, N, N), kx, ky, kz)
        filament = LineFeature((N, N, N), kx, ky, kz)
        wall = SheetFeature((N, N, N), kx, ky, kz)
        scales = [1.0, 2.0]
        runner = NEXUSPlus(filter, node, filament, wall, scales)

        # Use positive density (required for log filtering)
        field = abs.(randn(Float32, N, N, N)) .+ 1f0

        # Run the pipeline
        thresholds = NeoNEXUS.run(runner, field)

        # Check significance maps populated
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)

        # Check threshold maps have correct size
        @test size(node.thresholdMap) == (N, N, N)
        @test size(filament.thresholdMap) == (N, N, N)
        @test size(wall.thresholdMap) == (N, N, N)

        # Check thresholds returned (nodeThres is computed, not fixed)
        @test thresholds.nodeThres > 0f0
    end

end
