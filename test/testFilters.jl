@testset "Filters" begin
    N = 32
    kx = ky = kz = FFTW.fftfreq(N) .* 2π

    @testset "GaussianFourierFilter Construction" begin
        # Test k-vector constructor
        filter1 = GaussianFourierFilter(kx, ky, kz)
        @test length(filter1.kx) == N
        @test length(filter1.ky) == N
        @test length(filter1.kz) == N

        # Test grid-size constructor
        filter2 = GaussianFourierFilter((N, N, N))
        @test filter2.kx ≈ kx
        @test filter2.ky ≈ ky
        @test filter2.kz ≈ kz
    end

    @testset "Standard Gaussian Filtering" begin
        filter = GaussianFourierFilter((N, N, N))

        # Create test field: constant field should remain constant
        constField = ones(Float32, N, N, N) .* 5.0f0
        filtered = filter(constField, 1.0)
        @test all(isapprox.(filtered, 5.0, atol=1e-5))

        # Random field should be smoothed (reduced variance)
        randomField = randn(Float32, N, N, N)
        filtered = filter(randomField, 2.0)
        @test std(filtered) < std(randomField)  # Smoothing reduces variance
    end

    @testset "Multiple Dispatch - Feature Types" begin
        filter = GaussianFourierFilter((N, N, N))
        testField = abs.(randn(Float32, N, N, N)) .+ 1.0f0  # Positive for log safety

        # Create feature instances for dispatch testing
        sheet = SheetFeature((N, N, N), kx, ky, kz)
        node = NodeFeature((N, N, N), kx, ky, kz)

        # Standard filtering (2 args)
        standardResult = filter(testField, 1.0)

        # Node dispatch should match standard filtering
        nodeResult = filter(testField, 1.0, node)
        @test nodeResult ≈ standardResult

        # Sheet dispatch should use log-filtering (different result)
        sheetResult = filter(testField, 1.0, sheet)
        @test !isapprox(sheetResult, standardResult, atol=1e-3)
        @test all(sheetResult .> 0)  # Log-filter preserves positivity
    end

    @testset "Log-Filtering Behavior" begin
        filter = GaussianFourierFilter((N, N, N))
        sheet = SheetFeature((N, N, N), kx, ky, kz)

        # Multiplicative field: should be smoothed in log-space
        # exp(a + b) structure is multiplicative
        baseField = ones(Float32, N, N, N) .* 10.0f0
        baseField[N÷2, N÷2, N÷2] = 100.0f0  # Spike

        logFiltered = filter(baseField, 1.0, sheet)

        # Log-filtering should preserve positivity
        @test all(logFiltered .> 0)

        # The spike should be smoothed but result should still be mostly positive
        @test maximum(logFiltered) < 100.0f0  # Spike is reduced
        @test minimum(logFiltered) > 1.0f0   # No zeros introduced
    end


    @testset "Gaussian Fourier Filter" begin
        filter = GaussianFourierFilter((N, N, N))
        field = randn(Float32, N, N, N)

        # Filter is now implemented - test it works
        filtered = filter(field, 2.0)
        @test size(filtered) == (N, N, N)
    end

end
